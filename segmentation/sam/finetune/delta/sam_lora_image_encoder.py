import math
from torch_runtime import nn, torch
from torch_runtime import Parameter
from ...backbones.segment_anything.modeling import Sam


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_k: nn.Module | None,
            linear_b_k: nn.Module | None,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_k = linear_a_k
        self.linear_b_k = linear_b_k
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        if self.linear_a_q is not None and self.linear_b_q is not None:
            qkv[:, :, :, : self.dim] += self.linear_b_q(self.linear_a_q(x))
        if self.linear_a_k is not None and self.linear_b_k is not None:
            qkv[:, :, :, self.dim : 2 * self.dim] += self.linear_b_k(self.linear_a_k(x))
        if self.linear_a_v is not None and self.linear_b_v is not None:
            qkv[:, :, :, -self.dim:] += self.linear_b_v(self.linear_a_v(x))
        return qkv


class _LoRA_linear(nn.Module):
    def __init__(self, linear: nn.Module, linear_a: nn.Module, linear_b: nn.Module):
        super().__init__()
        self.linear = linear
        self.linear_a = linear_a
        self.linear_b = linear_b

    def forward(self, x):
        return self.linear(x) + self.linear_b(self.linear_a(x))


class LoRA_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, sam_model: Sam, r: int, lora_layer=None, lora_targets=None):
        super(LoRA_Sam, self).__init__()

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.blocks)))  # Only apply lora to the image encoder by default
        self.lora_targets = set(lora_targets or {"q", "v"}) or {"q", "v"}
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False
        for param in sam_model.prompt_encoder.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = w_b_linear_q = w_a_linear_k = w_b_linear_k = w_a_linear_v = w_b_linear_v = None
            if "q" in self.lora_targets:
                w_a_linear_q = nn.Linear(self.dim, r, bias=False)
                w_b_linear_q = nn.Linear(r, self.dim, bias=False)
                self.w_As.append(w_a_linear_q)
                self.w_Bs.append(w_b_linear_q)
            if "k" in self.lora_targets:
                w_a_linear_k = nn.Linear(self.dim, r, bias=False)
                w_b_linear_k = nn.Linear(r, self.dim, bias=False)
                self.w_As.append(w_a_linear_k)
                self.w_Bs.append(w_b_linear_k)
            if "v" in self.lora_targets:
                w_a_linear_v = nn.Linear(self.dim, r, bias=False)
                w_b_linear_v = nn.Linear(r, self.dim, bias=False)
                self.w_As.append(w_a_linear_v)
                self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_k,
                w_b_linear_k,
                w_a_linear_v,
                w_b_linear_v,
            )
            if "mlp" in self.lora_targets:
                for name in ("lin1", "lin2"):
                    linear = getattr(blk.mlp, name)
                    w_a_linear = nn.Linear(linear.in_features, r, bias=False)
                    w_b_linear = nn.Linear(r, linear.out_features, bias=False)
                    self.w_As.append(w_a_linear)
                    self.w_Bs.append(w_b_linear)
                    setattr(blk.mlp, name, _LoRA_linear(linear, w_a_linear, w_b_linear))
        self.reset_parameters()
        self.sam = sam_model

    def save_delta_parameters(self, filename: str) -> None:
        r"""
        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half   
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()  
        for key, value in state_dict.items():  
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value

        metadata = {
            "__lora_layers__": list(self.lora_layer),
            "__lora_targets__": sorted(self.lora_targets),
        }
        merged_dict = {**metadata, **a_tensors, **b_tensors, **prompt_encoder_tensors, **mask_decoder_tensors} 
        torch.save(merged_dict, filename)

    def load_delta_parameters(self, filename: str) -> None:
        r"""
        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        try:
            state_dict = torch.load(filename, map_location="cpu", weights_only=True)
        except Exception:
            state_dict = torch.load(filename, map_location="cpu", weights_only=False)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key].to(device=w_A_linear.weight.device, dtype=w_A_linear.weight.dtype)
            with torch.no_grad():
                w_A_linear.weight.copy_(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key].to(device=w_B_linear.weight.device, dtype=w_B_linear.weight.dtype)
            with torch.no_grad():
                w_B_linear.weight.copy_(saved_tensor)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # load prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)
        self.sam.load_state_dict(sam_dict)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, batched_input, multimask_output, image_size, boxes=None, points=None):
        return self.sam(batched_input, multimask_output, image_size, boxes=boxes, points=points)
