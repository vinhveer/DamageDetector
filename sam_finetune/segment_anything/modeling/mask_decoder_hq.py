from torch_runtime import F, nn, torch
from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoderHQ(nn.Module):
    requires_image_features = True
    hq_trainable_prefixes = (
        "hf_token",
        "hf_mlp",
        "compress_vit_feat",
        "embedding_encoder",
        "embedding_maskfeature",
    )
    balanced_trainable_prefixes = hq_trainable_prefixes + (
        "iou_token",
        "mask_tokens",
        "iou_prediction_head",
        "output_hypernetworks_mlps.0",
    )

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        vit_dim: int = 1024,
        output_scale_factor: int = 4,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.output_scale_factor = max(1, int(output_scale_factor))

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for _ in range(self.num_mask_tokens)]
        )
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        self.hf_token = nn.Embedding(1, transformer_dim)
        self.hf_mlp = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
        self.num_mask_tokens = self.num_mask_tokens + 1

        self.compress_vit_feat = nn.Sequential(
            nn.ConvTranspose2d(vit_dim, transformer_dim, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 8, kernel_size=2, stride=2),
        )
        self.embedding_encoder = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
        )
        self.embedding_maskfeature = nn.Sequential(
            nn.Conv2d(transformer_dim // 8, transformer_dim // 4, 3, 1, 1),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.Conv2d(transformer_dim // 4, transformer_dim // 8, 3, 1, 1),
        )
        self.reset_hq_parameters()

    def reset_hq_parameters(self) -> None:
        self._zero_module(self.compress_vit_feat[-1])
        self._zero_module(self.embedding_encoder[-1])
        self._zero_module(self.embedding_maskfeature[-1])

    def initialize_from_sam_state_dict(self, state_dict: dict) -> None:
        if not isinstance(state_dict, dict):
            return
        self._copy_named_parameter("iou_token.weight", state_dict)
        self._copy_mask_token_from_state(state_dict, dst_index=0, src_index=0)
        self._copy_hypernet_from_state(dst_mlp=self.output_hypernetworks_mlps[0], state_dict=state_dict, src_index=0)
        self._copy_iou_head_from_state(state_dict)

        with torch.no_grad():
            self.hf_token.weight.copy_(self.mask_tokens.weight[:1])
        self._copy_module_parameters(self.hf_mlp, self.output_hypernetworks_mlps[0])
        self.reset_hq_parameters()

    def set_hq_only_trainable(self) -> None:
        self.set_trainable_mode("hq_only")

    def set_trainable_mode(self, mode: str = "balanced") -> None:
        mode = str(mode or "balanced").strip().lower()
        for _, parameter in self.named_parameters():
            parameter.requires_grad = False
        if mode == "hq_only":
            trainable_prefixes = self.hq_trainable_prefixes
        else:
            trainable_prefixes = self.balanced_trainable_prefixes

        for prefix in trainable_prefixes:
            module = self._resolve_module(prefix)
            if module is None:
                continue
            for _, parameter in module.named_parameters():
                parameter.requires_grad = True

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        hq_token_only: bool = False,
        interm_embeddings: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if interm_embeddings is None:
            raise ValueError("MaskDecoderHQ requires intermediate image encoder embeddings.")
        vit_features = interm_embeddings[0].permute(0, 3, 1, 2)
        hq_features = self.embedding_encoder(image_embeddings) + self.compress_vit_feat(vit_features)

        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            hq_features=hq_features,
        )

        if multimask_output and masks.shape[1] > 1:
            mask_slice = slice(1, self.num_mask_tokens - 1)
            iou_pred = iou_pred[:, mask_slice]
            iou_pred, max_iou_idx = torch.max(iou_pred, dim=1)
            iou_pred = iou_pred.unsqueeze(1)
            masks_multi = masks[:, mask_slice, :, :]
            masks_sam = masks_multi[torch.arange(masks_multi.size(0), device=masks_multi.device), max_iou_idx].unsqueeze(1)
        else:
            mask_slice = slice(0, 1)
            iou_pred = iou_pred[:, mask_slice]
            masks_sam = masks[:, mask_slice]

        masks_hq = masks[:, slice(self.num_mask_tokens - 1, self.num_mask_tokens)]
        masks = masks_hq if hq_token_only else masks_sam + masks_hq
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        hq_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight, self.hf_token.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        if image_pe.shape[0] != tokens.shape[0]:
            pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        else:
            pos_src = image_pe
        if hq_features.shape[0] != tokens.shape[0]:
            hq_features = torch.repeat_interleave(hq_features, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding_sam = self.output_upscaling(src)
        upscaled_embedding_hq = self.embedding_maskfeature(upscaled_embedding_sam) + hq_features

        if self.output_scale_factor > 1:
            target_hw = (
                upscaled_embedding_sam.shape[2] * self.output_scale_factor,
                upscaled_embedding_sam.shape[3] * self.output_scale_factor,
            )
            upscaled_embedding_sam = F.interpolate(
                upscaled_embedding_sam, size=target_hw, mode="bilinear", align_corners=False
            )
            upscaled_embedding_hq = F.interpolate(
                upscaled_embedding_hq, size=target_hw, mode="bilinear", align_corners=False
            )

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            if i < self.num_mask_tokens - 1:
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            else:
                hyper_in_list.append(self.hf_mlp(mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        b, c, h, w = upscaled_embedding_sam.shape
        masks_sam = (hyper_in[:, : self.num_mask_tokens - 1] @ upscaled_embedding_sam.view(b, c, h * w)).view(b, -1, h, w)
        masks_hq = (hyper_in[:, self.num_mask_tokens - 1 :] @ upscaled_embedding_hq.view(b, c, h * w)).view(b, -1, h, w)
        masks = torch.cat([masks_sam, masks_hq], dim=1)
        iou_pred = self.iou_prediction_head(iou_token_out)
        return masks, iou_pred

    def _copy_named_parameter(self, name: str, state_dict: dict) -> None:
        source = state_dict.get(f"mask_decoder.{name}")
        target = self.state_dict().get(name)
        if not isinstance(source, torch.Tensor) or not isinstance(target, torch.Tensor):
            return
        if tuple(source.shape) != tuple(target.shape):
            return
        own_state = self.state_dict()
        with torch.no_grad():
            own_state[name].copy_(source)

    def _copy_mask_token_from_state(self, state_dict: dict, *, dst_index: int, src_index: int) -> None:
        source = state_dict.get("mask_decoder.mask_tokens.weight")
        if not isinstance(source, torch.Tensor):
            return
        if source.ndim != 2 or self.mask_tokens.weight.ndim != 2:
            return
        if src_index >= source.shape[0] or dst_index >= self.mask_tokens.weight.shape[0]:
            return
        with torch.no_grad():
            self.mask_tokens.weight[dst_index].copy_(source[src_index])

    def _copy_hypernet_from_state(self, dst_mlp: nn.Module, state_dict: dict, *, src_index: int) -> None:
        if not hasattr(dst_mlp, "layers"):
            return
        for layer_index, dst_layer in enumerate(dst_mlp.layers):
            weight_key = f"mask_decoder.output_hypernetworks_mlps.{src_index}.layers.{layer_index}.weight"
            bias_key = f"mask_decoder.output_hypernetworks_mlps.{src_index}.layers.{layer_index}.bias"
            weight = state_dict.get(weight_key)
            bias = state_dict.get(bias_key)
            if isinstance(weight, torch.Tensor) and tuple(weight.shape) == tuple(dst_layer.weight.shape):
                with torch.no_grad():
                    dst_layer.weight.copy_(weight)
            if isinstance(bias, torch.Tensor) and tuple(bias.shape) == tuple(dst_layer.bias.shape):
                with torch.no_grad():
                    dst_layer.bias.copy_(bias)

    def _copy_iou_head_from_state(self, state_dict: dict) -> None:
        for layer_index, dst_layer in enumerate(self.iou_prediction_head.layers):
            weight_key = f"mask_decoder.iou_prediction_head.layers.{layer_index}.weight"
            bias_key = f"mask_decoder.iou_prediction_head.layers.{layer_index}.bias"
            weight = state_dict.get(weight_key)
            bias = state_dict.get(bias_key)
            if isinstance(weight, torch.Tensor):
                if tuple(weight.shape) == tuple(dst_layer.weight.shape):
                    with torch.no_grad():
                        dst_layer.weight.copy_(weight)
                elif weight.ndim == 2 and dst_layer.weight.ndim == 2 and weight.shape[1] == dst_layer.weight.shape[1]:
                    with torch.no_grad():
                        dst_layer.weight.copy_(weight[: dst_layer.weight.shape[0]])
            if isinstance(bias, torch.Tensor):
                if tuple(bias.shape) == tuple(dst_layer.bias.shape):
                    with torch.no_grad():
                        dst_layer.bias.copy_(bias)
                elif bias.ndim == 1 and dst_layer.bias.ndim == 1:
                    with torch.no_grad():
                        dst_layer.bias.copy_(bias[: dst_layer.bias.shape[0]])

    @staticmethod
    def _copy_module_parameters(dst_module: nn.Module, src_module: nn.Module) -> None:
        with torch.no_grad():
            for dst_param, src_param in zip(dst_module.parameters(), src_module.parameters()):
                if tuple(dst_param.shape) == tuple(src_param.shape):
                    dst_param.copy_(src_param)

    @staticmethod
    def _zero_module(module: nn.Module) -> None:
        with torch.no_grad():
            if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
                module.weight.zero_()
            if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor) and module.bias is not None:
                module.bias.zero_()

    def _resolve_module(self, module_path: str) -> nn.Module | None:
        module: nn.Module | None = self
        for token in str(module_path).split("."):
            if module is None:
                return None
            if token.isdigit():
                try:
                    module = module[int(token)]
                except Exception:
                    return None
            else:
                module = getattr(module, token, None)
        return module


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
