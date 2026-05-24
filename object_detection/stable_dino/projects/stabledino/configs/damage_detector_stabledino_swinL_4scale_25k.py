"""StableDINO Swin-L config for damage detector 25k-step fine-tuning.

Backbone: Swin-L 384 (ImageNet-22K pretrained, embed_dim=192, depths=(2,2,18,2)).
Stronger than R50 — use when VRAM >= 24 GB (A100/A6000).
Download backbone: hf download microsoft/swin-large-patch4-window12-384-22k --local-dir /path/to/swin_large_384_22k
Pass to --init-checkpoint.
"""

from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import SwinTransformer
from detectron2.solver import WarmupParamScheduler

from object_detection.stable_dino.detrex_compat import get_detrex_config

from .data.coco_detection import dataloader
from .models.stabledino_r50 import model

optimizer = get_detrex_config("common/optim.py").AdamW
train = get_detrex_config("common/train.py").train

# ── Backbone: Swin-L 384 ──────────────────────────────────────────────────────
model.backbone = L(SwinTransformer)(
    pretrain_img_size=384,
    embed_dim=192,
    depths=(2, 2, 18, 2),
    num_heads=(6, 12, 24, 48),
    window_size=12,
    out_indices=(1, 2, 3),
)
model.neck.input_shapes = {
    "p1": ShapeSpec(channels=384),
    "p2": ShapeSpec(channels=768),
    "p3": ShapeSpec(channels=1536),
}
model.neck.in_features = ["p1", "p2", "p3"]

# ── Training ──────────────────────────────────────────────────────────────────
train.init_checkpoint = ""
train.output_dir = "./output/damage_detector_stabledino_swinL_4scale_25k"
train.max_iter = 25000
train.eval_period = 2000
train.log_period = 50
train.seed = 60
train.checkpointer.period = 2000
train.test_with_nms = 0.5
train.best_checkpointer = dict(
    enabled=True,
    metric="bbox/AP",
    mode="max",
)
train.finetune_checkpoint = dict(
    path="",
    ignore_prefixes=["class_embed", "label_enc"],
    ignore_shape_mismatch=True,
)

train.clip_grad.enabled = True
train.clip_grad.params.max_norm = 0.1
train.clip_grad.params.norm_type = 2

train.device = "cuda"
model.device = train.device

# ── Optimizer ─────────────────────────────────────────────────────────────────
optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=[17500, 22500, 25000],
    ),
    warmup_length=1000 / 25000,
    warmup_method="linear",
    warmup_factor=0.001,
)

# ── Dataloader ────────────────────────────────────────────────────────────────
dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 4
dataloader.evaluator.output_dir = train.output_dir
