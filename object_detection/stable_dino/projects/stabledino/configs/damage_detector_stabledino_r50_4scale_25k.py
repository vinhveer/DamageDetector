"""StableDINO R50 config for CRACK500 25k-step fine-tuning.

This schedule keeps the 30k recipe proportions but moves decay milestones
inside a 25k run, which is easier to finish on Kaggle T4 sessions.
"""

from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler

from object_detection.stable_dino.detrex_compat import get_detrex_config

from .data.coco_instance_seg import dataloader
from .models.stabledino_r50 import model

optimizer = get_detrex_config("common/optim.py").AdamW
train = get_detrex_config("common/train.py").train

train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/damage_detector_stabledino_r50_4scale_25k"
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

optimizer.lr = 2e-4
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

dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 4
dataloader.evaluator.output_dir = train.output_dir
