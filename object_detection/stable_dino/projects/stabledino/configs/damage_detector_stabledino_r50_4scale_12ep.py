from object_detection.stable_dino.detrex_compat import get_detrex_config

from .data.coco_detection import dataloader
from .models.stabledino_r50 import model

# Optim/schedule/train boilerplate from detrex.
optimizer = get_detrex_config("common/optim.py").AdamW
lr_multiplier = get_detrex_config("common/coco_schedule.py").lr_multiplier_12ep
train = get_detrex_config("common/train.py").train

# Defaults (overridden by object_detection.stable_dino.dataset_config via LazyConfig opts).
train.init_checkpoint = "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
train.output_dir = "./output/damage_detector_stabledino_r50_4scale_12ep"
train.max_iter = 90000
train.eval_period = 1000
train.log_period = 20
train.seed = 60
train.checkpointer.period = 1000
train.test_with_nms = 0.80
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

optimizer.lr = 1e-4
optimizer.betas = (0.9, 0.999)
optimizer.weight_decay = 1e-4
optimizer.params.lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1

dataloader.train.num_workers = 4
dataloader.train.total_batch_size = 16
dataloader.evaluator.output_dir = train.output_dir
