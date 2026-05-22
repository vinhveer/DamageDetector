#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
from collections import OrderedDict
import logging
import pickle
import os
import sys
import time
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

from contextlib import contextmanager, nullcontext

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

logger = logging.getLogger("detrex")


class Trainer(SimpleTrainer):
    """
    We've combine Simple and AMP Trainer together.
    """

    def __init__(
        self,
        model,
        dataloader,
        optimizer,
        amp=False,
        clip_grad_params=None,
        grad_scaler=None,
    ):
        super().__init__(model=model, data_loader=dataloader, optimizer=optimizer)

        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        if amp:
            if grad_scaler is None:
                from torch.cuda.amp import GradScaler

                grad_scaler = GradScaler()
            self.grad_scaler = grad_scaler

        # set True to use amp training
        self.amp = amp

        # gradient clip hyper-params
        self.clip_grad_params = clip_grad_params

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self.model.training, "[Trainer] model was changed to eval mode!"
        use_cuda_amp = bool(self.amp and torch.cuda.is_available())
        autocast = torch.amp.autocast if hasattr(torch, "amp") else None

        start = time.perf_counter()
        """
        If you want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        autocast_context = (
            autocast(device_type="cuda", enabled=True)
            if use_cuda_amp and autocast is not None
            else nullcontext()
        )
        with autocast_context:
            """
            If you want to do something with the losses, you can wrap the model.
            """
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        """
        If you need to accumulate gradients or do something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()

        if use_cuda_amp:
            self.grad_scaler.scale(losses).backward()
            if self.clip_grad_params is not None:
                self.grad_scaler.unscale_(self.optimizer)
                self.clip_grads(self.model.parameters())
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            losses.backward()
            if self.clip_grad_params is not None:
                self.clip_grads(self.model.parameters())
            self.optimizer.step()

        self._write_metrics(loss_dict, data_time)

    def clip_grads(self, params):
        params = list(filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                parameters=params,
                **self.clip_grad_params,
            )

def _remove_ddp(model):
    from torch.nn.parallel import DistributedDataParallel

    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


@contextmanager
def tune_nms_threshold_and_restore(model, temp_threshold):
    """Apply ema stored in `model` to model and returns a function to restore
    the weights are applied
    """
    model = _remove_ddp(model)
    old_nms_threshold = model.nms_thresh
    print(f"Changing NMS threshold from {old_nms_threshold} to {temp_threshold}")
    model.nms_thresh = temp_threshold

    yield model

    print(f"Restoring NMS threshold from {temp_threshold} to {old_nms_threshold}")
    model.nms_thresh = old_nms_threshold


def _extract_checkpoint_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model", "state_dict", "module"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def _normalize_checkpoint_key(key):
    for prefix in ("module.", "model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _should_ignore_key(key, ignore_prefixes):
    return any(key == prefix or key.startswith(f"{prefix}.") for prefix in ignore_prefixes)


def _torch_load_checkpoint(checkpoint_path):
    try:
        return torch.load(checkpoint_path, map_location="cpu")
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        logger.warning(
            "Retrying fine-tune checkpoint load with weights_only=False for trusted checkpoint: %s",
            checkpoint_path,
        )
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def load_finetune_checkpoint(cfg, model):
    finetune_cfg = cfg.train.get("finetune_checkpoint") or {}
    checkpoint_path = str(finetune_cfg.get("path", "")).strip()
    if not checkpoint_path:
        return
    ignore_prefixes = tuple(str(prefix) for prefix in finetune_cfg.get("ignore_prefixes", []))
    ignore_shape_mismatch = bool(finetune_cfg.get("ignore_shape_mismatch", True))
    target_model = _remove_ddp(model)
    model_state = target_model.state_dict()
    checkpoint = _torch_load_checkpoint(checkpoint_path)
    checkpoint_state = _extract_checkpoint_state(checkpoint)
    if not isinstance(checkpoint_state, dict):
        raise ValueError(f"Fine-tune checkpoint has no state dict: {checkpoint_path}")

    load_state = OrderedDict()
    skipped = []
    for raw_key, value in checkpoint_state.items():
        key = _normalize_checkpoint_key(str(raw_key))
        if _should_ignore_key(key, ignore_prefixes):
            skipped.append((key, "ignored prefix"))
            continue
        if key not in model_state:
            skipped.append((key, "missing target"))
            continue
        if hasattr(value, "shape") and tuple(value.shape) != tuple(model_state[key].shape):
            if ignore_shape_mismatch:
                skipped.append((key, f"shape {tuple(value.shape)} -> {tuple(model_state[key].shape)}"))
                continue
        load_state[key] = value

    missing, unexpected = target_model.load_state_dict(load_state, strict=False)
    logger.info(
        "Loaded fine-tune checkpoint %s: %d tensors loaded, %d skipped, %d missing, %d unexpected",
        checkpoint_path,
        len(load_state),
        len(skipped),
        len(missing),
        len(unexpected),
    )
    if skipped:
        preview = ", ".join(f"{key} ({reason})" for key, reason in skipped[:20])
        logger.info("Skipped fine-tune tensors: %s%s", preview, " ..." if len(skipped) > 20 else "")


def do_test(cfg, model):
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)

        # Running inference with NMS ...
        if cfg.train.get('test_with_nms') is not None and cfg.train.get('test_with_nms') > 0:
            print("Running inference with NMS ...")
            with tune_nms_threshold_and_restore(model, cfg.train.test_with_nms):
                ret_wnms = inference_on_dataset(
                    model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
                )
                ret_wnms = OrderedDict({k + "_wnms": v for k, v in ret_wnms.items()})
            print_csv_format(ret_wnms)
            ret.update(ret_wnms)

        return ret


def _build_best_checkpointer_hook(cfg, checkpointer):
    best_cfg = cfg.train.get("best_checkpointer") or {}
    if not best_cfg.get("enabled", True):
        return None
    if int(cfg.train.eval_period) <= 0:
        return None
    if not hasattr(hooks, "BestCheckpointer"):
        logger.warning("BestCheckpointer is unavailable in this Detectron2 build.")
        return None
    return hooks.BestCheckpointer(
        int(cfg.train.eval_period),
        checkpointer,
        str(best_cfg.get("metric", "bbox/AP")),
        mode=str(best_cfg.get("mode", "max")),
        file_prefix=str(best_cfg.get("file_prefix", "model_best")),
    )


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)

    trainer = Trainer(
        model=model,
        dataloader=train_loader,
        optimizer=optim,
        amp=cfg.train.amp.enabled,
        clip_grad_params=cfg.train.clip_grad.params if cfg.train.clip_grad.enabled else None,
    )

    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
    )

    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            _build_best_checkpointer_hook(cfg, checkpointer)
            if comm.is_main_process()
            else None,
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if not args.resume:
        load_finetune_checkpoint(cfg, model)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    trainer.train(start_iter, cfg.train.max_iter)


def main(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)
        DetectionCheckpointer(model).load(cfg.train.init_checkpoint)
        print(do_test(cfg, model))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    print("="*20)
    command_txt = " ".join(sys.argv)
    print("command: ", command_txt)
    print("="*20)
    args.comand_txt = command_txt

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
