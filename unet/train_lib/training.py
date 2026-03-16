import os

import torch
import torch.distributed as dist
from tqdm import tqdm

from .metrics import dice_score_from_prob, iou_score_from_prob
from .visualization import _HAS_MPL, plt, visualize_predictions


def _dist_on():
    return dist.is_available() and dist.is_initialized()


def _rank():
    return dist.get_rank() if _dist_on() else 0


def _world_size():
    return dist.get_world_size() if _dist_on() else 1


def _is_main():
    return _rank() == 0


def _all_reduce_sum(value, device):
    t = torch.tensor(value, device=device, dtype=torch.float64)
    if _dist_on():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def _unpack_batch(batch):
    if batch is None:
        return None, None, None
    if isinstance(batch, (tuple, list)) and len(batch) == 3:
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], None


def _compute_reconstructed_metrics(tile_predictions, thresholds, base_threshold):
    if not tile_predictions:
        return None

    image_metrics = []
    threshold_sums = {float(t): {"dice": 0.0, "iou": 0.0} for t in thresholds}
    base_threshold = float(base_threshold)

    for image_name, payload in tile_predictions.items():
        prob_sum = payload["prob_sum"]
        count_sum = payload["count_sum"].clamp_min(1.0)
        mask_sum = payload["mask_sum"]
        orig_h = int(payload["orig_h"])
        orig_w = int(payload["orig_w"])

        prob = (prob_sum / count_sum)[:, :orig_h, :orig_w].unsqueeze(0)
        target = (mask_sum > 0.5).float()[:, :orig_h, :orig_w].unsqueeze(0)

        base_dice = dice_score_from_prob(prob, target, thr=base_threshold)
        base_iou = iou_score_from_prob(prob, target, thr=base_threshold)
        image_metrics.append((base_dice, base_iou))

        for thr in threshold_sums:
            threshold_sums[thr]["dice"] += dice_score_from_prob(prob, target, thr=thr)
            threshold_sums[thr]["iou"] += iou_score_from_prob(prob, target, thr=thr)

    image_count = float(len(image_metrics))
    avg_dice = sum(x[0] for x in image_metrics) / image_count
    avg_iou = sum(x[1] for x in image_metrics) / image_count
    for thr in threshold_sums:
        threshold_sums[thr]["dice_avg"] = threshold_sums[thr]["dice"] / image_count
        threshold_sums[thr]["iou_avg"] = threshold_sums[thr]["iou"] / image_count

    return {
        "avg_dice": avg_dice,
        "avg_iou": avg_iou,
        "thresholds": threshold_sums,
        "image_count": int(image_count),
    }


def _gather_reconstructed_tiles(tile_predictions):
    if not _dist_on():
        return tile_predictions

    gathered = [None for _ in range(_world_size())]
    dist.all_gather_object(gathered, tile_predictions)

    if not _is_main():
        return None

    merged = {}
    for rank_tiles in gathered:
        if not rank_tiles:
            continue
        for image_name, payload in rank_tiles.items():
            target = merged.get(image_name)
            if target is None:
                merged[image_name] = {
                    "prob_sum": payload["prob_sum"].clone(),
                    "count_sum": payload["count_sum"].clone(),
                    "mask_sum": payload["mask_sum"].clone(),
                    "orig_h": int(payload["orig_h"]),
                    "orig_w": int(payload["orig_w"]),
                }
                continue
            target["prob_sum"] += payload["prob_sum"]
            target["count_sum"] += payload["count_sum"]
            target["mask_sum"] = torch.maximum(target["mask_sum"], payload["mask_sum"])
    return merged


def _broadcast_optional_metrics(metrics, device):
    if not _dist_on():
        return metrics

    payload = [metrics if _is_main() else None]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _iter_prefetch(loader, device):
    if device.type != "cuda":
        for batch in loader:
            yield batch
        return

    stream = torch.cuda.Stream(device=device)
    it = iter(loader)

    def _next_batch():
        while True:
            try:
                batch = next(it)
            except StopIteration:
                return None
            if batch is None:
                continue
            images, masks, metadata = _unpack_batch(batch)
            with torch.cuda.stream(stream):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
            return images, masks, metadata

    next_batch = _next_batch()
    while next_batch is not None:
        torch.cuda.current_stream(device).wait_stream(stream)
        batch = next_batch
        next_batch = _next_batch()
        yield batch


import csv
import logging

from model_io import save_checkpoint


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, output_dir, model_config=None, csv_path=None, grad_accum_steps=1, use_amp=False):
    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)

    use_cuda = device.type == "cuda" and use_amp
    use_prefetch = use_cuda
    non_blocking = use_cuda
    is_main = _is_main()
    
    # Initialize AMP Scaler
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    disable_visuals = bool(getattr(train_model, "_disable_visuals", False))
    disable_loss_curve = bool(getattr(train_model, "_disable_loss_curve", False))
    best_val_loss = float('inf')
    best_monitor_value = None
    best_monitor_label = None
    train_losses = []
    val_losses = []
    # Early stopping parameters.
    patience = int(getattr(train_model, "_early_stop_patience", 15))
    counter = 0

    for epoch in range(num_epochs):
        if hasattr(getattr(train_loader, "sampler", None), "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0

        train_iter = _iter_prefetch(train_loader, device) if use_prefetch else train_loader
        optimizer.zero_grad()
        
        train_pbar = tqdm(train_iter, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") if is_main else train_iter
        for i, batch in enumerate(train_pbar):
            if batch is None:
                continue
            images, masks, _ = _unpack_batch(batch)
            if not use_prefetch:
                images = images.to(device, non_blocking=non_blocking)
                masks = masks.to(device, non_blocking=non_blocking)

            # Forward pass with AMP
            with torch.amp.autocast('cuda', enabled=use_cuda):
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss = loss / grad_accum_steps

            # Backward pass with Scaler
            scaler.scale(loss).backward()
            
            if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * grad_accum_steps
            train_steps += 1

        train_loss_sum = _all_reduce_sum(train_loss, device=device)
        train_steps_sum = _all_reduce_sum(train_steps, device=device)
        if train_steps == 0:
            avg_train_loss = float("nan")
        else:
            avg_train_loss = train_loss_sum / max(1.0, train_steps_sum)
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        val_dice_sum = 0.0
        val_iou_sum = 0.0
        tile_predictions = {}
        val_thr_report = {}
        if hasattr(train_model, "_metric_thresholds"):
            for t in train_model._metric_thresholds:
                val_thr_report[t] = {"dice": 0.0, "iou": 0.0}

        with torch.no_grad():
            val_iter = _iter_prefetch(val_loader, device) if use_prefetch else val_loader
            val_pbar = tqdm(val_iter, desc=f"Epoch {epoch+1}/{num_epochs} [Val]") if is_main else val_iter
            for batch in val_pbar:
                if batch is None:
                    continue
                images, masks, metadata = _unpack_batch(batch)
                if not use_prefetch:
                    images = images.to(device, non_blocking=non_blocking)
                    masks = masks.to(device, non_blocking=non_blocking)

                # AMp for validation too
                with torch.amp.autocast('cuda', enabled=use_cuda):
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                val_loss += loss.item()
                val_steps += 1
                thr = getattr(train_model, "_metric_threshold", 0.5)
                prob = torch.sigmoid(outputs)
                val_dice_sum += dice_score_from_prob(prob, masks, thr=thr)
                val_iou_sum += iou_score_from_prob(prob, masks, thr=thr)

                if val_thr_report:
                    for t in val_thr_report.keys():
                        t_val = float(t)
                        val_thr_report[t]["dice"] += dice_score_from_prob(prob, masks, thr=t_val)
                        val_thr_report[t]["iou"] += iou_score_from_prob(prob, masks, thr=t_val)

                if metadata:
                    prob_cpu = prob.detach().float().cpu()
                    masks_cpu = masks.detach().float().cpu()
                    for idx, meta in enumerate(metadata):
                        image_name = meta["image_name"]
                        padded_h = int(meta["padded_h"])
                        padded_w = int(meta["padded_w"])
                        if image_name not in tile_predictions:
                            tile_predictions[image_name] = {
                                "prob_sum": torch.zeros((1, padded_h, padded_w), dtype=torch.float32),
                                "count_sum": torch.zeros((1, padded_h, padded_w), dtype=torch.float32),
                                "mask_sum": torch.zeros((1, padded_h, padded_w), dtype=torch.float32),
                                "orig_h": int(meta["orig_h"]),
                                "orig_w": int(meta["orig_w"]),
                            }
                        x = int(meta["x"])
                        y = int(meta["y"])
                        patch_h = int(meta["patch_h"])
                        patch_w = int(meta["patch_w"])
                        payload = tile_predictions[image_name]
                        payload["prob_sum"][:, y:y + patch_h, x:x + patch_w] += prob_cpu[idx]
                        payload["count_sum"][:, y:y + patch_h, x:x + patch_w] += 1.0
                        payload["mask_sum"][:, y:y + patch_h, x:x + patch_w] = torch.maximum(
                            payload["mask_sum"][:, y:y + patch_h, x:x + patch_w],
                            masks_cpu[idx],
                        )

        val_loss_sum = _all_reduce_sum(val_loss, device=device)
        val_steps_sum = _all_reduce_sum(val_steps, device=device)
        val_dice_sum_all = _all_reduce_sum(val_dice_sum, device=device)
        val_iou_sum_all = _all_reduce_sum(val_iou_sum, device=device)
        metric_threshold = getattr(train_model, "_metric_threshold", 0.5)
        threshold_values = [metric_threshold]
        if hasattr(train_model, "_metric_thresholds"):
            threshold_values.extend(float(t) for t in train_model._metric_thresholds)
        threshold_values = sorted(set(threshold_values))
        reconstructed_tile_predictions = _gather_reconstructed_tiles(tile_predictions)
        reconstructed_metrics = None
        if reconstructed_tile_predictions:
            reconstructed_metrics = _compute_reconstructed_metrics(
                reconstructed_tile_predictions,
                threshold_values,
                metric_threshold,
            )
        reconstructed_metrics = _broadcast_optional_metrics(reconstructed_metrics, device)
        sweep_best_iou = avg_val_iou = 0.0 if val_steps == 0 else None
        sweep_best_dice = avg_val_dice = 0.0 if val_steps == 0 else None
        sweep_best_iou_thr = metric_threshold
        sweep_best_dice_thr = metric_threshold
        if val_steps == 0:
            avg_val_loss = float("inf")
            avg_val_dice = 0.0
            avg_val_iou = 0.0
        else:
            avg_val_loss = val_loss_sum / max(1.0, val_steps_sum)
            avg_val_dice = val_dice_sum_all / max(1.0, val_steps_sum)
            avg_val_iou = val_iou_sum_all / max(1.0, val_steps_sum)
            if reconstructed_metrics is not None:
                avg_val_dice = reconstructed_metrics["avg_dice"]
                avg_val_iou = reconstructed_metrics["avg_iou"]
            sweep_best_iou = avg_val_iou
            sweep_best_dice = avg_val_dice
            if val_thr_report:
                for t, report in val_thr_report.items():
                    if reconstructed_metrics is not None and float(t) in reconstructed_metrics["thresholds"]:
                        avg_thr_dice = reconstructed_metrics["thresholds"][float(t)]["dice_avg"]
                        avg_thr_iou = reconstructed_metrics["thresholds"][float(t)]["iou_avg"]
                    else:
                        avg_thr_dice = _all_reduce_sum(report["dice"], device=device) / max(1.0, val_steps_sum)
                        avg_thr_iou = _all_reduce_sum(report["iou"], device=device) / max(1.0, val_steps_sum)
                    val_thr_report[t]["dice_avg"] = avg_thr_dice
                    val_thr_report[t]["iou_avg"] = avg_thr_iou
                    if avg_thr_iou > sweep_best_iou:
                        sweep_best_iou = avg_thr_iou
                        sweep_best_iou_thr = float(t)
                    if avg_thr_dice > sweep_best_dice:
                        sweep_best_dice = avg_thr_dice
                        sweep_best_dice_thr = float(t)
        val_losses.append(avg_val_loss)

        if is_main:
            logging.info(f"Epoch {epoch+1}/{num_epochs}:")
            logging.info(f"Train Loss: {avg_train_loss:.4f}")
            thr = metric_threshold
            logging.info(
                f"Val Loss: {avg_val_loss:.4f} | Val Dice@{thr}: {avg_val_dice:.4f} | Val IoU@{thr}: {avg_val_iou:.4f}"
            )
            if reconstructed_metrics is not None:
                logging.info(
                    f"Validation used reconstructed full-image metrics over {reconstructed_metrics['image_count']} image(s)."
                )
            if val_thr_report:
                sweep_items = []
                for t in sorted(val_thr_report.keys(), key=float):
                    sweep_items.append(
                        f"{float(t):.2f}:dice={val_thr_report[t]['dice_avg']:.4f},iou={val_thr_report[t]['iou_avg']:.4f}"
                    )
                logging.info("Threshold sweep: " + " | ".join(sweep_items))
                logging.info(
                    f"Best sweep metrics: IoU={sweep_best_iou:.4f}@{sweep_best_iou_thr:.2f} | "
                    f"Dice={sweep_best_dice:.4f}@{sweep_best_dice_thr:.2f}"
                )

        # Step the learning-rate scheduler SAFELY
        sched_metric = getattr(train_model, "_scheduler_metric", "loss")
        metric_val = avg_val_dice if sched_metric == "dice" else avg_val_loss
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric_val)
        else:
            # CosineAnnealing / StepLR / etc
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        if is_main:
            logging.info(f"Current learning rate: {current_lr:.7f}")

        # CSV Logging
        if csv_path and is_main:
            with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, avg_train_loss, avg_val_loss, avg_val_dice, avg_val_iou, current_lr])

        metrics = {
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_dice": avg_val_dice,
            "val_iou": avg_val_iou,
            "best_iou_sweep": sweep_best_iou,
            "best_iou_sweep_threshold": sweep_best_iou_thr,
            "best_dice_sweep": sweep_best_dice,
            "best_dice_sweep_threshold": sweep_best_dice_thr,
            "lr": current_lr,
        }

        # Save last model
        if is_main:
            last_model_path = os.path.join(output_dir, "last_model.pth")
            save_checkpoint(last_model_path, model, model_config, epoch=epoch + 1, metrics=metrics)
        
        # Save epoch model (if configured)
        save_all = getattr(train_model, "_save_all_epochs", True) 
        if save_all and is_main:
            epoch_model_path = os.path.join(output_dir, f"epoch_{epoch+1}.pth")
            save_checkpoint(epoch_model_path, model, model_config, epoch=epoch + 1, metrics=metrics)

        best_metric_name = getattr(train_model, "_best_model_metric", "best_iou")
        metric_candidates = {
            "best_iou": (avg_val_iou, True, f"Val IoU@{getattr(train_model, '_metric_threshold', 0.5):.2f}"),
            "best_dice": (avg_val_dice, True, f"Val Dice@{getattr(train_model, '_metric_threshold', 0.5):.2f}"),
            "best_iou_sweep": (sweep_best_iou, True, f"Best sweep IoU@{sweep_best_iou_thr:.2f}"),
            "best_dice_sweep": (sweep_best_dice, True, f"Best sweep Dice@{sweep_best_dice_thr:.2f}"),
            "loss": (avg_val_loss, False, "Val Loss"),
        }
        monitor_value, higher_is_better, monitor_label = metric_candidates[best_metric_name]
        improved = False
        if val_steps > 0:
            if best_monitor_value is None:
                improved = True
            elif higher_is_better:
                improved = monitor_value > best_monitor_value
            else:
                improved = monitor_value < best_monitor_value

        if improved:
            best_monitor_value = monitor_value
            best_monitor_label = monitor_label
            best_val_loss = avg_val_loss
            counter = 0 
            if is_main:
                model_path = os.path.join(output_dir, "best_model.pth")
                save_checkpoint(model_path, model, model_config, epoch=epoch + 1, metrics=metrics)
                logging.info(
                    f"New best {monitor_label}: {monitor_value:.4f}. Saved best model to {model_path}!"
                )
        else:
            counter += 1
            if is_main:
                logging.info(
                    f"{best_monitor_label or monitor_label} did not improve. Patience: {counter}/{patience}"
                )
            if counter >= patience:
                if is_main:
                    logging.info(
                        f"Early stopping triggered on {best_monitor_label or monitor_label}. "
                        f"No improvement for {patience} epochs."
                    )
                break

        # Visualize a few predictions.
        vis_every = int(getattr(train_model, "_visualize_every", 5))
        if is_main and not disable_visuals and _HAS_MPL and vis_every > 0 and (epoch + 1) % vis_every == 0:
            visualize_predictions(model, val_loader, device, epoch, output_dir, thr=thr)

    # Plot the loss curves.
    if is_main and not disable_loss_curve and _HAS_MPL:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        loss_curve_path = os.path.join(output_dir, 'loss_curve.png')
        plt.savefig(loss_curve_path)
        print(f'Loss curve saved to {loss_curve_path}')
        plt.close()
    elif not _HAS_MPL:
        print("matplotlib is not installed; skipping loss curves and prediction visualizations.")
    elif disable_loss_curve:
        print("Loss curve disabled; skipping loss_curve.png.")
