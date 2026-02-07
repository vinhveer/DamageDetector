import os

import torch
from tqdm import tqdm

from .metrics import dice_score_from_prob, iou_score_from_prob
from .visualization import _HAS_MPL, plt, visualize_predictions


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
            images, masks = batch
            with torch.cuda.stream(stream):
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
            return images, masks

    next_batch = _next_batch()
    while next_batch is not None:
        torch.cuda.current_stream(device).wait_stream(stream)
        batch = next_batch
        next_batch = _next_batch()
        yield batch


import csv
import logging

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, output_dir, csv_path=None, grad_accum_steps=1):
    # Ensure the output directory exists.
    os.makedirs(output_dir, exist_ok=True)

    use_cuda = device.type == "cuda"
    use_prefetch = use_cuda
    non_blocking = use_cuda
    
    # Initialize AMP Scaler
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    disable_visuals = bool(getattr(train_model, "_disable_visuals", False))
    disable_loss_curve = bool(getattr(train_model, "_disable_loss_curve", False))
    best_val_loss = float('inf')
    best_val_score = -1.0 # Tracks IoU or Dice
    train_losses = []
    val_losses = []
    # Early stopping parameters.
    patience = int(getattr(train_model, "_early_stop_patience", 15))
    counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_steps = 0

        train_iter = _iter_prefetch(train_loader, device) if use_prefetch else train_loader
        optimizer.zero_grad()
        
        for i, batch in enumerate(tqdm(train_iter, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')):
            if batch is None:
                continue
            images, masks = batch
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

        if train_steps == 0:
            print("Warning: no valid training batches in this epoch (all samples failed to load).")
            avg_train_loss = float("nan")
        else:
            avg_train_loss = train_loss / train_steps
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        val_loss = 0
        val_steps = 0
        val_dice_sum = 0.0
        val_iou_sum = 0.0
        val_thr_report = {}
        if hasattr(train_model, "_metric_thresholds"):
            for t in train_model._metric_thresholds:
                val_thr_report[t] = {"dice": 0.0, "iou": 0.0}

        with torch.no_grad():
            val_iter = _iter_prefetch(val_loader, device) if use_prefetch else val_loader
            for batch in tqdm(val_iter, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                if batch is None:
                    continue
                images, masks = batch
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

        if val_steps == 0:
            logging.warning("Warning: no valid validation batches (all samples failed to load).")
            avg_val_loss = float("inf")
            avg_val_dice = 0.0
            avg_val_iou = 0.0
        else:
            avg_val_loss = val_loss / val_steps
            avg_val_dice = val_dice_sum / val_steps
            avg_val_iou = val_iou_sum / val_steps
        val_losses.append(avg_val_loss)

        logging.info(f'Epoch {epoch+1}/{num_epochs}:')
        logging.info(f'Train Loss: {avg_train_loss:.4f}')
        thr = getattr(train_model, "_metric_threshold", 0.5)
        logging.info(f'Val Loss: {avg_val_loss:.4f} | Val Dice@{thr}: {avg_val_dice:.4f} | Val IoU@{thr}: {avg_val_iou:.4f}')
        if val_steps > 0 and val_thr_report:
            items = []
            for t in sorted(val_thr_report.keys()):
                d = val_thr_report[t]["dice"] / val_steps
                j = val_thr_report[t]["iou"] / val_steps
                items.append(f"{t}:D{d:.3f}/I{j:.3f}")
            print("Val metrics sweep (thr:Dice/IoU): " + " | ".join(items))

        # Step the learning-rate scheduler SAFELY
        sched_metric = getattr(train_model, "_scheduler_metric", "loss")
        metric_val = avg_val_dice if sched_metric == "dice" else avg_val_loss
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric_val)
        else:
            # CosineAnnealing / StepLR / etc
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f'Current learning rate: {current_lr:.7f}')

        # CSV Logging
        if csv_path:
             with open(csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([epoch+1, avg_train_loss, avg_val_loss, avg_val_dice, avg_val_iou, current_lr])

        # Save last model
        last_model_path = os.path.join(output_dir, 'last_model.pth')
        torch.save(model.state_dict(), last_model_path)
        
        # Save epoch model (if configured)
        save_all = getattr(train_model, "_save_all_epochs", True) 
        if save_all:
             epoch_model_path = os.path.join(output_dir, f'epoch_{epoch+1}.pth')
             torch.save(model.state_dict(), epoch_model_path)

        # Save the best model (using IoU as primary)
        current_perf = avg_val_iou 
        if val_steps > 0 and current_perf > best_val_score:
            best_val_score = current_perf
            best_val_loss = avg_val_loss  
            counter = 0 
            model_path = os.path.join(output_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            logging.info(f'New best IoU: {current_perf:.4f}. Saved best model to {model_path}!')
        else:
            counter += 1
            logging.info(f'Val IoU did not improve. Patience: {counter}/{patience}')
            if counter >= patience:
                logging.info(f'Early stopping triggered! No improvement for {patience} epochs.')
                break

        # Visualize a few predictions.
        vis_every = int(getattr(train_model, "_visualize_every", 5))
        if not disable_visuals and _HAS_MPL and vis_every > 0 and (epoch + 1) % vis_every == 0:
            visualize_predictions(model, val_loader, device, epoch, output_dir, thr=thr)

    # Plot the loss curves.
    if not disable_loss_curve and _HAS_MPL:
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
