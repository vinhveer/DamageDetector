import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import BinaryDiceLoss, BinaryTverskyLoss
from utils import test_single_volume
import csv

def calc_loss(outputs, high_res_label_batch, bce_loss, dice_loss, tversky_loss, dice_weight: float = 0.8, tversky_weight: float = 0.5):
    masks = outputs['masks']
    if masks.shape[1] != 1:
        raise ValueError(f"Expected a single binary mask logit, got shape {tuple(masks.shape)}")

    targets = high_res_label_batch.float().unsqueeze(1)
    loss_bce = bce_loss(masks, targets)
    loss_dice = dice_loss(masks, high_res_label_batch)
    loss_tversky = tversky_loss(masks, high_res_label_batch)
    region_loss = (1.0 - tversky_weight) * loss_dice + tversky_weight * loss_tversky
    loss = (1 - dice_weight) * loss_bce + dice_weight * region_loss
    return loss, loss_bce, loss_dice, loss_tversky


def _prepare_prompts(sampled_batch, *, use_boxes: bool, use_points: bool):
    box_batch = sampled_batch['box'].cuda() if use_boxes and 'box' in sampled_batch else None
    if use_points and 'point_coords' in sampled_batch and 'point_labels' in sampled_batch:
        point_coords_batch = sampled_batch['point_coords'].cuda()
        point_labels_batch = sampled_batch['point_labels'].cuda()
        points_batch = (point_coords_batch, point_labels_batch)
    else:
        points_batch = None
    return box_batch, points_batch


def _safe_thresholds(args) -> list[float]:
    values = getattr(args, 'val_thresholds', None) or [0.5]
    out = []
    for v in values:
        v = float(v)
        if 0.0 < v < 1.0:
            out.append(v)
    return sorted(set(out)) or [0.5]


def worker_init_fn(worker_id):
    # Retrieve seed from where? Simple workaround: use a global or pass it?
    # Actually, args is not available here. 
    # Usually we just set random seed based on worker_id + some constant
    random.seed(3407 + worker_id)

def trainer_generic(args, model, snapshot_path, multimask_output, low_res): 
    from datasets.dataset_generic import GenericDataset, RandomGenerator, ValGenerator

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations

    # Use GenericDataset. args.val_path and args.root_path should clearly point to folders.
    # We ignore list_dir if using GenericDataset unless we want to keep that logic.
    # The new GenericDataset scans folders.
    
    db_val = GenericDataset(base_dir=args.val_path, split="val_vol", 
                            transform=ValGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res]))

    db_train = GenericDataset(
        base_dir=args.root_path,
        split="train",
        transform=RandomGenerator(
            output_size=[args.img_size, args.img_size],
            low_res=[low_res, low_res],
            background_crop_prob=args.background_crop_prob,
            near_background_crop_prob=args.near_background_crop_prob,
        ),
        patches_per_image=args.patches_per_image,
    )
    
    print("The length of train set is: {}".format(len(db_train)))

    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'worker_init_fn': worker_init_fn
    }
    val_kwargs = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': max(1, args.num_workers // 2) if args.num_workers > 0 else 0,
        'pin_memory': True
    }
    
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 32 # Aggressive prefetching for 64GB RAM
        
        # Adjust val accordingly
        if val_kwargs['num_workers'] > 0:
            val_kwargs['persistent_workers'] = True
            val_kwargs['prefetch_factor'] = 16

    trainloader = DataLoader(db_train, **loader_kwargs)
    valloader = DataLoader(db_val, **val_kwargs)
                    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    bce_loss = nn.BCEWithLogitsLoss()
    dice_loss = BinaryDiceLoss()
    tversky_loss = BinaryTverskyLoss(alpha=args.tversky_alpha, beta=args.tversky_beta)
    val_thresholds = _safe_thresholds(args)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR # Import locally
    
    # Differential Learning Rates: Fast LoRA, Slow Decoder
    lora_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'mask_decoder' in name:
            decoder_params.append(param)
        else:
            lora_params.append(param)

    if args.AdamW:
        optimizer = optim.AdamW([
            {'params': lora_params, 'lr': b_lr},
            {'params': decoder_params, 'lr': b_lr * 0.1}
        ], betas=(0.9, 0.999), weight_decay=0.01)
    else:
        optimizer = optim.SGD([
            {'params': lora_params, 'lr': b_lr},
            {'params': decoder_params, 'lr': b_lr * 0.1}
        ], momentum=0.9, weight_decay=0.0001)

    print(f"Detected {len(lora_params)} param tensors for LoRA (LR: {b_lr})")
    print(f"Detected {len(decoder_params)} param tensors for Decoder (LR: {b_lr * 0.1})")

    # Cosine Scheduler: Smooth decay for better convergence
    scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs, eta_min=1e-6)

    if args.use_amp: # Using amp may cause unstable gradients during training
        scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp) 
        # scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)   
    iter_num = 0

    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)  
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    iterator = tqdm(range(max_epoch), ncols=70)
    best_performance = 0.0
    best_threshold = 0.5
    patience = max_epoch
    patience_counter = 0
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            optimizer.zero_grad()

            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w] tensor 
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()# For crack segmentation, low resolution label is not recommended

            use_points = random.random() < float(getattr(args, 'train_use_points_prob', 0.0))
            box_batch, points_batch = _prepare_prompts(
                sampled_batch,
                use_boxes=bool(getattr(args, 'train_use_boxes', True)),
                use_points=use_points,
            )

            # assert image_batch.max() <= 1, f'image_batch max: {image_batch.max()}'

            if args.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                    outputs = model(image_batch, multimask_output, args.img_size, boxes=box_batch, points=points_batch)
                    loss, loss_bce, loss_dice, loss_tversky = calc_loss(
                        outputs, label_batch, bce_loss, dice_loss, tversky_loss, args.dice_param, args.tversky_weight
                    )
                scaler.scale(loss).backward()
                
                # Gradient Clipping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
            else:
                outputs = model(image_batch, multimask_output, args.img_size, boxes=box_batch, points=points_batch)  
                loss, loss_bce, loss_dice, loss_tversky = calc_loss(
                    outputs, label_batch, bce_loss, dice_loss, tversky_loss, args.dice_param, args.tversky_weight
                )
                loss.backward()
                
                # Gradient Clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                warmup_factor = (iter_num + 1) / args.warmup_period
                # Update LoRA group
                optimizer.param_groups[0]['lr'] = base_lr * warmup_factor
                # Update Decoder group (if exists) with 0.1 ratio
                if len(optimizer.param_groups) > 1:
                    optimizer.param_groups[1]['lr'] = (base_lr * 0.1) * warmup_factor
                
                lr_ = optimizer.param_groups[0]['lr']
            else:
                # Handled by ReduceLROnPlateau at epoch end
                lr_ = optimizer.param_groups[0]['lr']

            iter_num = iter_num + 1

            logging.info(
                'iteration %d : loss : %f, loss_bce: %f, loss_dice: %f loss_tversky: %f lr: %f'
                % (iter_num, loss.item(), loss_bce.item(), loss_dice.item(), loss_tversky.item(), lr_)
            )

        val_interval = args.save_interval 
        if epoch_num % val_interval ==0:
            logging.info(f'{len(valloader)} val iterations per epoch')
            model.eval()
            box_by_thr = {thr: np.zeros(4, dtype=np.float64) for thr in val_thresholds}
            for i_batch, sampled_batch in tqdm(enumerate(valloader)):
                image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0] # tensor
                box_only, _ = _prepare_prompts(
                    sampled_batch,
                    use_boxes=True,
                    use_points=False,
                )

                case_logs = []
                for thr in val_thresholds:
                    metric_box_i = test_single_volume(
                        image, label, model, classes=args.num_classes, multimask_output=multimask_output,
                        patch_size=[args.img_size, args.img_size], test_save_path=None, boxes=box_only, points=None,
                        threshold_prob=thr,
                    )
                    box_by_thr[thr] += np.array(metric_box_i)
                    case_logs.append(f"thr={thr:.2f} box_iou={metric_box_i[3]:.4f}")
                logging.info('idx %d case %s %s' % (i_batch, case_name, ' | '.join(case_logs)))

            metric_box_by_thr = {thr: box_by_thr[thr] / len(db_val) for thr in val_thresholds}
            selected_thr = max(val_thresholds, key=lambda thr: (metric_box_by_thr[thr][3], metric_box_by_thr[thr][2]))
            metric_box = metric_box_by_thr[selected_thr]
            for thr in val_thresholds:
                logging.info(
                    'Validation threshold %.2f -> box_iou: %f'
                    % (thr, metric_box_by_thr[thr][3])
                )
            logging.info(
                'Validation selected threshold %.2f box_only: mean_pr: %f mean_re: %f mean_f1: %f mean_iou : %f'
                % (selected_thr, metric_box[0], metric_box[1], metric_box[2], metric_box[3])
            )
            logging.info("Validation in epoch %d Finished!" % epoch_num)


            with open(snapshot_path + '/val.csv', 'a', newline='') as f:
                writercsv = csv.writer(f)
                writercsv.writerow([
                    epoch_num,
                    selected_thr,
                    metric_box[0], metric_box[1], metric_box[2], metric_box[3],
                ])

            performance = metric_box[3]
            if performance > best_performance:
                best_performance = performance
                best_threshold = selected_thr
                patience_counter = 0
                save_best_path = os.path.join(snapshot_path, 'best_model.pth')
                try:
                    model.save_delta_parameters(save_best_path)
                except:
                    model.module.save_delta_parameters(save_best_path)
                with open(os.path.join(snapshot_path, 'best_threshold.txt'), 'w', encoding='utf-8') as f:
                    f.write(f"{best_threshold:.4f}\n")
                logging.info(f"New best box-only IoU: {best_performance} at threshold {best_threshold:.2f}. Saved best model to {save_best_path}")
            else:
                patience_counter += 1
                logging.info(f"No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                logging.info("Early stopping triggered")
                break
            
        # Step the scheduler each epoch, independent of validation
        scheduler.step()


        save_interval = args.save_interval 
        if epoch_num  % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_delta_parameters(save_mode_path) # save delta, prompt encoder, and decoder
            except:
                model.module.save_delta_parameters(save_mode_path)

            logging.info("save model to {}".format(save_mode_path))
        
        if epoch_num >= max_epoch - 1 or epoch_num >= stop_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(epoch_num) + '.pth')
            try:
                model.save_delta_parameters(save_mode_path)
            except:
                model.module.save_delta_parameters(save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    return "Training Finished!"
