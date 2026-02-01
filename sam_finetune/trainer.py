import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from utils import DiceLoss, Focal_loss
from torchvision import transforms
from utils import test_single_volume
import csv

def calc_loss(outputs, high_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    masks = outputs['masks']  
    loss_ce = ce_loss(masks, high_res_label_batch[:].long())  
    loss_dice = dice_loss(masks, high_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice

    return loss, loss_ce, loss_dice


def worker_init_fn(worker_id):
    # Retrieve seed from where? Simple workaround: use a global or pass it?
    # Actually, args is not available here. 
    # Usually we just set random seed based on worker_id + some constant
    random.seed(3407 + worker_id)

def trainer_generic(args, model, snapshot_path, multimask_output, low_res): 
    # from datasets.dataset_khanhha import Khanhha_dataset, RandomGenerator
    from datasets.dataset_generic import GenericDataset, RandomGenerator

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
    
    db_val = GenericDataset(base_dir=args.val_path, split="val_vol", output_size=[args.img_size, args.img_size])

    db_train = GenericDataset(base_dir=args.root_path, split="train",
                               transform=RandomGenerator(output_size=[args.img_size, args.img_size], low_res=[low_res, low_res]))
    
    print("The length of train set is: {}".format(len(db_train)))

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True,
                             worker_init_fn=worker_init_fn, persistent_workers=True, prefetch_factor=2)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=16, persistent_workers=True, prefetch_factor=2)
                    
    if args.n_gpu > 1:
        model = nn.DataParallel(model)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes + 1)
    if args.warmup:
        b_lr = base_lr / args.warmup_period
    else:
        b_lr = base_lr
    if args.AdamW:
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, betas=(0.9, 0.999), weight_decay=0.01)
    else:
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=b_lr, momentum=0.9, weight_decay=0.0001)  # Even pass the model.parameters(), the `requires_grad=False` layers will not update
    
    if args.use_amp: # Using amp may cause unstable gradients during training
        # scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp) 
        scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)   
    iter_num = 0

    max_epoch = args.max_epochs
    stop_epoch = args.stop_epoch
    max_iterations = args.max_epochs * len(trainloader)  
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))

    iterator = tqdm(range(max_epoch), ncols=70)
    best_performance = 0.0
    patience = max_epoch
    patience_counter = 0
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            optimizer.zero_grad()

            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']  # [b, c, h, w], [b, h, w] tensor 
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()# For crack segmentation, low resolution label is not recommended

            # Extract Prompts
            if 'box' in sampled_batch:
                box_batch = sampled_batch['box'].cuda()
                point_coords_batch = sampled_batch['point_coords'].cuda()
                point_labels_batch = sampled_batch['point_labels'].cuda()
                points_batch = (point_coords_batch, point_labels_batch)
            else:
                box_batch = None
                points_batch = None

            assert image_batch.max() <= 1, f'image_batch max: {image_batch.max()}'

            if args.use_amp:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                    outputs = model(image_batch, multimask_output, args.img_size, boxes=box_batch, points=points_batch)
                    loss, loss_ce, loss_dice = calc_loss(outputs, label_batch, ce_loss, dice_loss, args.dice_param)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
            else:
                outputs = model(image_batch, multimask_output, args.img_size, boxes=box_batch, points=points_batch)  
                loss, loss_ce, loss_dice = calc_loss(outputs, label_batch, ce_loss, dice_loss, args.dice_param)
                loss.backward()
                optimizer.step()
            if args.warmup and iter_num < args.warmup_period:
                lr_ = base_lr * ((iter_num + 1) / args.warmup_period)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            else:
                if args.warmup:
                    shift_iter = iter_num - args.warmup_period
                    assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                else:
                    shift_iter = iter_num
                lr_ = base_lr * (1.0 - shift_iter / max_iterations) ** args.lr_exp  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_

            iter_num = iter_num + 1

            logging.info('iteration %d : loss : %f, loss_ce: %f, loss_dice: %f  lr: %f' % (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), lr_))

        val_interval = args.save_interval 
        if epoch_num % val_interval ==0:
            logging.info(f'{len(valloader)} val iterations per epoch')
            model.eval()
            metric_list = 0.0
            for i_batch, sampled_batch in tqdm(enumerate(valloader)):
                image, label, case_name = sampled_batch['image'], sampled_batch['label'], sampled_batch['case_name'][0] # tensor
                
                # Extract Prompts for Validation
                if 'box' in sampled_batch:
                    val_box = sampled_batch['box'].cuda()
                    val_pt_coords = sampled_batch['point_coords'].cuda()
                    val_pt_labels = sampled_batch['point_labels'].cuda()
                    val_points = (val_pt_coords, val_pt_labels)
                else:
                    val_box = None
                    val_points = None

                metric_i = test_single_volume(image, label, model, classes=args.num_classes, multimask_output=multimask_output,
                                    patch_size=[args.img_size, args.img_size],test_save_path=None, boxes=val_box, points=val_points)
                metric_list += np.array(metric_i)
                logging.info('idx %d case %s mean_pr %f mean_re %f mean_f1 %f mean_iou %f' % (
                                i_batch, case_name, metric_i[0], metric_i[1],metric_i[2], metric_i[3]))
            metric_list = metric_list / len(db_val)
            logging.info('Validation performance: mean_pr: %f mean_re: %f mean_f1: %f  mean_iou : %f' % (metric_list[0], metric_list[1],metric_list[2], metric_list[3]))
            logging.info("Validation in epoch %d Finished!" % epoch_num)


            with open(snapshot_path + '/val.csv', 'a', newline='') as f:
                writercsv = csv.writer(f)
                writercsv.writerow([epoch_num, metric_list[0], metric_list[1],metric_list[2], metric_list[3]])

            performance = metric_list[3]
            if performance > best_performance:
                best_performance = performance
                patience_counter = 0
                save_best_path = os.path.join(snapshot_path, 'best_model.pth')
                try:
                    model.save_delta_parameters(save_best_path)
                except:
                    model.module.save_delta_parameters(save_best_path)
                logging.info(f"New best IoU: {best_performance}. Saved best model to {save_best_path}")
            else:
                patience_counter += 1
                logging.info(f"No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                logging.info("Early stopping triggered")
                break


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
