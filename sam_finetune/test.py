import os
import sys
import logging
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from importlib import import_module

from utils import test_single_volume
from segment_anything import sam_model_registry
from datasets.dataset_generic import GenericDataset, ValGenerator


def inference(args, multimask_output, model, test_save_path=None):
    db_test = GenericDataset(
        base_dir=args.volume_path,
        split='test_vol',
        transform=ValGenerator(output_size=[args.img_size, args.img_size], low_res=[args.img_size, args.img_size]),
    )
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=0)
    logging.info(f'{len(testloader)} test iterations')
    model.eval()

    metric_oracle = 0.0
    metric_box = 0.0
    for i_batch, sampled_batch in enumerate(testloader):
        image = sampled_batch['image']
        label = sampled_batch['label']
        case_name = sampled_batch['case_name'][0]

        box = sampled_batch['box'].cuda() if 'box' in sampled_batch else None
        point_coords = sampled_batch['point_coords'].cuda() if 'point_coords' in sampled_batch else None
        point_labels = sampled_batch['point_labels'].cuda() if 'point_labels' in sampled_batch else None
        points = (point_coords, point_labels) if point_coords is not None and point_labels is not None else None

        metric_oracle_i = test_single_volume(
            image, label, model, classes=args.num_classes, multimask_output=multimask_output,
            patch_size=[args.img_size, args.img_size], test_save_path=test_save_path, case=case_name,
            boxes=box, points=points, threshold_prob=args.pred_threshold,
        )
        metric_box_i = test_single_volume(
            image, label, model, classes=args.num_classes, multimask_output=multimask_output,
            patch_size=[args.img_size, args.img_size], test_save_path=None, case=case_name,
            boxes=box, points=None, threshold_prob=args.pred_threshold,
        )
        metric_oracle += np.array(metric_oracle_i)
        metric_box += np.array(metric_box_i)
        logging.info(
            'idx %d case %s oracle_pr %f oracle_re %f oracle_f1 %f oracle_iou %f box_pr %f box_re %f box_f1 %f box_iou %f'
            % (
                i_batch, case_name,
                metric_oracle_i[0], metric_oracle_i[1], metric_oracle_i[2], metric_oracle_i[3],
                metric_box_i[0], metric_box_i[1], metric_box_i[2], metric_box_i[3],
            )
        )

    metric_oracle = metric_oracle / len(db_test)
    metric_box = metric_box / len(db_test)
    logging.info(
        'Testing oracle: mean_pr %f mean_re %f mean_f1 %f mean_iou : %f'
        % (metric_oracle[0], metric_oracle[1], metric_oracle[2], metric_oracle[3])
    )
    logging.info(
        'Testing box_only: mean_pr %f mean_re %f mean_f1 %f mean_iou : %f'
        % (metric_box[0], metric_box[1], metric_box[2], metric_box[3])
    )
    logging.info("Testing Finished!")
    return 1


def config_to_dict(config):
    items_dict = {}
    with open(config, 'r') as f:
        items = f.readlines()
    for line in items:
        key, value = line.strip().split(': ', 1)
        items_dict[key] = value
    return items_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='The config file provided by the trained model')
    parser.add_argument('--volume_path', type=str, required=True)
    parser.add_argument('--num_classes', type=int, default=1, help='For crack segmentation, the output class should be 1')
    parser.add_argument('--output_dir', type=str, default='./output/test')
    parser.add_argument('--img_size', type=int, default=448, help='Input image size of the network')
    parser.add_argument('--seed', type=int, default=3407, help='random seed')
    parser.add_argument('--is_savenii', action='store_true', help='Whether to save results during inference')
    parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
    parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_h_4b8939.pth', help='Pretrained checkpoint')
    parser.add_argument('--delta_ckpt', type=str, default=None, help='The checkpoint from adapter/LoRA')
    parser.add_argument('--vit_name', type=str, default='vit_h', help='Select one vit model')
    parser.add_argument('--delta_type', type=str, default='adapter', help='choose from "adapter" or "lora" or "both"')
    parser.add_argument('--middle_dim', type=int, default=32, help='Middle dim of adapter')
    parser.add_argument('--scaling_factor', type=float, default=0.1, help='Scaling_factor of adapter')
    parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
    parser.add_argument('--pred_threshold', type=float, default=0.5, help='Probability threshold for binary crack mask')

    args = parser.parse_args()

    if args.config is not None:
        config_dict = config_to_dict(args.config)
        for key, value in config_dict.items():
            if not hasattr(args, key):
                continue
            current = getattr(args, key)
            if isinstance(current, bool):
                setattr(args, key, str(value).lower() in {'1', 'true', 'yes'})
            elif isinstance(current, int):
                setattr(args, key, int(value))
            elif isinstance(current, float):
                setattr(args, key, float(value))
            else:
                setattr(args, key, value)

    if args.config is not None:
        best_threshold_path = os.path.join(os.path.dirname(args.config), 'best_threshold.txt')
        if os.path.exists(best_threshold_path):
            try:
                args.pred_threshold = float(open(best_threshold_path, 'r', encoding='utf-8').read().strip())
            except Exception:
                pass

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    sam, _ = sam_model_registry[args.vit_name](
        image_size=args.img_size,
        num_classes=args.num_classes,
        checkpoint=args.ckpt,
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
    )
    if args.delta_type == 'adapter':
        pkg = import_module('delta.sam_adapter_image_encoder')
        net = pkg.Adapter_Sam(sam, args.middle_dim, args.scaling_factor).cuda()
    elif args.delta_type == 'lora':
        pkg = import_module('delta.sam_lora_image_encoder')
        net = pkg.LoRA_Sam(sam, args.rank).cuda()
    else:
        pkg = import_module('delta.sam_adapter_lora_image_encoder')
        net = pkg.LoRA_Adapter_Sam(sam, args.middle_dim, args.rank).cuda()

    if not args.delta_ckpt:
        raise ValueError("delta_ckpt is required. Pass it directly or provide it via --config.")
    net.load_delta_parameters(args.delta_ckpt)
    multimask_output = args.num_classes > 1

    log_folder = os.path.join(args.output_dir, 'test_log')
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_folder, 'log.txt'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S',
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    if args.is_savenii:
        test_save_path = os.path.join(args.output_dir, 'predictions')
        os.makedirs(test_save_path, exist_ok=True)
        os.makedirs(test_save_path + '/img/', exist_ok=True)
        os.makedirs(test_save_path + '/pred/', exist_ok=True)
        os.makedirs(test_save_path + '/gt/', exist_ok=True)
    else:
        test_save_path = None

    inference(args, multimask_output, net, test_save_path)
