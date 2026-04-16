import argparse
import os
import random
import numpy as np
from torch_runtime import cudnn, torch
from importlib import import_module
from segment_anything import sam_model_registry
from trainer import trainer_generic


NUM_CLASSES = 1


parser = argparse.ArgumentParser(description='Train SAM finetune for crack segmentation using the GenericDataset folder layout.')
parser.add_argument('--root_path', type=str, required=True, help='Training dataset root containing images/ and masks/')
parser.add_argument('--val_path', type=str, required=True, help='Validation dataset root containing images/ and masks/')
parser.add_argument('--output', type=str, default='./output/training')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum iterations number to train')
parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--stop_epoch', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=12, help='batch_size per gpu')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=448, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=3407, help='random seed')
parser.add_argument('--vit_name', type=str,
                    default='vit_h', help='select one vit model')
parser.add_argument('--ckpt', type=str, default='checkpoints/sam_vit_h_4b8939.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--delta_ckpt', type=str, default=None, help='Finetuned delta checkpoint')
parser.add_argument('--delta_type', type=str, default='adapter', help='choose from "adapter" or "lora" or "both"')
parser.add_argument('--middle_dim', type=int, default=32, help='Middle dim of adapter')
parser.add_argument('--scaling_factor', type=float, default=0.1, help='Scaling_factor of adapter')
parser.add_argument('--rank', type=int, default=4, help='Rank for LoRA adaptation')
parser.add_argument('--decoder_type', type=str, default='baseline', choices=['baseline', 'hq'], help='Mask decoder type')
parser.add_argument('--centerline_head', action='store_true', help='Enable a dedicated centerline prediction head in the decoder')
parser.add_argument('--decoder_lr_mult', type=float, default=None, help='Learning-rate multiplier applied to trainable decoder params (default: 0.25 for HQ, 0.1 otherwise)')
parser.add_argument('--hq_trainable_mode', type=str, default='balanced', choices=['hq_only', 'balanced'], help='Trainable policy for HQ decoder params')
parser.add_argument('--warmup', action='store_true', help='If activated, warp up the learning from a lower lr to the base_lr')
parser.add_argument('--warmup_period', type=int, default=300,
                    help='Warp up iterations, only valid whrn warmup is activated')
parser.add_argument('--AdamW', action='store_true', default=True, help='If activated, use AdamW to finetune SAM model')
parser.add_argument('--dice_param', type=float, default=None, help='Legacy alias for --dice_weight')
parser.add_argument('--lr_exp', type=float, default=0.9, help='The learning rate decay expotential')
parser.add_argument('--tf32', action='store_true', help='If activated, use tf32 to accelerate the training process')
parser.add_argument('--use_amp', action='store_true', help='If activated, adopt mixed precision for acceleration, but may cause NaN')
parser.add_argument('--save_interval', type=int, default=1, help='Save and validation intervals')
parser.add_argument('--num_workers', type=int, default=8, help='number of dataloader workers')
parser.add_argument('--patches_per_image', type=int, default=1, help='number of random patches to crop per image per epoch')
parser.add_argument('--train_use_boxes', type=int, default=1, help='Use box prompts during training')
parser.add_argument('--train_full_image_box', action='store_true', help='Use a full-image box prompt for every training sample')
parser.add_argument('--train_use_points_prob', type=float, default=0.1, help='Legacy mode only: probability of adding GT points during training')
parser.add_argument('--prompt_policy', type=str, default='hybrid_v1', choices=['hybrid', 'hybrid_v1', 'hybrid_val_aligned', 'hybrid_val_balanced', 'hybrid_tight_heavy', 'points_heavy', 'legacy'], help='Training prompt policy preset')
parser.add_argument('--background_crop_prob', type=float, default=0.2, help='Probability of sampling a random background crop even when crack exists')
parser.add_argument('--near_background_crop_prob', type=float, default=0.15, help='Probability of sampling a crop near but not centered on the crack')
parser.add_argument('--hard_negative_crop_prob', type=float, default=0.10, help='Probability of sampling a hard-negative background crop')
parser.add_argument('--augment_profile', type=str, default='balanced', choices=['balanced', 'aggressive'], help='Augmentation profile for crack segmentation')
parser.add_argument('--tversky_alpha', type=float, default=0.3, help='False-positive weight in Tversky loss')
parser.add_argument('--tversky_beta', type=float, default=0.7, help='False-negative weight in Tversky loss')
parser.add_argument('--bce_weight', type=float, default=1.0, help='Weight for BCEWithLogits loss')
parser.add_argument('--dice_weight', type=float, default=0.35, help='Weight for Dice loss')
parser.add_argument('--tversky_weight', type=float, default=0.35, help='Weight for Tversky loss')
parser.add_argument('--focal_weight', type=float, default=0.25, help='Weight for binary focal loss')
parser.add_argument('--focal_alpha', type=float, default=0.25, help='Alpha for binary focal loss')
parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma for binary focal loss')
parser.add_argument('--cldice_weight', type=float, default=0.0, help='Weight for clDice topology-aware loss')
parser.add_argument('--cldice_iters', type=int, default=20, help='Soft skeleton iterations for clDice')
parser.add_argument('--centerline_aux_weight', type=float, default=0.0, help='Weight for centerline auxiliary supervision on main logits')
parser.add_argument('--pos_weight', type=str, default='auto', help="Positive class weight for BCE or 'auto'")
parser.add_argument('--pos_weight_min', type=float, default=1.0, help='Minimum auto positive class weight')
parser.add_argument('--pos_weight_max', type=float, default=20.0, help='Maximum auto positive class weight')
parser.add_argument('--pos_weight_sample', type=int, default=200, help='Number of masks sampled for auto pos weight')
parser.add_argument('--val_thresholds', type=float, nargs='+', default=[0.35, 0.4, 0.45, 0.5, 0.55, 0.6], help='Candidate probability thresholds for validation model selection')
parser.add_argument('--full_image_eval', action='store_true', help='Validate with a full-image box prompt instead of dataset-provided boxes')
parser.add_argument('--tile_overlap', type=int, default=-1, help='Tile overlap for tiled validation/inference (-1 = img_size // 2)')
parser.add_argument('--legacy_box_eval', action='store_true', help='Also run legacy crop+GT-box validation diagnostics')
parser.add_argument('--pipeline_stage', type=str, default='coarse', choices=['coarse', 'refine'], help='Training stage for the pipeline')
parser.add_argument('--roi_size', type=int, default=768, help='High-resolution ROI size for refine-stage training')
parser.add_argument('--roi_positive_band_low', type=float, default=0.20, help='Low end of coarse-score ROI mining band')
parser.add_argument('--roi_positive_band_high', type=float, default=0.90, help='High end of coarse-score ROI mining band')
parser.add_argument('--extra_train_roots', type=str, nargs='*', default=None, help='Additional labeled training dataset roots to mix into training')
parser.add_argument('--pseudo_label_roots', type=str, nargs='*', default=None, help='Additional pseudo-labeled dataset roots to mix into training')
parser.add_argument('--refine_enabled', action='store_true', help='Persist coarse_refine metadata in the inference sidecar for downstream runtime/test usage')
parser.add_argument('--refine_tile_size', type=int, default=768, help='Default fixed ROI size for coarse_refine inference')
parser.add_argument('--refine_tile_sizes', type=int, nargs='*', default=None, help='Optional multi-scale ROI sizes for coarse_refine inference')
parser.add_argument('--refine_max_rois', type=int, default=16, help='Default max number of refine ROIs per image')
parser.add_argument('--refine_roi_padding', type=int, default=64, help='Default padding around mined refine ROIs')
parser.add_argument('--refine_merge_mode', type=str, default='weighted_replace', help='Default coarse/refine merge mode')
parser.add_argument('--refine_score_threshold', type=float, default=0.15, help='Default ROI mining heat threshold')

args = parser.parse_args()


def _configure_hq_lora_training(net, mode: str) -> None:
    sam_model = getattr(net, "sam", net)
    mask_decoder = getattr(sam_model, "mask_decoder", None)
    if mask_decoder is None:
        return
    if hasattr(mask_decoder, "set_trainable_mode"):
        mask_decoder.set_trainable_mode(mode)
    elif hasattr(mask_decoder, "set_hq_only_trainable"):
        mask_decoder.set_hq_only_trainable()


def _apply_hq_balancing_defaults(args) -> None:
    decoder_type = str(getattr(args, "decoder_type", "baseline")).strip().lower()
    if args.decoder_lr_mult is None:
        args.decoder_lr_mult = 0.25 if decoder_type == "hq" else 0.1

    if decoder_type != "hq":
        return

    prompt_policy = str(getattr(args, "prompt_policy", "hybrid_v1")).strip().lower()
    if prompt_policy in {"hybrid", "hybrid_v1"}:
        args.prompt_policy = "hybrid_val_balanced"

    if abs(float(args.tversky_alpha) - 0.3) < 1e-8 and abs(float(args.tversky_beta) - 0.7) < 1e-8:
        args.tversky_alpha = 0.4
        args.tversky_beta = 0.6

    if str(getattr(args, "pipeline_stage", "coarse")).strip().lower() == "refine" and prompt_policy in {"hybrid", "hybrid_v1"}:
        args.prompt_policy = "hybrid_val_balanced"

if __name__ == "__main__":
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
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
    if args.dice_param is not None:
        args.dice_weight = float(args.dice_param)
    _apply_hq_balancing_defaults(args)
    if str(getattr(args, "pipeline_stage", "coarse")).strip().lower() == "refine":
        args.img_size = int(getattr(args, "roi_size", args.img_size))
        args.refine_enabled = True
    if float(getattr(args, "centerline_aux_weight", 0.0)) > 0.0:
        args.centerline_head = True
    args.is_pretrain = True
    args.exp = 'generic_' + str(args.img_size)
    snapshot_path = os.path.join(args.output, "{}".format(args.exp))
    snapshot_path = snapshot_path + '_pretrain' if args.is_pretrain else snapshot_path
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[
                                          :-3] + 'k' 
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) 
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) 
    snapshot_path = snapshot_path + '_s' + str(args.seed) 
    snapshot_path = snapshot_path + '_type_' + str(args.delta_type) 
    snapshot_path = snapshot_path + '_stage_' + str(getattr(args, 'pipeline_stage', 'coarse'))
    if args.delta_type =='adapter':
        snapshot_path = snapshot_path + '_dim' + str(args.middle_dim)         
        snapshot_path = snapshot_path + '_sf' + str(args.scaling_factor)   
    elif args.delta_type =='lora':
        snapshot_path = snapshot_path + '_r' + str(args.rank)
    else:
        snapshot_path = snapshot_path + '_dim' + str(args.middle_dim)                          
        snapshot_path = snapshot_path + '_r' + str(args.rank)

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # register model
    sam, img_embedding_size = sam_model_registry[args.vit_name](image_size=args.img_size,
                                                                num_classes=NUM_CLASSES,
                                                                checkpoint=args.ckpt, 
                                                                pixel_mean=[0.485, 0.456, 0.406],
                                                                pixel_std=[0.229, 0.224, 0.225],
                                                                decoder_type=args.decoder_type,
                                                                centerline_head=bool(getattr(args, "centerline_head", False)))

    if args.delta_type == 'adapter':
        pkg = import_module('delta.sam_adapter_image_encoder')
        net = pkg.Adapter_Sam(sam, args.middle_dim, args.scaling_factor).cuda()
    elif args.delta_type == 'lora':
        pkg = import_module('delta.sam_lora_image_encoder') 
        net = pkg.LoRA_Sam(sam, args.rank).cuda()
    else:
        pkg = import_module('delta.sam_adapter_lora_image_encoder') 
        net = pkg.LoRA_Adapter_Sam(sam, args.middle_dim, args.rank).cuda()    

    if args.delta_ckpt is not None:
        net.load_delta_parameters(args.delta_ckpt)

    if str(args.decoder_type).strip().lower() == "hq":
        _configure_hq_lora_training(net, args.hq_trainable_mode)

    multimask_output = False

    low_res = img_embedding_size * 4  # It's better to use high resolution in crack segmentation

    config_file = os.path.join(snapshot_path, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        config_items.append(f'{key}: {value}\n')
    config_items.append(f'num_classes: {NUM_CLASSES}\n')
    config_items.append('dataset_backend: generic\n')

    with open(config_file, 'w') as f:
        f.writelines(config_items)

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total number of parameters:{total_params}")

    total_params_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters:{total_params_train}")

    trainer_generic(args, net, snapshot_path, multimask_output, low_res)
