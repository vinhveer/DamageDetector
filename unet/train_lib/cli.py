import argparse
import os

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train U-Net for crack segmentation.")
    
    # Dataset paths
    parser.add_argument("--train-images", type=str, required=True, help="Path to training images folder")
    parser.add_argument("--train-masks", type=str, required=True, help="Path to training masks folder")
    parser.add_argument("--val-images", type=str, required=True, help="Path to validation images folder")
    parser.add_argument("--val-masks", type=str, required=True, help="Path to validation masks folder")
    parser.add_argument("--mask-prefix", type=str, default="auto", help="Prefix for mask files (default: auto)")
    
    # Output and logging
    parser.add_argument("--output-dir", type=str, default="output_results", help="Directory to save results")
    parser.add_argument("--visualize", dest="no_visualize", action="store_false", help="Enable visualization (default: disabled)")
    parser.add_argument("--loss-curve", dest="no_loss_curve", action="store_false", help="Enable loss curve plotting (default: disabled)")
    parser.add_argument("--visualize-every", type=int, default=0, help="Visualize every N epochs (0 to disable)")
    parser.set_defaults(no_visualize=True, no_loss_curve=True)

    # Preprocessing
    parser.add_argument("--preprocess", type=str, default="patch", choices=["patch", "letterbox", "stretch"], help="Preprocessing method")
    parser.add_argument("--preprocess-train", type=str, default=None, help="Preprocessing method for training (overrides --preprocess)")
    parser.add_argument("--preprocess-val", type=str, default=None, help="Preprocessing method for validation (overrides --preprocess)")
    parser.add_argument("--input-size", type=int, default=256, help="Input size for the model")
    parser.add_argument("--patches-per-image", type=int, default=1, help="Number of patches per image (for patch mode)")
    parser.add_argument("--max-patch-tries", type=int, default=5, help="Max tries to find a patch with crack")
    parser.add_argument("--val-stride", type=int, default=0, help="Stride for validation patching (0 = input_size)")
    
    # Augmentation
    parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    parser.add_argument("--aug-prob", type=float, default=0.5, help="Augmentation probability")
    parser.add_argument("--rotate-limit", type=int, default=10, help="Rotation limit in degrees")
    parser.add_argument("--brightness-limit", type=float, default=0.2, help="Brightness limit")
    parser.add_argument("--contrast-limit", type=float, default=0.2, help="Contrast limit")

    # Caching
    parser.add_argument("--cache-data", action="store_true", help="Cache data in memory")
    parser.add_argument("--cache-dir", type=str, default=None, help="Directory to cache preprocessed images")
    parser.add_argument("--cache-rebuild", action="store_true", help="Rebuild cache")
    
    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs", type=int, default=80, help="Number of epochs")
    parser.add_argument("--learning-rate", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.00001, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--early-stop-patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    
    # Model
    parser.add_argument("--model-config", type=str, default="unet/model_config.yaml", help="Path to model config YAML")
    
    # Loss weights
    parser.add_argument("--pos-weight", type=str, default="5.0", help="Positive class weight (or 'auto')")
    parser.add_argument("--pos-weight-min", type=float, default=1.0, help="Min positive weight")
    parser.add_argument("--pos-weight-max", type=float, default=20.0, help="Max positive weight")
    parser.add_argument("--pos-weight-sample", type=int, default=200, help="Samples for pos weight calculation")
    parser.add_argument("--bce-weight", type=float, default=0.4, help="BCE loss weight")
    parser.add_argument("--dice-weight", type=float, default=0.6, help="Dice loss weight")
    parser.add_argument("--focal-weight", type=float, default=0.0, help="Focal loss weight")
    parser.add_argument("--focal-alpha", type=float, default=0.25, help="Focal loss alpha")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma")
    
    # Metrics and Scheduler
    parser.add_argument("--metric-threshold", type=float, default=0.5, help="Threshold for metrics")
    parser.add_argument("--metric-thresholds", type=str, default="", help="Comma-separated thresholds")
    parser.add_argument("--scheduler-metric", type=str, default="loss", help="Metric for scheduler")
    parser.add_argument("--scheduler-factor", type=float, default=0.5, help="Scheduler factor")
    parser.add_argument("--scheduler-patience", type=int, default=10, help="Scheduler patience")
    parser.add_argument("--scheduler-t0", type=int, default=10, help="Scheduler T0")
    parser.add_argument("--scheduler-tmult", type=int, default=2, help="Scheduler T_mult")
    
    # DataLoader
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Prefetch factor")
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false", help="Disable persistent workers")
    parser.set_defaults(persistent_workers=True)
    parser.add_argument("--pin-memory", action="store_true", help="Pin memory")

    return parser

def validate_args(args):
    # Already handled by required=True in argparse, but can add custom checks if needed
    pass
