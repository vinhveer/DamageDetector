import cv2
import os

# Tối ưu cho DataLoader đa luồng trên Windows
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from train_lib.cli import build_arg_parser, validate_args
from train_lib.runner import run_training


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)
    run_training(args)


if __name__ == '__main__':
    main()
