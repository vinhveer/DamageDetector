import cv2
import os

def _apply_thread_workarounds() -> None:
    force_limit = str(os.environ.get("DAMAGEDETECTOR_LIMIT_CPU_THREADS", "")).strip().lower() in {"1", "true", "yes"}
    if os.name == "nt" or force_limit:
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")


_apply_thread_workarounds()

from .train_lib.cli import build_arg_parser, validate_args
from .train_lib.runner import run_training


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    validate_args(args)
    run_training(args)


if __name__ == '__main__':
    main()
