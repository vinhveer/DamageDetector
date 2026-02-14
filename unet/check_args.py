import sys
from unittest.mock import MagicMock
sys.modules['cv2'] = MagicMock()
sys.modules['cv2'].setNumThreads = MagicMock()
sys.modules['cv2'].ocl = MagicMock()

import os
# Add unet to path so we can import train_lib
sys.path.append(os.path.join(os.getcwd(), 'unet'))

from train_lib.cli import build_arg_parser

if __name__ == "__main__":
    parser = build_arg_parser()
    parser.print_help()
