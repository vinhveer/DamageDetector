#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/kaggle/working/DamageDetector}"

python -m pip install --upgrade pip
python -m pip install -e "$REPO_DIR"
python -m pip install open-clip-torch pillow tqdm pandas

python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('cuda_count', torch.cuda.device_count())"
