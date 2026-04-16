#!/usr/bin/env bash
set -euo pipefail

# setup_blip2_env.sh
# Creates a conda environment, installs PyTorch (GPU or CPU), repo deps,
# and runs verification checks for BLIP2 training.
#
# Usage:
#   ./scripts/setup_blip2_env.sh [--env-name NAME] [--gpu|--cpu] [--cuda 11.8] [--force]
# Defaults: env-name=blip2_env, GPU if nvidia-smi present, cuda=11.8

ENV_NAME=blip2_env
FORCE_RECREATE=0
CUDA_VER=11.8
PREF_CPU=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="$2"; shift 2;;
    --force) FORCE_RECREATE=1; shift;;
    --cpu) PREF_CPU=1; shift;;
    --gpu) PREF_CPU=0; shift;;
    --cuda) CUDA_VER="$2"; shift 2;;
    -h|--help) echo "Usage: $0 [--env-name NAME] [--gpu|--cpu] [--cuda 11.8] [--force]"; exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "Repo dir: ${REPO_DIR}"

# Ensure conda is available and source its profile
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH. Please install Miniconda/Anaconda and retry." >&2
  exit 1
fi

CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1090
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Install mamba in base if missing
if ! command -v mamba >/dev/null 2>&1; then
  echo "Installing mamba in base environment..."
  conda activate base
  conda install -n base -c conda-forge mamba -y
  conda deactivate || true
fi

# GPU detection
if [[ ${PREF_CPU} -eq 0 ]] && command -v nvidia-smi >/dev/null 2>&1; then
  HAS_GPU=1
else
  HAS_GPU=0
fi

echo "Env: ${ENV_NAME} | GPU available: ${HAS_GPU} | CUDA: ${CUDA_VER}"

# Recreate env if requested
if conda env list | awk '{print $1}' | grep -xq "${ENV_NAME}"; then
  if [[ ${FORCE_RECREATE} -eq 1 ]]; then
    echo "Removing existing environment ${ENV_NAME}..."
    mamba env remove -n "${ENV_NAME}" -y || conda env remove -n "${ENV_NAME}" -y
  else
    echo "Environment ${ENV_NAME} already exists — will reuse it. Use --force to recreate."
  fi
fi

# Create environment if not exists
if ! conda env list | awk '{print $1}' | grep -xq "${ENV_NAME}"; then
  echo "Creating conda environment ${ENV_NAME} (python=3.10)..."
  mamba create -n "${ENV_NAME}" python=3.10 -y
fi

echo "Activating ${ENV_NAME}..."
conda activate "${ENV_NAME}"

# Install PyTorch
if [[ ${HAS_GPU} -eq 1 ]] && [[ ${PREF_CPU} -eq 0 ]]; then
  echo "Installing PyTorch GPU build with CUDA ${CUDA_VER}..."
  mamba install -y -c pytorch -c nvidia pytorch torchvision torchaudio "pytorch-cuda=${CUDA_VER}"
else
  echo "Installing PyTorch CPU-only build..."
  mamba install -y -c pytorch pytorch torchvision torchaudio cpuonly
fi

echo "Installing repository (editable) and COCO workflow deps via pip..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${REPO_DIR}" --no-deps

python -m pip install \
  contexttimer \
  decord \
  "einops>=0.4.1" \
  fairscale==0.4.4 \
  ftfy \
  iopath \
  omegaconf \
  opencv-python-headless==4.5.5.64 \
  packaging \
  pandas \
  pycocoevalcap \
  pycocotools \
  pyyaml_env_tag==0.1 \
  scikit-image \
  sentencepiece \
  "transformers==4.46.2" \
  "peft==0.13.2" \
  "timm==1.0.15" \
  tqdm \
  webdataset \
  kaggle \
  easydict==1.9 \
  "numpy<2"

echo "Verifying PyTorch and BLIP2 imports..."
python - <<'PY'
import sys
try:
    import torch
    print('torch', torch.__version__, 'cuda available=', torch.cuda.is_available())
except Exception as e:
    print('ERROR importing torch:', e)
    sys.exit(2)

try:
    from lavis.models.blip2_models.blip2_t5 import Blip2T5
    print('BLIP2 import OK')
except Exception as e:
    print('ERROR importing BLIP2:', e)
    sys.exit(3)
PY

echo "Setup finished. If verification succeeded, you can start training with:"
echo "  tmux new -s coco_train && conda activate ${ENV_NAME} && export KAGGLE_USERNAME=your_user && export KAGGLE_KEY=your_key && cd ${REPO_DIR} && ./run_coco_server.sh train"

exit 0
