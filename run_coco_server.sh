#!/usr/bin/env bash

set -euo pipefail

ENV_NAME="${ENV_NAME:-blip_source_env}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Repo dir: ${REPO_DIR}"

if command -v conda >/dev/null 2>&1; then
    CONDA_BASE="$(conda info --base)"
    # shellcheck disable=SC1091
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
    echo "Activated conda environment: ${ENV_NAME}"
else
    echo "conda not found on PATH. Activate your environment manually, then rerun this script."
    exit 1
fi

cd "${REPO_DIR}"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

# Required for the PVT + QFormer LoRA training path used by train_coco_server.py.
python -m pip install \
    "transformers==4.46.2" \
    "peft==0.13.2" \
    "timm==1.0.15" \
    kaggle

echo
echo "Starting COCO training / inference pipeline..."
echo "The Python script will ask for your Kaggle username and API key if they are not already configured."
echo

python train_coco_server.py "$@"