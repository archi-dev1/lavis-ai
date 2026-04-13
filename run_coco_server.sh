#!/usr/bin/env bash
#
# Usage:
#   ./run_coco_server.sh train                          # train only
#   ./run_coco_server.sh train --epochs 5               # train with overrides
#   ./run_coco_server.sh infer --checkpoint path/to.pt  # inference only
#   ./run_coco_server.sh all                            # train then infer on best model
#   ./run_coco_server.sh                                # defaults to 'all'

set -euo pipefail

ENV_NAME="${ENV_NAME:-blip_source_env}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${REPO_DIR}/output/blip2_coco_pvt_lora"

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
# Install the package itself without repo-wide optional dependencies that are
# unrelated to the COCO captioning workflow (for example, open3d).
python -m pip install -e . --no-deps

# Required for the PVT + QFormer LoRA COCO training/inference workflow.
python -m pip install \
    contexttimer \
    decord \
    einops>=0.4.1 \
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
    "easydict==1.9" \
    "numpy<2"

MODE="${1:-all}"
shift 2>/dev/null || true   # consume the mode arg; remaining args forwarded

case "${MODE}" in
    train)
        echo
        echo "=== Starting COCO training ==="
        python train_coco_server.py "$@"
        ;;
    infer|inference)
        echo
        echo "=== Starting COCO inference ==="
        python infer_coco_server.py "$@"
        ;;
    all)
        echo
        echo "=== Starting COCO training ==="
        python train_coco_server.py "$@"

        BEST="${OUTPUT_DIR}/best_model.pt"
        if [[ -f "${BEST}" ]]; then
            echo
            echo "=== Training done. Running inference on best model ==="
            python infer_coco_server.py --checkpoint "${BEST}"
        else
            echo "Warning: best_model.pt not found at ${BEST}. Skipping inference."
        fi
        ;;
    *)
        echo "Unknown mode: ${MODE}"
        echo "Usage: $0 {train|infer|all} [extra args...]"
        exit 1
        ;;
esac