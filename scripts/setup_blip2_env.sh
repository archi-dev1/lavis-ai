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
    echo "Verifying PyTorch and BLIP2 imports..."
    python - <<'PY'
    import sys,traceback
    ok = True
    try:
        import torch
        print('torch', torch.__version__, 'cuda available=', torch.cuda.is_available())
    except Exception:
        traceback.print_exc()
        ok = False

    try:
        from lavis.models.blip2_models.blip2_t5 import Blip2T5
        print('BLIP2 import OK')
    except Exception:
        traceback.print_exc()
        ok = False

    if not ok:
        sys.exit(2)
    sys.exit(0)
    PY

    RET=$?
    if [[ $RET -ne 0 ]]; then
      echo "Initial verification failed. Attempting automatic repair..."

      echo "Removing potentially conflicting OpenMP/MKL packages (intel-openmp libiomp mkl mkl-service)..."
      mamba remove -y intel-openmp libiomp mkl mkl-service || true

      echo "Removing existing PyTorch packages..."
      mamba remove -y pytorch torchvision torchaudio "pytorch-cuda*" || true

      echo "Reinstalling PyTorch (${HAS_GPU} -> GPU=1/CPU=0)..."
      if [[ ${HAS_GPU} -eq 1 ]]; then
        mamba install -y -c pytorch -c nvidia pytorch torchvision torchaudio "pytorch-cuda=${CUDA_VER}" || true
      else
        mamba install -y -c pytorch pytorch torchvision torchaudio cpuonly || true
      fi

      echo "Re-running verification after repair..."
      python - <<'PY'
    import sys,traceback
    ok = True
    try:
        import torch
        print('torch', torch.__version__, 'cuda available=', torch.cuda.is_available())
    except Exception:
        traceback.print_exc()
        ok = False

    try:
        from lavis.models.blip2_models.blip2_t5 import Blip2T5
        print('BLIP2 import OK')
    except Exception:
        traceback.print_exc()
        ok = False

    if not ok:
        sys.exit(2)
    sys.exit(0)
    PY

      RET2=$?
      if [[ $RET2 -ne 0 ]]; then
        echo "Repair reinstall did not fix import. Trying LD_PRELOAD workaround..."
        CONDA_PREFIX_ACT=$(conda info --base)/envs/${ENV_NAME}
        LIBIOMP_PATH=$(find "$CONDA_PREFIX_ACT" -name 'libiomp*.so' | head -n1 || true)
        if [[ -n "$LIBIOMP_PATH" ]]; then
          echo "Found libiomp at: $LIBIOMP_PATH. Testing LD_PRELOAD..."
          LD_PRELOAD="$LIBIOMP_PATH" python - <<'PY'
    import sys,traceback
    try:
        import torch
        print('torch', torch.__version__, 'cuda available=', torch.cuda.is_available())
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        sys.exit(3)
    PY
          if [[ $? -eq 0 ]]; then
            echo "LD_PRELOAD workaround fixed the import. Creating activate.d to set LD_PRELOAD on env activation."
            ACT_DIR="$CONDA_PREFIX_ACT/etc/conda/activate.d"
            mkdir -p "$ACT_DIR"
            echo "export LD_PRELOAD=\"$LIBIOMP_PATH\"" > "$ACT_DIR/ld_preload.sh"
            echo "Wrote $ACT_DIR/ld_preload.sh"
          else
            echo "LD_PRELOAD test failed. Manual inspection required. See setup log for details."
          fi
        else
          echo "No libiomp found in env; unable to try LD_PRELOAD workaround. Manual fix required."
        fi
        echo "Setup finished with errors — PyTorch import still failing. Inspect and retry." >&2
        exit 3
      fi
    fi

    echo "Setup finished successfully. You can start training with:"
    echo "  tmux new -s coco_train && conda activate ${ENV_NAME} && export KAGGLE_USERNAME=your_user && export KAGGLE_KEY=your_key && cd ${REPO_DIR} && ./run_coco_server.sh train"

    exit 0
