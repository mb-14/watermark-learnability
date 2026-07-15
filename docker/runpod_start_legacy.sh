#!/usr/bin/env bash
# Bootstrap paper-era (legacy) logit distill on a stock RunPod PyTorch image.
# Idempotent: safe to re-run when the persistent /workspace already has the repo.
#
# Installs requirements-legacy.txt (custom transformers fork + older accelerate/datasets).
# Keeps the image's torch: constraints file pins the currently-installed torch build.
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/mb-14/watermark-learnability.git}
REPO_BRANCH=${REPO_BRANCH:-modernize-logit-distill}
WORK=/workspace/watermark-learnability
LOG_DIR=${LOG_DIR:-/workspace/logs}
LOG_FILE="${LOG_DIR}/logit_distill_legacy_${KGW_HASH_KEY:-nokey}_$(date -u +%Y%m%dT%H%M%SZ).log"

export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/workspace/.cache/torch}
export OUTPUT_DIR=${OUTPUT_DIR:-/workspace/out/}
mkdir -p "${HF_HOME}" "${OUTPUT_DIR}" "${LOG_DIR}"

if [[ ! -d "${WORK}/.git" ]]; then
  git clone --branch "${REPO_BRANCH}" --depth 1 "${REPO_URL}" "${WORK}"
else
  if ! git -C "${WORK}" fetch --depth 1 origin "${REPO_BRANCH}"; then
    echo "Warning: git fetch failed (disk full or network). Continuing with existing checkout."
  else
    git -C "${WORK}" checkout "${REPO_BRANCH}" || true
    git -C "${WORK}" reset --hard "origin/${REPO_BRANCH}" || true
  fi
fi

cd "${WORK}"
# Prefer standalone `hf` CLI for Hub uploads (Trainer native/Xet push is flaky on large shards).
# Keep this separate from the pinned legacy huggingface-hub==0.16.4 used for training.
export PATH="${HOME}/.local/bin:${PATH}"
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
if ! command -v hf >/dev/null 2>&1; then
  echo "Installing standalone Hugging Face CLI (hf)..."
  curl -LsSf https://hf.co/cli/install.sh | bash -s
  export PATH="${HOME}/.local/bin:${PATH}"
fi
hf version || true
# git-lfs still useful for base model downloads via older hub APIs.
if ! command -v git-lfs >/dev/null 2>&1; then
  apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y git-lfs
  git lfs install
fi
pip install --upgrade pip

# If a previous broken legacy install yanked image torch, restore a CUDA 12.4 build
# matching runpod/pytorch:2.4.0-*-cuda12.4.*. Otherwise keep whatever is installed.
python - <<'PY'
import importlib.util
import subprocess
import sys

spec = importlib.util.find_spec("torch")
need_restore = False
if spec is None:
    need_restore = True
    print("torch missing; will restore torch==2.4.0+cu124")
else:
    import torch
    ver = torch.__version__
    print(f"found torch {ver} cuda={getattr(torch.version, 'cuda', None)}")
    # Force-reinstall previously pulled torch 2.13 / cu13; reject that.
    major_minor = tuple(int(x) for x in ver.split("+")[0].split(".")[:2])
    cuda = getattr(torch.version, "cuda", None) or ""
    if major_minor >= (2, 5) or cuda.startswith("13"):
        need_restore = True
        print(f"torch {ver} is too new / wrong CUDA for legacy A/B; restoring 2.4.0+cu124")

if need_restore:
    # Drop leftover CUDA-13 nvidia-* wheels from a prior force-reinstall.
    loose = subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze"], text=True
    )
    purge = [
        line.split("==")[0]
        for line in loose.splitlines()
        if line.startswith("nvidia-") and ("cu13" in line or "cuda-toolkit==13" in line)
    ]
    purge += [line.split("==")[0] for line in loose.splitlines() if line.startswith("cuda-toolkit==13") or line.startswith("cuda-bindings==")]
    # Also common leftover NCCL from torch 2.13.
    for pkg in list(purge):
        if "nccl" in pkg.lower() and "cu13" in pkg:
            pass
    if purge:
        print("purging leftover CUDA13 pkgs:", purge)
        subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", *purge])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch==2.4.0", "torchaudio==2.4.0",
        "--index-url", "https://download.pytorch.org/whl/cu124",
    ])
PY
# Avoid NVLS multicast failures on some RunPod H100 machines (NCCL error 401).
export NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}

# Pin torch/torchaudio so accelerate etc. cannot upgrade them.
python - <<'PY'
from pathlib import Path
import importlib.metadata as md
lines = []
for pkg in ("torch", "torchaudio", "torchvision"):
    try:
        lines.append(f"{pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        pass
Path("/tmp/legacy-torch-constraints.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
print("constraints:", Path("/tmp/legacy-torch-constraints.txt").read_text().strip() or "<empty>")
PY

# Install legacy HF stack; do NOT force-reinstall (avoids yanking CUDA torch again).
pip install -r requirements-legacy.txt \
  --constraint /tmp/legacy-torch-constraints.txt \
  --upgrade-strategy only-if-needed
# Force era fsspec: hub may leave >=2023.12 which breaks datasets 2.13 globs.
pip install 'fsspec==2023.9.2' \
  --constraint /tmp/legacy-torch-constraints.txt

# Keep container alive for SSH even if training exits; log to a file.
set +e
bash docker/entrypoint_logit_distill_legacy.sh 2>&1 | tee -a "${LOG_FILE}"
train_status=${PIPESTATUS[0]}
set -e
echo "Legacy training exited with code ${train_status}. Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"

if [[ -t 0 ]]; then
  exec bash
else
  exec sleep infinity
fi
