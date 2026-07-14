#!/usr/bin/env bash
# Bootstrap logit distill on a stock RunPod PyTorch image (no custom registry needed).
# Idempotent: safe to re-run when the persistent /workspace already has the repo.
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/mb-14/watermark-learnability.git}
REPO_BRANCH=${REPO_BRANCH:-modernize-logit-distill}
WORK=/workspace/watermark-learnability
LOG_DIR=${LOG_DIR:-/workspace/logs}
LOG_FILE="${LOG_DIR}/logit_distill_${KGW_HASH_KEY:-nokey}_$(date -u +%Y%m%dT%H%M%SZ).log"

export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/workspace/.cache/torch}
export OUTPUT_DIR=${OUTPUT_DIR:-/workspace/out/}
mkdir -p "${HF_HOME}" "${OUTPUT_DIR}" "${LOG_DIR}"

if [[ ! -d "${WORK}/.git" ]]; then
  git clone --branch "${REPO_BRANCH}" --depth 1 "${REPO_URL}" "${WORK}"
else
  # Best-effort update; do not fail the job if the volume is temporarily full.
  if ! git -C "${WORK}" fetch --depth 1 origin "${REPO_BRANCH}"; then
    echo "Warning: git fetch failed (disk full or network). Continuing with existing checkout."
  else
    git -C "${WORK}" checkout "${REPO_BRANCH}" || true
    git -C "${WORK}" reset --hard "origin/${REPO_BRANCH}" || true
  fi
fi

cd "${WORK}"
pip install --upgrade pip
pip install -r requirements.txt
pip install "huggingface_hub[cli]>=0.24.0"
# Do not upgrade torch here: bumping past the image's torchvision breaks imports
# (torchvision::nms). Resume uses a transformers torch.load bypass instead.

# Keep container alive for SSH even if training exits; log to a file.
set +e
bash docker/entrypoint_logit_distill.sh 2>&1 | tee -a "${LOG_FILE}"
train_status=${PIPESTATUS[0]}
set -e
echo "Training exited with code ${train_status}. Log: ${LOG_FILE}" | tee -a "${LOG_FILE}"

# Prefer an interactive shell so RunPod SSH / console stay usable after training.
if [[ -t 0 ]]; then
  exec bash
else
  exec sleep infinity
fi
