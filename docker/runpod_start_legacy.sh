#!/usr/bin/env bash
# Bootstrap paper-era (legacy) logit distill on a stock RunPod PyTorch image.
# Idempotent: safe to re-run when the persistent /workspace already has the repo.
#
# Installs requirements-legacy.txt (custom transformers fork + older accelerate/datasets).
# Does not upgrade/downgrade the image torch — keeps the CUDA torch shipped with the image.
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
pip install --upgrade pip
# Paper-era HF stack (fork). Torch stays whatever the RunPod image provides.
# Force reinstall pins so a previous bad install (e.g. pyarrow 25) cannot stick.
pip install --upgrade --force-reinstall -r requirements-legacy.txt

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
