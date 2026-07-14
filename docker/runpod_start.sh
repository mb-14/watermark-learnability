#!/usr/bin/env bash
# Bootstrap logit distill on a stock RunPod PyTorch image (no custom registry needed).
set -euo pipefail

REPO_URL=${REPO_URL:-https://github.com/mb-14/watermark-learnability.git}
REPO_BRANCH=${REPO_BRANCH:-modernize-logit-distill}
WORK=/workspace/watermark-learnability

export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-/workspace/.cache/torch}
export OUTPUT_DIR=${OUTPUT_DIR:-/workspace/out/}
mkdir -p "${HF_HOME}" "${OUTPUT_DIR}"

if [[ ! -d "${WORK}/.git" ]]; then
  git clone --branch "${REPO_BRANCH}" --depth 1 "${REPO_URL}" "${WORK}"
else
  git -C "${WORK}" fetch --depth 1 origin "${REPO_BRANCH}"
  git -C "${WORK}" checkout "${REPO_BRANCH}"
  git -C "${WORK}" pull --ff-only origin "${REPO_BRANCH}" || true
fi

cd "${WORK}"
pip install --upgrade pip
pip install -r requirements.txt
pip install "huggingface_hub[cli]>=0.24.0"

# Reuse the Docker entrypoint logic (expects this path).
exec bash docker/entrypoint_logit_distill.sh
