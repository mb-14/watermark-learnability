#!/usr/bin/env bash
# Env-driven entrypoint for logit watermark distillation (Docker / RunPod).
#
# Required:
#   WATERMARK_TYPE   e.g. kgw-k1-gamma0.25-delta2
#
# Common optional:
#   KGW_HASH_KEY     default 15485863 (KGW only)
#   OUTPUT_DIR       default /workspace/out/
#   MASTER_PORT      default 29500
#   MODEL_NAME_OR_PATH  default meta-llama/Llama-2-7b-hf
#   NPROC_PER_NODE   default 4
#   HF_TOKEN or RUNPOD_SECRET_HF_TOKEN  Hugging Face token (gated Llama + Hub push)
#   HF_HUB_USER      default mbakshi1094 (used to build HUB_MODEL_ID if unset)
#   PUSH_TO_HUB      true/false (default false)
#   HUB_MODEL_ID     optional; default ${HF_HUB_USER}/llama-2-7b-logit-watermark-distill-${WATERMARK_TYPE}-hk${KGW_HASH_KEY}
#   HUB_PRIVATE_REPO true/false (default false)
#   ATTN_IMPLEMENTATION  sdpa | flash_attention_2 (default sdpa)
#   TORCH_COMPILE    True/False (default False)
#   MAX_STEPS        override training steps
#   EXTRA_ARGS       extra CLI args appended to the train command
set -euo pipefail

cd /workspace/watermark-learnability

WATERMARK_TYPE=${WATERMARK_TYPE:?Set WATERMARK_TYPE (e.g. kgw-k1-gamma0.25-delta2)}
OUTPUT_DIR=${OUTPUT_DIR:-/workspace/out/}
MASTER_PORT=${MASTER_PORT:-29500}
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-meta-llama/Llama-2-7b-hf}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
KGW_HASH_KEY=${KGW_HASH_KEY:-15485863}
PUSH_TO_HUB=${PUSH_TO_HUB:-false}
HUB_PRIVATE_REPO=${HUB_PRIVATE_REPO:-false}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
TORCH_COMPILE=${TORCH_COMPILE:-False}
HF_HUB_USER=${HF_HUB_USER:-mbakshi1094}

# RunPod secrets are injected as RUNPOD_SECRET_<NAME>.
HF_TOKEN="${HF_TOKEN:-${RUNPOD_SECRET_HF_TOKEN:-}}"

export KGW_HASH_KEY ATTN_IMPLEMENTATION TORCH_COMPILE NPROC_PER_NODE

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"

if [[ -n "${HF_TOKEN}" ]]; then
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
  huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null || true
else
  echo "Warning: no HF_TOKEN or RUNPOD_SECRET_HF_TOKEN set; gated model download / Hub push may fail."
fi

# Build optional Hub-push / step overrides as EXTRA_ARGS for the launch script.
extra=()
if [[ -n "${MAX_STEPS:-}" ]]; then
  extra+=(--max_steps "${MAX_STEPS}")
fi
if [[ "${PUSH_TO_HUB,,}" == "true" || "${PUSH_TO_HUB}" == "1" ]]; then
  if [[ -z "${HUB_MODEL_ID:-}" ]]; then
    if [[ "${WATERMARK_TYPE}" == kgw* ]]; then
      HUB_MODEL_ID="${HF_HUB_USER}/llama-2-7b-logit-watermark-distill-${WATERMARK_TYPE}-hk${KGW_HASH_KEY}"
    else
      HUB_MODEL_ID="${HF_HUB_USER}/llama-2-7b-logit-watermark-distill-${WATERMARK_TYPE}"
    fi
  fi
  extra+=(--push_to_hub True --hub_model_id "${HUB_MODEL_ID}" --token "${HF_TOKEN}")
  if [[ "${HUB_PRIVATE_REPO,,}" == "true" || "${HUB_PRIVATE_REPO}" == "1" ]]; then
    extra+=(--hub_private_repo True)
  else
    extra+=(--hub_private_repo False)
  fi
  echo "Hub push enabled: ${HUB_MODEL_ID}"
fi
# shellcheck disable=SC2206
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # Intentionally unquoted to allow multiple flags.
  extra+=(${EXTRA_ARGS})
fi

export TRAIN_EXTRA_ARGS="${extra[*]:-}"

echo "Starting logit distill: watermark=${WATERMARK_TYPE} hash_key=${KGW_HASH_KEY} nproc=${NPROC_PER_NODE} push_to_hub=${PUSH_TO_HUB}"
exec bash scripts/train/train_llama_logit_distill.sh \
  "${WATERMARK_TYPE}" \
  "${OUTPUT_DIR}" \
  "${MASTER_PORT}" \
  "${MODEL_NAME_OR_PATH}"
