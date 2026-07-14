#!/usr/bin/env bash
# Env-driven entrypoint for paper-era (legacy) logit watermark distillation.
#
# Same env surface as docker/entrypoint_logit_distill.sh, but launches the
# custom-transformers / torch-2.0 training path (train_logit_distill_legacy.py).
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
#   HF_TOKEN or RUNPOD_SECRET_HF_TOKEN
#   WANDB_API_KEY or RUNPOD_SECRET_WAND_API_KEY / RUNPOD_SECRET_WANDB_API_KEY
#   HF_HUB_USER      default mbakshi1094
#   PUSH_TO_HUB      true/false (default false)
#   HUB_MODEL_ID     optional; default
#     ${HF_HUB_USER}/llama-2-7b-logit-watermark-distill-${WATERMARK_TYPE}-hk${KGW_HASH_KEY}-legacy
#   HUB_PRIVATE_REPO true/false (default false)
#   HUB_STRATEGY     end (default; every_save can flake on Hub)
#   RESUME_FROM_CHECKPOINT  path or true
#   MAX_STEPS        override training steps
#   CLEANUP_HASH_KEYS  prior hash keys whose checkpoint-* dirs to delete
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
HUB_STRATEGY=${HUB_STRATEGY:-end}
HF_HUB_USER=${HF_HUB_USER:-mbakshi1094}
# Also wipe other prior hash-key modern runs by default (volume space).
CLEANUP_HASH_KEYS=${CLEANUP_HASH_KEYS:-}
# When true (default), delete modern-stack checkpoint-* / leftover weights for THIS
# hash key too so a legacy retrain does not compete with ~40GB+ old FSDP ckpts.
CLEANUP_MODERN_SAME_KEY=${CLEANUP_MODERN_SAME_KEY:-true}

HF_TOKEN="${HF_TOKEN:-${RUNPOD_SECRET_HF_TOKEN:-}}"
WANDB_API_KEY="${WANDB_API_KEY:-${RUNPOD_SECRET_WAND_API_KEY:-${RUNPOD_SECRET_WANDB_API_KEY:-}}}"

export KGW_HASH_KEY NPROC_PER_NODE
if [[ -n "${WANDB_API_KEY}" ]]; then
  export WANDB_API_KEY
  unset WANDB_DISABLED WANDB_MODE || true
else
  export WANDB_DISABLED=true
  export WANDB_MODE=disabled
fi

mkdir -p "${OUTPUT_DIR}" "${HF_HOME:-/workspace/.cache/huggingface}"

echo "Workspace disk before cleanup:"
df -h "${OUTPUT_DIR}" /workspace 2>/dev/null || df -h /workspace || true

cleanup_run_dir() {
  local run_dir=$1
  if [[ ! -d "${run_dir}" ]]; then
    return 0
  fi
  echo "Cleaning prior-run artifacts under ${run_dir}"
  shopt -s nullglob
  for ckpt in "${run_dir}"/checkpoint-*; do
    echo "  rm -rf ${ckpt}"
    rm -rf "${ckpt}"
  done
  shopt -u nullglob
  # Drop leftover Hub/local full model files from the old run if present.
  rm -f "${run_dir}"/model.safetensors* "${run_dir}"/pytorch_model* \
    "${run_dir}"/optimizer.pt "${run_dir}"/scheduler.pt \
    "${run_dir}"/trainer_state.json "${run_dir}"/rng_state*.pth 2>/dev/null || true
  du -sh "${run_dir}" 2>/dev/null || true
}

if [[ "${WATERMARK_TYPE}" == kgw* ]]; then
  # Always free the modern-stack dir for this hash key before a legacy train.
  if [[ "${CLEANUP_MODERN_SAME_KEY,,}" == "true" || "${CLEANUP_MODERN_SAME_KEY}" == "1" ]]; then
    cleanup_run_dir "${OUTPUT_DIR%/}/llama-2-7b-logit-watermark-distill-${WATERMARK_TYPE}-hk${KGW_HASH_KEY}"
  fi
  if [[ -n "${CLEANUP_HASH_KEYS}" ]]; then
    for old_hk in ${CLEANUP_HASH_KEYS}; do
      if [[ "${old_hk}" == "${KGW_HASH_KEY}" ]]; then
        continue
      fi
      for suffix in "" "-legacy"; do
        cleanup_run_dir "${OUTPUT_DIR%/}/llama-2-7b-logit-watermark-distill-${WATERMARK_TYPE}-hk${old_hk}${suffix}"
      done
    done
  fi
fi

echo "Workspace disk after cleanup:"
df -h /workspace || true

if [[ -n "${HF_TOKEN}" ]]; then
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
  # huggingface-hub 0.16 uses huggingface-cli
  huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null || true
else
  echo "Warning: no HF_TOKEN or RUNPOD_SECRET_HF_TOKEN set; gated model download / Hub push may fail."
fi

if [[ "${WATERMARK_TYPE}" == kgw* ]]; then
  MODEL_OUT_DIR="${OUTPUT_DIR%/}/llama-2-7b-logit-watermark-distill-${WATERMARK_TYPE}-hk${KGW_HASH_KEY}-legacy"
else
  MODEL_OUT_DIR="${OUTPUT_DIR%/}/llama-2-7b-logit-watermark-distill-${WATERMARK_TYPE}-legacy"
fi
if [[ -d "${MODEL_OUT_DIR}" ]]; then
  shopt -s nullglob
  for ckpt in "${MODEL_OUT_DIR}"/checkpoint-*; do
    if [[ -d "${ckpt}" && ! -f "${ckpt}/trainer_state.json" ]]; then
      echo "Removing incomplete checkpoint (no trainer_state.json): ${ckpt}"
      rm -rf "${ckpt}"
    fi
  done
  shopt -u nullglob
fi

extra=()
if [[ -n "${MAX_STEPS:-}" ]]; then
  extra+=(--max_steps "${MAX_STEPS}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  extra+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
  echo "Resume from checkpoint: ${RESUME_FROM_CHECKPOINT}"
fi
if [[ "${PUSH_TO_HUB,,}" == "true" || "${PUSH_TO_HUB}" == "1" ]]; then
  if [[ -z "${HUB_MODEL_ID:-}" ]]; then
    if [[ "${WATERMARK_TYPE}" == kgw* ]]; then
      HUB_MODEL_ID="${HF_HUB_USER}/llama-2-7b-logit-watermark-distill-${WATERMARK_TYPE}-hk${KGW_HASH_KEY}-legacy"
    else
      HUB_MODEL_ID="${HF_HUB_USER}/llama-2-7b-logit-watermark-distill-${WATERMARK_TYPE}-legacy"
    fi
  fi
  # Older transformers: --push_to_hub and related flags still work.
  extra+=(--push_to_hub True --hub_model_id "${HUB_MODEL_ID}" --hub_strategy "${HUB_STRATEGY}")
  if [[ "${HUB_PRIVATE_REPO,,}" == "true" || "${HUB_PRIVATE_REPO}" == "1" ]]; then
    extra+=(--hub_private_repo True)
  else
    extra+=(--hub_private_repo False)
  fi
  echo "Hub push enabled: ${HUB_MODEL_ID} (strategy=${HUB_STRATEGY})"
fi
# shellcheck disable=SC2206
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  extra+=(${EXTRA_ARGS})
fi

export TRAIN_EXTRA_ARGS="${extra[*]:-}"

echo "Starting LEGACY logit distill: watermark=${WATERMARK_TYPE} hash_key=${KGW_HASH_KEY} nproc=${NPROC_PER_NODE} push_to_hub=${PUSH_TO_HUB} out=${MODEL_OUT_DIR}"
echo "Stack: transformers-watermark-learnability fork + torch 2.0.1 / accelerate 0.21"
exec bash scripts/train/train_llama_logit_distill_legacy.sh \
  "${WATERMARK_TYPE}" \
  "${OUTPUT_DIR}" \
  "${MASTER_PORT}" \
  "${MODEL_NAME_OR_PATH}"
