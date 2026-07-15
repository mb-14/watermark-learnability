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
#   WANDB_API_KEY or RUNPOD_SECRET_WAND_API_KEY / RUNPOD_SECRET_WANDB_API_KEY
#   HF_HUB_USER      default mbakshi1094 (used to build HUB_MODEL_ID if unset)
#   PUSH_TO_HUB      true/false (default false)
#                    When true, training ends with `hf upload` (not Trainer native push).
#   HUB_MODEL_ID     optional; default ${HF_HUB_USER}/llama-2-7b-logit-watermark-distill-${WATERMARK_TYPE}-hk${KGW_HASH_KEY}
#   HUB_PRIVATE_REPO true/false (default false)
#   HUB_STRATEGY     end|every_save|checkpoint|... (default end; every_save previously crashed mid-push)
#   HF_HUB_DISABLE_XET  default 1 (safer large-file uploads via hf CLI)
#   RESUME_FROM_CHECKPOINT  path or true (optional; otherwise auto-detect last good ckpt)
#   ATTN_IMPLEMENTATION  sdpa | flash_attention_2 (default sdpa)
#   TORCH_COMPILE    True/False (default False)
#   MAX_STEPS        override training steps
#   CLEANUP_HASH_KEYS  space-separated prior hash keys whose checkpoint-* dirs are
#                      deleted before training to free network-volume space (default: 12997009)
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
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
TORCH_COMPILE=${TORCH_COMPILE:-False}
HF_HUB_USER=${HF_HUB_USER:-mbakshi1094}
CLEANUP_HASH_KEYS=${CLEANUP_HASH_KEYS:-12997009}

# RunPod secrets are injected as RUNPOD_SECRET_<NAME>.
HF_TOKEN="${HF_TOKEN:-${RUNPOD_SECRET_HF_TOKEN:-}}"
# Secret name on this account is WAND (not WANDB); accept both spellings.
WANDB_API_KEY="${WANDB_API_KEY:-${RUNPOD_SECRET_WAND_API_KEY:-${RUNPOD_SECRET_WANDB_API_KEY:-}}}"

export KGW_HASH_KEY ATTN_IMPLEMENTATION TORCH_COMPILE NPROC_PER_NODE
export PATH="${HOME}/.local/bin:${PATH}"
export HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET:-1}
if [[ -n "${WANDB_API_KEY}" ]]; then
  export WANDB_API_KEY
  unset WANDB_DISABLED WANDB_MODE || true
else
  # Transformers auto-enables wandb when installed; skip login prompts headless.
  export WANDB_DISABLED=true
  export WANDB_MODE=disabled
fi

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"

echo "Workspace disk before cleanup:"
df -h "${OUTPUT_DIR}" /workspace 2>/dev/null || df -h /workspace || true

# Free space from prior hash-key runs (full FSDP checkpoints are ~40GB+ each).
if [[ "${WATERMARK_TYPE}" == kgw* && -n "${CLEANUP_HASH_KEYS}" ]]; then
  for old_hk in ${CLEANUP_HASH_KEYS}; do
    if [[ "${old_hk}" == "${KGW_HASH_KEY}" ]]; then
      continue
    fi
    old_out="${OUTPUT_DIR%/}/llama-2-7b-logit-watermark-distill-${WATERMARK_TYPE}-hk${old_hk}"
    if [[ -d "${old_out}" ]]; then
      echo "Cleaning prior-run artifacts under ${old_out}"
      shopt -s nullglob
      for ckpt in "${old_out}"/checkpoint-*; do
        echo "  rm -rf ${ckpt}"
        rm -rf "${ckpt}"
      done
      shopt -u nullglob
      # Drop leftover Hub/local full model files from the old run if present.
      rm -f "${old_out}"/model.safetensors* "${old_out}"/pytorch_model* \
        "${old_out}"/optimizer.pt "${old_out}"/scheduler.pt \
        "${old_out}"/trainer_state.json "${old_out}"/rng_state*.pth 2>/dev/null || true
      du -sh "${old_out}" 2>/dev/null || true
    fi
  done
fi

echo "Workspace disk after cleanup:"
df -h /workspace || true

if [[ -n "${HF_TOKEN}" ]]; then
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
  huggingface-cli login --token "${HF_TOKEN}" --add-to-git-credential 2>/dev/null || true
else
  echo "Warning: no HF_TOKEN or RUNPOD_SECRET_HF_TOKEN set; gated model download / Hub push may fail."
fi

# Drop incomplete checkpoints (e.g. died mid-save / mid Hub push) so get_last_checkpoint
# resumes from the last trainable one (needs trainer_state.json).
if [[ "${WATERMARK_TYPE}" == kgw* ]]; then
  MODEL_OUT_DIR="${OUTPUT_DIR%/}/llama-2-7b-logit-watermark-distill-${WATERMARK_TYPE}-hk${KGW_HASH_KEY}"
else
  MODEL_OUT_DIR="${OUTPUT_DIR%/}/llama-2-7b-logit-watermark-distill-${WATERMARK_TYPE}"
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

# Build optional Hub-push / step overrides as EXTRA_ARGS for the launch script.
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
      HUB_MODEL_ID="${HF_HUB_USER}/llama-2-7b-logit-watermark-distill-${WATERMARK_TYPE}-hk${KGW_HASH_KEY}"
    else
      HUB_MODEL_ID="${HF_HUB_USER}/llama-2-7b-logit-watermark-distill-${WATERMARK_TYPE}"
    fi
  fi
  # push_to_hub enables end-of-run upload via overridden Trainer.push_to_hub -> hf upload.
  extra+=(--push_to_hub True --hub_model_id "${HUB_MODEL_ID}" --hub_strategy "${HUB_STRATEGY}")
  if [[ "${HUB_PRIVATE_REPO,,}" == "true" || "${HUB_PRIVATE_REPO}" == "1" ]]; then
    extra+=(--hub_private_repo True)
  else
    extra+=(--hub_private_repo False)
  fi
  if ! command -v hf >/dev/null 2>&1; then
    echo "WARNING: hf CLI not found on PATH; Hub upload will fail. Install via https://hf.co/cli"
  fi
  echo "Hub push enabled via hf upload: ${HUB_MODEL_ID} (strategy=${HUB_STRATEGY}, HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET})"
fi
# shellcheck disable=SC2206
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # Intentionally unquoted to allow multiple flags.
  extra+=(${EXTRA_ARGS})
fi

export TRAIN_EXTRA_ARGS="${extra[*]:-}"

echo "Starting logit distill: watermark=${WATERMARK_TYPE} hash_key=${KGW_HASH_KEY} nproc=${NPROC_PER_NODE} push_to_hub=${PUSH_TO_HUB} out=${MODEL_OUT_DIR}"
exec bash scripts/train/train_llama_logit_distill.sh \
  "${WATERMARK_TYPE}" \
  "${OUTPUT_DIR}" \
  "${MASTER_PORT}" \
  "${MODEL_NAME_OR_PATH}"
