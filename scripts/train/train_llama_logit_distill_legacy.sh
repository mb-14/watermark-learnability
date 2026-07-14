#!/bin/bash
# Paper-era logit distill launcher (custom transformers fork, old FSDP wrap).
#
# Usage:
#   bash scripts/train/train_llama_logit_distill_legacy.sh \
#     <watermark_type> <output_dir/> <master_port> [llama_path]
#
# Env overrides:
#   NPROC_PER_NODE    default 4
#   KGW_HASH_KEY      default 15485863 (KGW runs only)
#   TRAIN_EXTRA_ARGS  extra flags appended to train_logit_distill_legacy.py
set -euo pipefail

watermark=$1
out_dir=$2
port=$3
llama=${4:-"meta-llama/Llama-2-7b-hf"}

nproc=${NPROC_PER_NODE:-4}
kgw_hash_key=${KGW_HASH_KEY:-15485863}

model_name="llama-2-7b-logit-watermark-distill-${watermark}"

if [ "$watermark" = "aar-k2" ]; then
    watermark_args=(--watermark_type aar --aar_watermark_k 2)
elif [ "$watermark" = "aar-k3" ]; then
    watermark_args=(--watermark_type aar --aar_watermark_k 3)
elif [ "$watermark" = "aar-k4" ]; then
    watermark_args=(--watermark_type aar --aar_watermark_k 4)
elif [ "$watermark" = "kgw-k0-gamma0.25-delta1" ]; then
    watermark_args=(--watermark_type kgw
      --kgw_watermark_gamma 0.25
      --kgw_watermark_delta 1.0
      --kgw_watermark_seeding_scheme simple_0
      --kgw_watermark_hash_key "${kgw_hash_key}")
    model_name="${model_name}-hk${kgw_hash_key}-legacy"
elif [ "$watermark" = "kgw-k0-gamma0.25-delta2" ]; then
    watermark_args=(--watermark_type kgw
      --kgw_watermark_gamma 0.25
      --kgw_watermark_delta 2.0
      --kgw_watermark_seeding_scheme simple_0
      --kgw_watermark_hash_key "${kgw_hash_key}")
    model_name="${model_name}-hk${kgw_hash_key}-legacy"
elif [ "$watermark" = "kgw-k1-gamma0.25-delta1" ]; then
    watermark_args=(--watermark_type kgw
      --kgw_watermark_gamma 0.25
      --kgw_watermark_delta 1.0
      --kgw_watermark_seeding_scheme simple_1
      --kgw_watermark_hash_key "${kgw_hash_key}")
    model_name="${model_name}-hk${kgw_hash_key}-legacy"
elif [ "$watermark" = "kgw-k1-gamma0.25-delta2" ]; then
    watermark_args=(--watermark_type kgw
      --kgw_watermark_gamma 0.25
      --kgw_watermark_delta 2.0
      --kgw_watermark_seeding_scheme simple_1
      --kgw_watermark_hash_key "${kgw_hash_key}")
    model_name="${model_name}-hk${kgw_hash_key}-legacy"
elif [ "$watermark" = "kgw-k2-gamma0.25-delta2" ]; then
    watermark_args=(--watermark_type kgw
      --kgw_watermark_gamma 0.25
      --kgw_watermark_delta 2.0
      --kgw_watermark_seeding_scheme simple_2
      --kgw_watermark_hash_key "${kgw_hash_key}")
    model_name="${model_name}-hk${kgw_hash_key}-legacy"
elif [ "$watermark" = "kth-shift1" ]; then
    watermark_args=(--watermark_type kth --kth_watermark_key_len 256 --kth_watermark_num_shifts 1)
elif [ "$watermark" = "kth-shift2" ]; then
    watermark_args=(--watermark_type kth --kth_watermark_key_len 256 --kth_watermark_num_shifts 2)
elif [ "$watermark" = "kth-shift4" ]; then
    watermark_args=(--watermark_type kth --kth_watermark_key_len 256 --kth_watermark_num_shifts 4)
elif [ "$watermark" = "kth-shift256" ]; then
    watermark_args=(--watermark_type kth --kth_watermark_key_len 256 --kth_watermark_num_shifts 256)
else
    echo "Unsupported watermark type ${watermark}."
    exit 1
fi

if [[ "$watermark" == kth* ]]; then
    batch_size=32
    block_size=256
else
    batch_size=16
    block_size=512
fi

# shellcheck disable=SC2206
extra_args=()
if [[ -n "${TRAIN_EXTRA_ARGS:-}" ]]; then
  extra_args=(${TRAIN_EXTRA_ARGS})
fi

# Paper recipe: bf16 + FSDP full_shard auto_wrap LlamaDecoderLayer; no SDPA / torch_dtype /
# dataloader_num_workers knobs from the modern launcher.
torchrun --nproc_per_node="${nproc}" --master_port="${port}" train_logit_distill_legacy.py \
    --model_name_or_path "${llama}" \
    --dataset_name Skylion007/openwebtext \
    --streaming \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps 1 \
    --do_train \
    --max_steps 5000 \
    --logging_steps 1 \
    --output_dir "${out_dir}${model_name}" \
    --learning_rate 1e-5 \
    --lr_scheduler_type "cosine" \
    --warmup_steps 500 \
    --block_size ${block_size} \
    --save_steps 1000 \
    --save_total_limit 1 \
    --tf32 True \
    --bf16 True \
    --gradient_checkpointing True \
    "${watermark_args[@]}" \
    --watermark_seed 42 \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap "LlamaDecoderLayer" \
    "${extra_args[@]}"
