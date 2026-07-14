# CUDA PyTorch base for multi-GPU FSDP logit distillation (RunPod / local).
# Build:
#   docker build -t watermark-logit-distill:latest .
# Run (example):
#   docker run --gpus all --shm-size=64g \
#     -e HF_TOKEN -e WATERMARK_TYPE=kgw-k1-gamma0.25-delta2 \
#     -e KGW_HASH_KEY=15485863 -e PUSH_TO_HUB=true \
#     -e HUB_MODEL_ID=user/llama-2-7b-logit-kgw-hk15485863 \
#     -e NPROC_PER_NODE=4 \
#     -v /workspace/out:/workspace/out \
#     watermark-logit-distill:latest

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TORCH_HOME=/workspace/.cache/torch \
    WORKDIR=/workspace/watermark-learnability

WORKDIR ${WORKDIR}

# System deps for Cython / networking
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install "huggingface_hub[cli]>=0.24.0"

COPY . .

RUN chmod +x docker/entrypoint_logit_distill.sh scripts/train/train_llama_logit_distill.sh

ENTRYPOINT ["/workspace/watermark-learnability/docker/entrypoint_logit_distill.sh"]
