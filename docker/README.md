# Docker / RunPod logit distillation

Build once, then launch multi-GPU logit distill runs with env vars (hash key, Hub push, etc.).

Hub uploads default under [`mbakshi1094`](https://huggingface.co/mbakshi1094).

There are **two images**:

| Tag | Stack | When to use |
|-----|-------|-------------|
| `watermark-logit-distill:latest` | Stock HF ≥4.51, torch 2.5, SDPA | Default / modern path |
| `watermark-logit-distill:legacy` | [Paper fork](https://github.com/chenchenygu/transformers-watermark-learnability) (~4.29.2 + Llama-2), torch 2.0.1, eager attn | Reproduce cygu-quality distilled models / A/B the stack |

Legacy uses the same env vars (`WATERMARK_TYPE`, `KGW_HASH_KEY`, …) but writes checkpoints / Hub repos with a `-legacy` suffix so they do not overwrite modern runs.

## Build

From the repo root:

```bash
docker build -t watermark-logit-distill:latest .
docker build -f Dockerfile.legacy -t watermark-logit-distill:legacy .
```

Push to a registry if you use RunPod custom images:

```bash
docker tag watermark-logit-distill:latest <registry>/watermark-logit-distill:latest
docker tag watermark-logit-distill:legacy <registry>/watermark-logit-distill:legacy
docker push <registry>/watermark-logit-distill:latest
docker push <registry>/watermark-logit-distill:legacy
```

## Run locally (4 GPUs)

```bash
docker run --gpus all --shm-size=64g --rm \
  -e HF_TOKEN \
  -e WATERMARK_TYPE=kgw-k1-gamma0.25-delta2 \
  -e KGW_HASH_KEY=15485863 \
  -e NPROC_PER_NODE=4 \
  -e OUTPUT_DIR=/workspace/out/ \
  -v "$PWD/out:/workspace/out" \
  watermark-logit-distill:latest
```

## Push to Hugging Face after training

With `PUSH_TO_HUB=true`, training finishes by calling the **`hf upload` CLI** (not Trainer’s native Hub/Xet push), with `HF_HUB_DISABLE_XET=1` by default. Checkpoints / `.git` are excluded. Legacy RunPod bootstrap installs the standalone `hf` CLI so it does not conflict with the pinned paper-era `huggingface-hub`.

```bash
docker run --gpus all --shm-size=64g --rm \
  -e HF_TOKEN \
  -e WATERMARK_TYPE=kgw-k1-gamma0.25-delta2 \
  -e KGW_HASH_KEY=9999991 \
  -e PUSH_TO_HUB=true \
  -e NPROC_PER_NODE=4 \
  watermark-logit-distill:latest
```

If `HUB_MODEL_ID` is unset, the entrypoint builds:
`mbakshi1094/llama-2-7b-logit-watermark-distill-${WATERMARK_TYPE}-hk${KGW_HASH_KEY}`.

## Persistent storage (network volume)

HF model cache and checkpoints should live on a **network volume** so pods can be deleted/recreated without re-downloading Llama.

1. Create a volume in a DC that supports *both* network volumes and your GPU type:

```bash
runpodctl nv create --name watermark-hf-cache --size 200 --data-center-id US-CA-2
```

Existing volume from this setup: **`q15kkgopbh`** (`watermark-hf-cache`, 200GB, `US-CA-2`).

2. Launch the pod **in that same DC** and attach the volume (not a disposable `volumeInGb` disk):

```bash
# via REST / console: networkVolumeId=q15kkgopbh, volumeMountPath=/workspace
# HF_HOME=/workspace/.cache/huggingface
```

Network volumes are **DC-scoped**. If that DC has no A100 capacity, either wait/retry there or create another volume in a DC that currently has GPUs.

## Auth

Locally, put `RUNPOD_API_KEY` in a gitignored `.env` (see `.env.example`), then:

```bash
set -a && source .env && set +a
runpodctl config --apiKey "$RUNPOD_API_KEY"
```

On the pod, use the RunPod secret `HF_TOKEN` (available as `RUNPOD_SECRET_HF_TOKEN`) for Hugging Face. Do not put Hub tokens in the image.

| Env | Example | Purpose |
|-----|---------|---------|
| `WATERMARK_TYPE` | `kgw-k1-gamma0.25-delta2` | Watermark recipe |
| `KGW_HASH_KEY` | `15485863` | KGW PRF salt |
| `RUNPOD_SECRET_HF_TOKEN` | (from secret) | Gated Llama + Hub upload |
| `PUSH_TO_HUB` | `true` | Upload after train |
| `HUB_MODEL_ID` | optional | Override auto Hub repo id |
| `HF_HUB_USER` | `mbakshi1094` | Hub namespace for auto ids |
| `NPROC_PER_NODE` | `4` | GPU count |
| `OUTPUT_DIR` | `/workspace/out/` | Checkpoint root |
| `MAX_STEPS` | `5000` | Optional step override |

Example with `runpodctl` (image must already be pushed; wire the secret in the RunPod console or your preferred secret injection method):

```bash
runpodctl pod create \
  --name "logit-kgw-hk15485863" \
  --gpu-id "NVIDIA A100 80GB PCIe" \
  --gpu-count 4 \
  --image "<registry>/watermark-logit-distill:latest" \
  --env "WATERMARK_TYPE=kgw-k1-gamma0.25-delta2,KGW_HASH_KEY=15485863,PUSH_TO_HUB=true,NPROC_PER_NODE=4" \
  --container-disk-in-gb 50 \
  --volume-in-gb 200
```

With `PUSH_TO_HUB=true` and hash key `15485863`, the model lands at:
`https://huggingface.co/mbakshi1094/llama-2-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2-hk15485863`

Sweep hash keys by launching one pod (or one `docker run`) per `KGW_HASH_KEY`.

## Legacy (paper-era) stack

To train with the original custom transformers fork + torch 2.0.1 (for A/B vs the modern stack):

```bash
docker build -f Dockerfile.legacy -t watermark-logit-distill:legacy .

docker run --gpus all --shm-size=64g --rm \
  -e HF_TOKEN \
  -e WATERMARK_TYPE=kgw-k1-gamma0.25-delta2 \
  -e KGW_HASH_KEY=12997009 \
  -e PUSH_TO_HUB=true \
  -e NPROC_PER_NODE=4 \
  -e OUTPUT_DIR=/workspace/out/ \
  -v "$PWD/out:/workspace/out" \
  watermark-logit-distill:legacy
```

Mistral-7B-v0.3 on the legacy path (remapped to Llama inside the train script):

```bash
docker run --gpus all --shm-size=64g --rm \
  -e HF_TOKEN \
  -e WATERMARK_TYPE=kgw-k1-gamma0.25-delta2 \
  -e KGW_HASH_KEY=12997009 \
  -e MODEL_NAME_OR_PATH=mistralai/Mistral-7B-v0.3 \
  -e PUSH_TO_HUB=true \
  -e NPROC_PER_NODE=4 \
  -e OUTPUT_DIR=/workspace/out/ \
  -v "$PWD/out:/workspace/out" \
  watermark-logit-distill:legacy
```

Defaults:

- Launcher: `scripts/train/train_llama_logit_distill_legacy.sh`
- Train script: `train_logit_distill_legacy.py` (pre-modernize FSDP / no SDPA)
- Checkpoint dir / Hub id: `${MODEL_SLUG}-…-hk${KGW_HASH_KEY}-legacy`
  (`MODEL_SLUG` is `llama-2-7b` or `mistral-7b` from `MODEL_NAME_OR_PATH`)

Local (non-Docker) legacy install:

```bash
# use a CUDA torch 2.0.1 env matching the paper, then:
pip install -r requirements-legacy.txt
bash scripts/train/train_llama_logit_distill_legacy.sh \
  kgw-k1-gamma0.25-delta2 out/ 29500
```
