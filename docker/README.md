# Docker / RunPod logit distillation

Build once, then launch multi-GPU logit distill runs with env vars (hash key, Hub push, etc.).

Hub uploads default under [`mbakshi1094`](https://huggingface.co/mbakshi1094).

## Build

From the repo root:

```bash
docker build -t watermark-logit-distill:latest .
```

Push to a registry if you use RunPod custom images:

```bash
docker tag watermark-logit-distill:latest <registry>/watermark-logit-distill:latest
docker push <registry>/watermark-logit-distill:latest
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
