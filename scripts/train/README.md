## Watermark distillation training scripts

This subdirectory contains scripts for training using logit-based and sampling-based watermark distillation.

### Logit-based watermark distillation

[`train_llama_logit_distill.sh`](train_llama_logit_distill.sh) runs logit-based watermark distillation on Llama 2 7B. The training configuration is for 4 NVIDIA A100 80GB GPUs (FSDP full shard, bf16, SDPA attention). The script is run from the top-level directory as
```
bash scripts/train/train_llama_logit_distill.sh <watermark_type> <output_dir/> <master_port> <llama_path>
```
- `watermark_type` specifies the watermarking strategy for training. The possible types are listed at the [end](#watermark-types) of this README.
- `output_dir` specifies the directory where the model should be stored (with the trailing `/`). This should not include the model name itself, which is automatically computed by the script.
- `master_port` is the port that is passed to `torchrun`. This can be more or less arbitrarily selected.
- `llama_path` (optional) specifies the path where the base Llama 2 7B model weights are loaded from. Defaults to [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf), which downloads from Hugging Face.

Optional environment overrides:
- `ATTN_IMPLEMENTATION=flash_attention_2` — use FlashAttention 2 if [`flash-attn`](https://github.com/Dao-AILab/flash-attention) is installed (default: `sdpa`).
- `TORCH_COMPILE=True` — enable `torch.compile` via Hugging Face `TrainingArguments`.
- `KGW_HASH_KEY=15485863` — KGW PRF salt (default `15485863`; included in the output dir name for KGW runs).
- `NPROC_PER_NODE=4` — number of GPUs for `torchrun`.
- `TRAIN_EXTRA_ARGS="--push_to_hub True --hub_model_id user/repo"` — extra CLI flags. Hub push uses the `hf upload` CLI (not Trainer native/Xet push).

Modern flags already enabled in the script: `--torch_dtype bfloat16`, `--attn_implementation`, `--dataloader_num_workers 4`, `--dataloader_pin_memory True`, and `--fsdp_config` with `LlamaDecoderLayer` wrapping.

For containerized / RunPod automation (hash-key sweeps + Hub upload), see [`docker/README.md`](/docker/README.md).

**Paper-era / legacy stack** (custom transformers fork, torch 2.0.1, no SDPA): use
[`train_llama_logit_distill_legacy.sh`](train_llama_logit_distill_legacy.sh) with
`requirements-legacy.txt`, or build `Dockerfile.legacy` (`watermark-logit-distill:legacy`).
Output dirs and Hub ids get a `-legacy` suffix. Same `KGW_HASH_KEY` env override applies.

The paper fork has no native Mistral classes. For [`mistralai/Mistral-7B-v0.3`](https://huggingface.co/mistralai/Mistral-7B-v0.3), `train_logit_distill_legacy.py` remaps the config/weights to Llama (keeps GQA + `rope_theta=1e6`) and still FSDP-wraps `LlamaDecoderLayer`. Example:

```bash
MODEL_NAME_OR_PATH=mistralai/Mistral-7B-v0.3 \
  bash scripts/train/train_llama_logit_distill_legacy.sh \
  kgw-k1-gamma0.25-delta2 /workspace/out/ 29500 mistralai/Mistral-7B-v0.3
```

Output / Hub slug becomes `mistral-7b-logit-watermark-distill-…-legacy` (override with `MODEL_NAME_PREFIX`).

### Sampling-based watermark distillation

To perform sampling-based watermark distillation, you can either use the training data we have uploaded to Hugging Face (listed in the top-level [README.md](/README.md#training-data-for-sampling-based-watermark-distillation)) or generate the training data yourself. `generate_sampling_distill_train_data.sh` generates watermarked samples from the teacher Llama 2 7B to use as training data. We used 1 NVIDIA A100 80GB GPU. The script is run from the top-level directory as
```
bash scripts/train/generate_sampling_distill_train_data.sh <watermark_type> <llama_path>
```
- `watermark_type` specifies the watermarking strategy for training. The possible types are listed at the [end](#watermark-types) of this README.
- `llama_path` (optional) specifies the path where the base Llama 2 7B model weights are loaded from. Defaults to [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf), which downloads from Hugging Face.

Then, to run sampling-based watermark distillation on Llama 2 7B as the student (on 4 A100 NVIDIA 80GB GPUs), the script is run as
```
bash scripts/train/train_llama_sampling_distill.sh <watermark_type> <output_dir/> <master_port> <llama_path> <dataset_location>
```
- `watermark_type` specifies the watermarking strategy for training. The possible types are listed at the [end](#watermark-types) of this README.
- `output_dir` specifies the directory where the model should be stored (with the trailing `/`). This should not include the model name itself, which is automatically computed by the script.
- `master_port` is the port that is passed to `torchrun`. This can be more or less arbitrarily selected.
- `llama_path` (optional) specifies the path where the base Llama 2 7B model weights are loaded from. Defaults to [`meta-llama/Llama-2-7b-hf`](https://huggingface.co/meta-llama/Llama-2-7b-hf), which downloads from Hugging Face.
- `dataset_location` (optional) should be set to `hf` to download the training data from Hugging Face, or `local` if you generated the training data yourself in the previous step. Defaults to `hf`.

To run sampling-based watermark distillation on Pythia 1.4B as the student (on 1 A100 NVIDIA 80GB GPU), the script is similarly run as
```
bash scripts/train/train_pythia_sampling_distill.sh <watermark_type> <output_dir/> <master_port> <pythia_path> <dataset_location>
```
- `pythia_path` (optional) specifies the path where the base Pythia 1.4B model weights are loaded from. Defaults to [`EleutherAI/pythia-1.4b`](https://huggingface.co/EleutherAI/pythia-1.4b), which downloads from Hugging Face.

### Watermark types

These are the strings that can be passed into the training scripts to specify the watermark type. The watermark configuration files are in [`experiments/watermark-configs`](/experiments/watermark-configs).

KGW 
- `kgw-k0-gamma0.25-delta1`
- `kgw-k0-gamma0.25-delta2`
- `kgw-k1-gamma0.25-delta1`
- `kgw-k1-gamma0.25-delta2`
- `kgw-k2-gamma0.25-delta2`

Aar
- `aar-k2`
- `aar-k3`
- `aar-k4`

KTH
- `kth-shift1`
- `kth-shift2`
- `kth-shift4`
- `kth-shift256`
