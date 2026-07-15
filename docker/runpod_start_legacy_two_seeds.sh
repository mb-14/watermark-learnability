#!/usr/bin/env bash
# Bootstrap legacy Mistral distill once, then train two KGW hash keys sequentially.
# Used as RunPod dockerStartCmd so a single network volume can host both seeds.
set -euo pipefail

export PATH="${HOME}/.local/bin:${PATH}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export REPO_URL="${REPO_URL:-https://github.com/mb-14/watermark-learnability.git}"
export REPO_BRANCH="${REPO_BRANCH:-modernize-logit-distill}"
export WORK=/workspace/watermark-learnability
export LOG_DIR="${LOG_DIR:-/workspace/logs}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/workspace/.cache/torch}"
export OUTPUT_DIR="${OUTPUT_DIR:-/workspace/out/}"
SEED_KEYS=(${SEED_KEYS:-12997009 22983996})

mkdir -p "${HF_HOME}" "${OUTPUT_DIR}" "${LOG_DIR}"

if [[ ! -d "${WORK}/.git" ]]; then
  git clone --branch "${REPO_BRANCH}" --depth 1 "${REPO_URL}" "${WORK}"
else
  if git -C "${WORK}" fetch --depth 1 origin "${REPO_BRANCH}"; then
    git -C "${WORK}" checkout "${REPO_BRANCH}" || true
    git -C "${WORK}" reset --hard "origin/${REPO_BRANCH}" || true
  else
    echo "Warning: git fetch failed; continuing with existing checkout."
  fi
fi
cd "${WORK}"

if ! command -v hf >/dev/null 2>&1; then
  echo "Installing standalone Hugging Face CLI (hf)..."
  curl -LsSf https://hf.co/cli/install.sh | bash -s
  export PATH="${HOME}/.local/bin:${PATH}"
fi
hf version || true
if ! command -v git-lfs >/dev/null 2>&1; then
  apt-get update -y && DEBIAN_FRONTEND=noninteractive apt-get install -y git-lfs
  git lfs install
fi

pip install --upgrade pip
python - <<'PY'
import importlib.util
import subprocess
import sys

spec = importlib.util.find_spec("torch")
need_restore = False
if spec is None:
    need_restore = True
else:
    import torch

    ver = torch.__version__
    major_minor = tuple(int(x) for x in ver.split("+")[0].split(".")[:2])
    cuda = getattr(torch.version, "cuda", None) or ""
    if major_minor >= (2, 5) or cuda.startswith("13"):
        need_restore = True
if need_restore:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "torch==2.4.0",
            "torchaudio==2.4.0",
            "--index-url",
            "https://download.pytorch.org/whl/cu124",
        ]
    )
PY

export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
python - <<'PY'
from pathlib import Path
import importlib.metadata as md

lines = []
for pkg in ("torch", "torchaudio", "torchvision"):
    try:
        lines.append(f"{pkg}=={md.version(pkg)}")
    except md.PackageNotFoundError:
        pass
Path("/tmp/legacy-torch-constraints.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
print("constraints:", Path("/tmp/legacy-torch-constraints.txt").read_text().strip() or "<empty>")
PY
pip install -r requirements-legacy.txt \
  --constraint /tmp/legacy-torch-constraints.txt \
  --upgrade-strategy only-if-needed
pip install 'fsspec==2023.9.2' --constraint /tmp/legacy-torch-constraints.txt

overall=0
for KEY in "${SEED_KEYS[@]}"; do
  export KGW_HASH_KEY="${KEY}"
  others=()
  for k in "${SEED_KEYS[@]}" 15485863; do
    if [[ "${k}" != "${KEY}" ]]; then
      others+=("${k}")
    fi
  done
  export CLEANUP_HASH_KEYS="${others[*]}"
  LOG_FILE="${LOG_DIR}/logit_distill_legacy_${KEY}_$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "======== Starting Mistral legacy hash_key=${KEY} cleanup=${CLEANUP_HASH_KEYS} ========" | tee -a "${LOG_FILE}"
  set +e
  bash docker/entrypoint_logit_distill_legacy.sh 2>&1 | tee -a "${LOG_FILE}"
  status=${PIPESTATUS[0]}
  set -e
  echo "Hash key ${KEY} exited with code ${status}" | tee -a "${LOG_FILE}"
  if [[ "${status}" -ne 0 ]]; then
    overall="${status}"
    echo "Stopping seed loop after failure on ${KEY}" | tee -a "${LOG_FILE}"
    break
  fi
  OUT_DIR="${OUTPUT_DIR%/}/mistral-7b-logit-watermark-distill-kgw-k1-gamma0.25-delta2-hk${KEY}-legacy"
  if [[ -d "${OUT_DIR}" ]]; then
    rm -rf "${OUT_DIR}"/checkpoint-* || true
  fi
done

echo "All seed runs finished (overall=${overall}). Sleeping to keep pod SSH-able."
exec sleep infinity
