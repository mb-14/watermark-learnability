"""Upload model folders via the ``hf upload`` CLI.

Trainer / huggingface_hub native push (esp. Xet) has been flaky on large
shard uploads from RunPod. Prefer the CLI path with Xet disabled.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_EXCLUDES = (
    "checkpoint-*",
    ".git",
    ".git/*",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state*",
)


def _resolve_hf_cli() -> str:
    for name in ("hf", "huggingface-cli"):
        path = shutil.which(name)
        if path:
            return path
    raise FileNotFoundError(
        "Neither `hf` nor `huggingface-cli` found on PATH. "
        "Install with: curl -LsSf https://hf.co/cli/install.sh | bash -s"
    )


def upload_folder_via_hf_cli(
    repo_id: str,
    local_dir: str,
    *,
    private: bool = False,
    commit_message: Optional[str] = None,
    exclude: Sequence[str] = DEFAULT_EXCLUDES,
    disable_xet: bool = True,
) -> str:
    """Upload ``local_dir`` to Hub repo ``repo_id`` using ``hf upload``.

    Returns the Hub repo URL (best-effort parsed from CLI output).
    """
    local_path = Path(local_dir)
    if not local_path.is_dir():
        raise FileNotFoundError(f"Upload directory does not exist: {local_path}")

    hf_bin = _resolve_hf_cli()
    cmd = [
        hf_bin,
        "upload",
        repo_id,
        str(local_path),
        "--repo-type",
        "model",
    ]
    if private:
        cmd.append("--private")
    if commit_message:
        cmd.extend(["--commit-message", commit_message])
    for pattern in exclude:
        cmd.extend(["--exclude", pattern])

    env = os.environ.copy()
    if disable_xet:
        env["HF_HUB_DISABLE_XET"] = "1"
    # Prefer token from env; do not put it on argv.
    token = env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        env["HF_TOKEN"] = token
        env["HUGGING_FACE_HUB_TOKEN"] = token

    logger.info("Running Hub upload via CLI: %s", " ".join(cmd))
    # Stream progress to the training log (do not capture — multi-GB uploads are long).
    completed = subprocess.run(cmd, check=False, env=env)
    if completed.returncode != 0:
        raise RuntimeError(
            f"`hf upload` failed with exit {completed.returncode} for {repo_id}"
        )

    url = f"https://huggingface.co/{repo_id}"
    logger.info("Hub upload finished: %s", url)
    return url
