#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 Chenchen Gu
# Adapted from https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm.py
#   - Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# LEGACY STACK: paper-era training path (custom transformers fork, old FSDP wrap).
# Prefer train_logit_distill.py for the modern stack. Keep hash_key for KGW A/B runs.

# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import functools
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Optional, Union

import datasets
import evaluate
import torch
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.testing_utils import CaptureLogger
from transformers.trainer_pt_utils import get_module_class_from_name
from transformers.trainer_utils import FSDPOption, get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from watermarks.aar.aar_watermark import AarWatermark
from watermarks.hf_cli_upload import upload_folder_via_hf_cli
from watermarks.kgw.kgw_watermark import KGWWatermark
from watermarks.kth.kth_watermark import KTHWatermark
from watermarks.watermark_types import WatermarkType


require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _config_dict_is_mistral(config_dict: dict) -> bool:
    if config_dict.get("model_type") == "mistral":
        return True
    architectures = config_dict.get("architectures") or []
    return any("Mistral" in str(a) for a in architectures)


def _patch_llama_rope_theta() -> None:
    """Honor config.rope_theta when the paper-era Llama stack ignores it.

    Mistral-7B-v0.3 uses rope_theta=1e6; without this, remapping mistral→llama
    silently trains with the default base=10000 RoPE and diverges from the
    pretrained weights.
    """
    from transformers.models.llama import modeling_llama as llama_modeling

    if getattr(llama_modeling.LlamaAttention, "_rope_theta_patched", False):
        return

    def _init_rope(self):
        base = float(getattr(self.config, "rope_theta", 10000.0) or 10000.0)
        if self.config.rope_scaling is None:
            self.rotary_emb = llama_modeling.LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=base,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = llama_modeling.LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=base,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = llama_modeling.LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=base,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    llama_modeling.LlamaAttention._init_rope = _init_rope
    llama_modeling.LlamaAttention._rope_theta_patched = True
    logger.info("Patched LlamaAttention._init_rope to honor config.rope_theta")


def _load_config_legacy(model_name_or_path: Optional[str], config_name: Optional[str], config_kwargs: dict):
    """Load AutoConfig, remapping Mistral → Llama on the paper-era fork (no Mistral)."""
    from transformers import LlamaConfig

    path = config_name or model_name_or_path
    if path is None:
        return None, False

    config_dict, _ = PretrainedConfig.get_config_dict(path, **config_kwargs)
    if not _config_dict_is_mistral(config_dict):
        return AutoConfig.from_pretrained(path, **config_kwargs), False

    _patch_llama_rope_theta()
    remapped = dict(config_dict)
    rope_theta = remapped.get("rope_theta", 10000.0)
    remapped["model_type"] = "llama"
    remapped["architectures"] = ["LlamaForCausalLM"]
    # Unused by Llama; drop so from_dict does not warn loudly on unknowns.
    remapped.pop("sliding_window", None)
    remapped.pop("attention_dropout", None)
    config = LlamaConfig.from_dict(remapped)
    # from_dict may drop non-signature attrs depending on version; keep RoPE base.
    if not hasattr(config, "rope_theta") or config.rope_theta is None:
        config.rope_theta = rope_theta
    logger.warning(
        "Legacy transformers fork has no Mistral classes; loading %s as Llama "
        "(FSDP wrap LlamaDecoderLayer, rope_theta=%s).",
        path,
        getattr(config, "rope_theta", rope_theta),
    )
    return config, True


def _legacy_converted_weights_dir(model_name_or_path: str, cache_dir: Optional[str] = None) -> Path:
    root = Path(cache_dir or os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    slug = model_name_or_path.strip("/").replace("/", "__")
    return root / "legacy_pytorch_converted" / slug


def _ensure_legacy_pytorch_weights(
    model_name_or_path: str,
    cache_dir: Optional[str] = None,
    use_auth_token=None,
    is_main_process: bool = True,
) -> str:
    """Materialize pytorch_model*.bin for safetensors-only Hub models (e.g. Mistral-7B-v0.3).

    The paper-era transformers fork cannot load .safetensors weights.
    """
    import shutil
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from transformers.modeling_utils import shard_checkpoint

    out_dir = _legacy_converted_weights_dir(model_name_or_path, cache_dir)
    marker = out_dir / ".conversion_complete"
    if marker.exists() and (list(out_dir.glob("pytorch_model*.bin")) or (out_dir / "pytorch_model.bin").exists()):
        return str(out_dir)

    if not is_main_process:
        # Other ranks wait for rank0 conversion (filesystem sync on network volume).
        for _ in range(3600):
            if marker.exists():
                return str(out_dir)
            import time

            time.sleep(2)
        raise RuntimeError(f"Timed out waiting for safetensors→bin conversion at {out_dir}")

    logger.warning(
        "Legacy stack cannot load safetensors; converting %s → pytorch_model*.bin under %s",
        model_name_or_path,
        out_dir,
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    snap_kwargs = {
        "repo_id": model_name_or_path,
        "cache_dir": cache_dir,
        "local_files_only": False,
    }
    # huggingface-hub 0.16 uses use_auth_token
    if use_auth_token:
        snap_kwargs["use_auth_token"] = use_auth_token
    snap = Path(
        snapshot_download(
            **snap_kwargs,
            allow_patterns=[
                "*.safetensors",
                "*.json",
                "tokenizer*",
                "*.model",
                "special_tokens_map.json",
                "added_tokens.json",
            ],
        )
    )

    # Already has bins in the snapshot?
    if list(snap.glob("pytorch_model*.bin")) or (snap / "pytorch_model.bin").exists():
        return str(snap)

    index_path = snap / "model.safetensors.index.json"
    single = snap / "model.safetensors"
    state = {}
    if index_path.exists():
        index = json.loads(index_path.read_text())
        for shard_name in sorted(set(index["weight_map"].values())):
            state.update(load_file(str(snap / shard_name), device="cpu"))
    elif single.exists():
        state = load_file(str(single), device="cpu")
    else:
        raise FileNotFoundError(f"No safetensors weights found under {snap}")

    shards, index_json = shard_checkpoint(state, max_shard_size="10GB")
    for shard_file, shard in shards.items():
        torch.save(shard, out_dir / shard_file)
    if index_json is not None:
        (out_dir / "pytorch_model.bin.index.json").write_text(json.dumps(index_json, indent=2))

    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer.model",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ):
        src = snap / name
        if src.exists():
            shutil.copy2(src, out_dir / name)

    # Help AutoConfig/Llama load without model_type=mistral in the copied config.
    cfg_path = out_dir / "config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        cfg["model_type"] = "llama"
        cfg["architectures"] = ["LlamaForCausalLM"]
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")

    marker.write_text("ok\n")
    logger.info("Wrote legacy pytorch bins to %s", out_dir)
    return str(out_dir)


def _load_causal_lm_legacy(model_name_or_path: str, config, load_as_llama: bool, **from_pretrained_kwargs):
    if load_as_llama:
        from transformers import LlamaForCausalLM

        return LlamaForCausalLM.from_pretrained(
            model_name_or_path,
            config=config,
            **from_pretrained_kwargs,
        )
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=config,
        **from_pretrained_kwargs,
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    watermark_type: WatermarkType = field(
        default=None,
        metadata={
            "help": (
                "Type of watermark to use."
            ),
            "choices": [WatermarkType.AAR, WatermarkType.KGW, WatermarkType.KTH],
        },
    )
    watermark_seed: int = field(
        default=42,
        metadata={
            "help": (
                "Seed for watermarking."
            )
        },
    )
    aar_watermark_k: int = field(
        default=1,
        metadata={
            "help": (
                "Number of previous tokens to hash for Aar watermark."
            )
        }
    )
    kth_watermark_key_len: int = field(
        default=256,
        metadata={
            "help": (
                "Key length for KTH watermark."
            )
        }
    )
    kth_watermark_num_shifts: int = field(
        default=1,
        metadata={
            "help": (
                "Number of possible shifts for KTH watermark."
            )
        }
    )
    kgw_watermark_gamma: float = field(
        default=0.5,
        metadata={
            "help": "Gamma for watermark, i.e. proportion of vocab that is greenlist."
        },
    )
    kgw_watermark_delta: float = field(
        default=2.0,
        metadata={
            "help": "Delta for watermark, i.e. value to add to logits in greenlist."
        },
    )
    kgw_watermark_seeding_scheme: str = field(
        default="simple_1",
        metadata={
            "help": "Seeding scheme to use for watermark. See kgw_watermarking for more details."
        },
    )
    kgw_watermark_hash_key: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "KGW PRF hash_key / salt. If unset, uses the seeding-scheme default "
                "(15485863 for simple_0/simple_1/simple_2)."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set. (Only if streaming is False.)"
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


@dataclass
class LogitsDistillTrainingArguments(TrainingArguments):
    """Add some custom training arguments."""
    save_checkpoint_models: bool = field(
        default=False,
        metadata={"help": "Save model at every checkpoint, no deletion, no optimizer states."},
    )


def _normalize_fsdp_optim_state_dict(osd):
    """Convert scalar Adam state (e.g. step: int) to tensors for PyTorch 2.4+ FSDP.

    Legacy transformers saves ``step`` as a Python int. ``FSDP.scatter_full_optim_state_dict``
    then calls ``.cpu()`` on every state value and crashes with
    ``AttributeError: 'int' object has no attribute 'cpu'``.
    """
    if not isinstance(osd, dict) or "state" not in osd:
        return osd
    for st in osd["state"].values():
        if not isinstance(st, dict):
            continue
        for key, val in list(st.items()):
            # bool is a subclass of int — leave it alone
            if isinstance(val, bool):
                continue
            if isinstance(val, int):
                st[key] = torch.tensor(val, dtype=torch.int64)
            elif isinstance(val, float):
                st[key] = torch.tensor(val, dtype=torch.float32)
    return osd


class WatermarkLogitsDistillTrainer(Trainer):
    def __init__(
        self,
        teacher_model: PreTrainedModel,
        watermarker: Union[AarWatermark, KTHWatermark, KGWWatermark],
        argmax_watermark: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.teacher_model = self._fsdp_teacher_model(self.teacher_model)
        self.teacher_model.eval()
        self.watermarker = watermarker
        self.argmax_watermark = argmax_watermark
        if self.argmax_watermark:
            self.loss_fct = torch.nn.CrossEntropyLoss()
        else:
            self.loss_fct = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """Resume FSDP optim state after normalizing int/float Adam scalars."""
        _orig_load = torch.load

        def _load_and_normalize(*args, **kwargs):
            obj = _orig_load(*args, **kwargs)
            if isinstance(obj, dict) and "state" in obj and "param_groups" in obj:
                obj = _normalize_fsdp_optim_state_dict(obj)
            return obj

        torch.load = _load_and_normalize
        try:
            return super()._load_optimizer_and_scheduler(checkpoint)
        finally:
            torch.load = _orig_load

    def push_to_hub(self, commit_message=None, blocking=True, **kwargs):
        """Push via ``hf upload`` CLI instead of Trainer/huggingface_hub native push."""
        del blocking  # CLI upload is synchronous
        url = None
        if self.is_world_process_zero():
            card_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ("commit_message", "blocking", "token", "revision")
            }
            try:
                self.create_model_card(**card_kwargs)
            except Exception as e:
                logger.warning("create_model_card failed before Hub upload: %s", e)

            repo_id = self.args.hub_model_id or Path(self.args.output_dir).name
            private = bool(getattr(self.args, "hub_private_repo", False))
            msg = commit_message or kwargs.get("commit_message") or "End of training"
            url = upload_folder_via_hf_cli(
                repo_id=repo_id,
                local_dir=self.args.output_dir,
                private=private,
                commit_message=msg,
            )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()
        return url

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute distillation loss.

        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if "labels" in inputs:
            labels = inputs.pop("labels")

        outputs = model(**inputs)

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)

        # argmax watermark, use cross entropy loss against one-hot labels
        if self.argmax_watermark:
            watermark_tokens = self.watermarker.watermark_logits_argmax(
                inputs["input_ids"],
                teacher_outputs.logits,
            )

            # compute cross entropy loss
            loss = self.loss_fct(
                outputs.logits.view(-1, outputs.logits.shape[-1]),
                watermark_tokens.view(-1),
            )
        else:  # if not argmax, do distillation against distorted distribution
            # get watermarked logits
            watermarked_logits = self.watermarker.watermark_logits(inputs["input_ids"], teacher_outputs.logits)

            # compute distillation loss
            loss = self.loss_fct(
                torch.nn.functional.log_softmax(outputs.logits, dim=-1),
                torch.nn.functional.log_softmax(watermarked_logits, dim=-1),
            ) / outputs.logits.shape[1]

        return (loss, outputs) if return_outputs else loss
    

    def _save(self, output_dir: Optional[str] = None, **kwargs):
        super()._save(output_dir=output_dir, **kwargs)
        try:
            output_dir = output_dir if output_dir is not None else self.args.output_dir
            watermark_config = {}
            for k, v in vars(self.watermarker).items():
                if isinstance(v, (str, int, float, bool, list)):
                    watermark_config[k] = v
            config_dir = os.path.join(output_dir, "watermark_config.json")
            with open(config_dir, "w") as f:
                json.dump(watermark_config, f)
        except Exception as e:
            print(f"Failed to save watermark config: {e}")

    def _save_checkpoint(self, *args, **kwargs):
        """
        If self.args.save_checkpoint_models is True, save model at every checkpoint, no optimizer states.
        Save checkpoint normally as well.
        """
        if self.args.save_checkpoint_models:
            save_folder = f"model-step-{self.state.global_step}"
            run_dir = self.args.output_dir
            output_dir = os.path.join(run_dir, save_folder)
            self.save_model(output_dir, _internal_call=True)
        super()._save_checkpoint(*args, **kwargs)

    def _fsdp_teacher_model(self, model):
        if self.fsdp is not None:
            # PyTorch FSDP!
            from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.fully_sharded_data_parallel import MixedPrecision
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

            if FSDPOption.OFFLOAD in self.args.fsdp:
                cpu_offload = CPUOffload(offload_params=True)
            else:
                cpu_offload = CPUOffload(offload_params=False)

            auto_wrap_policy = None
            if FSDPOption.AUTO_WRAP in self.args.fsdp:
                if self.args.fsdp_min_num_params > 0:
                    auto_wrap_policy = functools.partial(
                        size_based_auto_wrap_policy, min_num_params=self.args.fsdp_min_num_params
                    )
                elif self.args.fsdp_transformer_layer_cls_to_wrap is not None:
                    transformer_cls_to_wrap = get_module_class_from_name(
                        model, self.args.fsdp_transformer_layer_cls_to_wrap
                    )
                    if transformer_cls_to_wrap is None:
                        raise Exception("Could not find the transformer layer class to wrap in the model.")
                    auto_wrap_policy = functools.partial(
                        transformer_auto_wrap_policy,
                        # Transformer layer class to wrap
                        transformer_layer_cls={transformer_cls_to_wrap},
                    )
            mixed_precision_policy = None
            dtype = None
            if self.args.fp16:
                dtype = torch.float16
            elif self.args.bf16:
                dtype = torch.bfloat16
            if dtype is not None:
                mixed_precision_policy = MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)
            if type(model) != FSDP:
                # XXX: Breaking the self.model convention but I see no way around it for now.
                model = FSDP(
                    model,
                    sharding_strategy=self.fsdp,
                    cpu_offload=cpu_offload,
                    auto_wrap_policy=auto_wrap_policy,
                    mixed_precision=mixed_precision_policy,
                    device_id=self.args.device,
                  )
        return model
    

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LogitsDistillTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            streaming=data_args.streaming,
        )
        # if "validation" not in raw_datasets.keys():
        #     raw_datasets["validation"] = load_dataset(
        #         data_args.dataset_name,
        #         data_args.dataset_config_name,
        #         split=f"train[:{data_args.validation_split_percentage}%]",
        #         cache_dir=model_args.cache_dir,
        #         use_auth_token=True if model_args.use_auth_token else None,
        #         streaming=data_args.streaming,
        #     )
        #     raw_datasets["train"] = load_dataset(
        #         data_args.dataset_name,
        #         data_args.dataset_config_name,
        #         split=f"train[{data_args.validation_split_percentage}%:]",
        #         cache_dir=model_args.cache_dir,
        #         use_auth_token=True if model_args.use_auth_token else None,
        #         streaming=data_args.streaming,
        #     )
    else:
        data_files = {}
        dataset_args = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        extension = (
            data_args.train_file.split(".")[-1]
            if data_args.train_file is not None
            else data_args.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = data_args.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{data_args.validation_split_percentage}%]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{data_args.validation_split_percentage}%:]",
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    load_as_llama = False
    if model_args.config_name or model_args.model_name_or_path:
        config, load_as_llama = _load_config_legacy(
            model_args.model_name_or_path,
            model_args.config_name,
            config_kwargs,
        )
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # Mistral-7B-v0.3 tokenizer.json needs a newer `tokenizers` than the paper-era
    # stack; force the slow SentencePiece path (tokenizer.model) instead.
    use_fast = model_args.use_fast_tokenizer
    if load_as_llama and use_fast:
        logger.warning(
            "Disabling fast tokenizer for remapped Mistral checkpoint "
            "(legacy tokenizers cannot parse its tokenizer.json)."
        )
        use_fast = False

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": use_fast,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_path = model_args.model_name_or_path
    if load_as_llama and model_args.model_name_or_path:
        load_path = _ensure_legacy_pytorch_weights(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
            is_main_process=training_args.local_rank in (-1, 0),
        )
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        from_pretrained_kwargs = dict(
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
        )
        model = _load_causal_lm_legacy(
            load_path,
            config,
            load_as_llama,
            **from_pretrained_kwargs,
        )
        teacher_model = _load_causal_lm_legacy(
            load_path,
            config,
            load_as_llama,
            **from_pretrained_kwargs,
        )
    else:
        if load_as_llama:
            from transformers import LlamaForCausalLM

            model = LlamaForCausalLM.from_config(config)
            teacher_model = LlamaForCausalLM.from_config(config)
        else:
            model = AutoModelForCausalLM.from_config(config)
            teacher_model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
        teacher_model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with training_args.main_process_first(desc="dataset map tokenization"):
        if not data_args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = data_args.max_train_samples
            if not data_args.streaming:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = data_args.max_eval_samples
            if not data_args.streaming:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)
        
    # Whether to use argmax watermarking, and train using cross-entropy loss vs one-hot labels.
    argmax_watermark = None
        
    # Initialize watermarker
    if model_args.watermark_type == WatermarkType.AAR:
        watermarker = AarWatermark(
            vocab_size=len(tokenizer),
            k=model_args.aar_watermark_k,
            seed=model_args.watermark_seed,
            device=device,
        )
        argmax_watermark = True
        assert argmax_watermark, "Aar watermark only supports argmax watermarking"
    elif model_args.watermark_type == WatermarkType.KTH:
        watermarker = KTHWatermark(
            vocab_size=len(tokenizer),
            key_len=model_args.kth_watermark_key_len,
            seed=model_args.watermark_seed,
            device=device,
            num_shifts=model_args.kth_watermark_num_shifts,
        )
        argmax_watermark = True
        assert argmax_watermark, "KTH watermark only supports argmax watermarking"
    elif model_args.watermark_type == WatermarkType.KGW:
        watermarker = KGWWatermark(
            vocab=tokenizer.get_vocab().values(),
            gamma=model_args.kgw_watermark_gamma,
            delta=model_args.kgw_watermark_delta,
            seeding_scheme=model_args.kgw_watermark_seeding_scheme,
            tokenizer=tokenizer,
            device=device,
            hash_key=model_args.kgw_watermark_hash_key,
        )
        argmax_watermark = False
        assert not argmax_watermark, "KGW watermark only supports non-argmax watermarking"
    else:
        raise ValueError(f"Invalid watermark type: {model_args.watermark_type}")
    
    assert argmax_watermark is not None, "argmax_watermark must be set"

    # Initialize our Trainer
    teacher_model = teacher_model.to(device)

    trainer = WatermarkLogitsDistillTrainer(
        teacher_model=teacher_model,
        watermarker=watermarker,
        argmax_watermark=argmax_watermark,
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if training_args.do_eval and not is_torch_tpu_available()
        else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        if data_args.max_train_samples is not None:
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        # max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        # metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        if data_args.max_eval_samples is not None:
            metrics["eval_samples"] = data_args.max_eval_samples

        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
