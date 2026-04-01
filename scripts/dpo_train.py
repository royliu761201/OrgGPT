from __future__ import annotations

import os

# Force single-card execution on physical GPU 4 before importing torch-backed libraries.
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(proxy_var, None)

import argparse
import inspect
import sys
from pathlib import Path

from datasets import load_dataset
from unsloth import FastLanguageModel

try:
    from unsloth import PatchDPOTrainer
except ImportError:
    PatchDPOTrainer = None

if PatchDPOTrainer is not None:
    PatchDPOTrainer()

from trl import DPOConfig, DPOTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config


TRAIN_DATA_PATH = Path("/tmp/orggpt/orggpt_dpo_train.jsonl")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "dpo"
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def patch_lora_config_kwargs() -> None:
    try:
        from peft import LoraConfig
    except ImportError:
        return

    original_init = LoraConfig.__init__
    if getattr(original_init, "_orggpt_filtered_kwargs_patch", False):
        return

    accepted_kwargs = set(inspect.signature(original_init).parameters)

    def patched_init(self, *args, **kwargs):
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in accepted_kwargs}
        return original_init(self, *args, **filtered_kwargs)

    patched_init._orggpt_filtered_kwargs_patch = True
    LoraConfig.__init__ = patched_init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OrgGPT DPO training entrypoint.")
    parser.add_argument(
        "--model-name",
        required=True,
        help="Unsloth-compatible 14B base or instruct checkpoint identifier.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for trainer checkpoints and the final adapter.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument(
        "--config-key",
        default="DPO_CONFIG",
        help="Config registry key on config.CONFIG to read lr/beta/max_len/batch_size from.",
    )
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    return parser.parse_args()


def resolve_precision_flags() -> tuple[bool, bool]:
    try:
        import torch
    except ImportError:
        return False, True

    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return True, False
    return False, True


def load_training_dataset():
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(f"DPO training file not found: {TRAIN_DATA_PATH}")

    dataset = load_dataset("json", data_files=str(TRAIN_DATA_PATH), split="train")
    required_columns = {"prompt", "chosen", "rejected"}
    missing_columns = sorted(required_columns.difference(dataset.column_names))
    if missing_columns:
        raise ValueError(
            "DPO dataset must include prompt/chosen/rejected columns. "
            f"Missing: {', '.join(missing_columns)}"
        )
    return dataset


def load_model_and_tokenizer(model_name: str, dpo_settings: dict, args: argparse.Namespace):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=dpo_settings["max_len"],
        dtype=None,
        load_in_4bit=True,
        local_files_only=True,
    )
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=list(DEFAULT_TARGET_MODULES),
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    model.config.use_cache = False
    return model, tokenizer


def build_training_args(output_dir: str, args: argparse.Namespace, dpo_settings: dict) -> DPOConfig:
    bf16_enabled, fp16_enabled = resolve_precision_flags()
    return DPOConfig(
        output_dir=output_dir,
        learning_rate=dpo_settings["lr"],
        beta=dpo_settings["beta"],
        max_length=dpo_settings["max_len"],
        per_device_train_batch_size=dpo_settings["batch_size"],
        num_train_epochs=args.num_train_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        bf16=bf16_enabled,
        fp16=fp16_enabled,
        seed=args.seed,
    )


def build_trainer(model, tokenizer, dataset, training_args: DPOConfig) -> DPOTrainer:
    trainer_kwargs = {
        "model": model,
        "ref_model": None,
        "args": training_args,
        "train_dataset": dataset,
    }
    try:
        return DPOTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        return DPOTrainer(tokenizer=tokenizer, **trainer_kwargs)


def main() -> None:
    args = parse_args()
    patch_lora_config_kwargs()
    try:
        dpo_settings = getattr(config.CONFIG, args.config_key)
    except AttributeError as exc:
        raise ValueError(f"Unknown DPO config key: {args.config_key}") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_training_dataset()
    model, tokenizer = load_model_and_tokenizer(args.model_name, dpo_settings, args)
    training_args = build_training_args(str(output_dir), args, dpo_settings)
    trainer = build_trainer(model, tokenizer, dataset, training_args)

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
