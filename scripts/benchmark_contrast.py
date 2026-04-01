from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

# Pin the run to the same physical L20 used for DPO training.
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["UNSLOTH_DISABLE_STATISTICS"] = "1"
for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(proxy_var, None)

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine.iedm import IEDM


DEFAULT_BASE_FALLBACK = Path("/jhdx0003008/models/Qwen2.5-14B-Instruct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Figure 2 contrast benchmark for OrgGPT.")
    parser.add_argument("--base-model", required=True, help="Requested base model reference.")
    parser.add_argument("--adapter-dir", required=True, help="Path to the trained DPO adapter directory.")
    parser.add_argument("--data-path", required=True, help="JSONL file with prompt/chosen/rejected rows.")
    parser.add_argument("--output-json", required=True, help="Final aggregated JSON results.")
    parser.add_argument("--batch-size", type=int, default=4, help="Generation batch size.")
    parser.add_argument("--max-new-tokens", type=int, default=192, help="Maximum new tokens per sample.")
    return parser.parse_args()


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant", 1)[-1].strip()
    if text.endswith("<|im_end|>"):
        text = text[: -len("<|im_end|>")].strip()
    return text


def load_records(data_path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    with data_path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                {
                    "index": index,
                    "domain": row.get("domain", ""),
                    "source_id": row.get("source_id", ""),
                    "prompt": row["prompt"],
                    "chosen": clean_text(row["chosen"]),
                    "rejected": clean_text(row.get("rejected", "")),
                }
            )
    if not records:
        raise ValueError(f"No benchmark rows found in {data_path}")
    return records


def resolve_dtype() -> torch.dtype:
    if torch.cuda.is_available() and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def resolve_base_model(requested_base_model: str) -> tuple[str, str | None]:
    requested_path = Path(requested_base_model)
    if requested_path.is_dir():
        return str(requested_path), None

    if requested_base_model == "unsloth/Qwen2.5-14B-instruct-bnb-4bit" and DEFAULT_BASE_FALLBACK.is_dir():
        note = (
            "Requested Unsloth 4-bit repo is not cached on L20; "
            f"using local Qwen2.5-14B-Instruct with 4-bit runtime quantization at {DEFAULT_BASE_FALLBACK}."
        )
        return str(DEFAULT_BASE_FALLBACK), note

    return requested_base_model, None


def load_model_and_tokenizer(model_ref: str, *, is_adapter: bool):
    dtype = resolve_dtype()
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
    )
    common_kwargs = {
        "device_map": "auto",
        "torch_dtype": dtype,
        "quantization_config": quantization_config,
        "local_files_only": Path(model_ref).is_dir(),
    }

    if is_adapter:
        model = AutoPeftModelForCausalLM.from_pretrained(model_ref, **common_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_ref, **common_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_ref, local_files_only=Path(model_ref).is_dir())
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    model.eval()
    return model, tokenizer


def batched(items: list[dict[str, str]], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def generate_batch(model, tokenizer, prompts: list[str], max_new_tokens: int) -> list[str]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    input_length = int(encoded["input_ids"].shape[1])
    with torch.inference_mode():
        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    predictions: list[str] = []
    for row_index in range(len(prompts)):
        completion_ids = outputs[row_index, input_length:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=False)
        predictions.append(clean_text(text))
    return predictions


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty value list.")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "pstdev": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def benchmark_track(
    name: str,
    model,
    tokenizer,
    records: list[dict[str, str]],
    engine: IEDM,
    batch_size: int,
    max_new_tokens: int,
) -> list[dict]:
    outputs: list[dict] = []
    total_batches = (len(records) + batch_size - 1) // batch_size
    for batch_index, (_, batch_records) in enumerate(batched(records, batch_size), start=1):
        prompts = [record["prompt"] for record in batch_records]
        predictions = generate_batch(model, tokenizer, prompts, max_new_tokens)
        for record, prediction in zip(batch_records, predictions):
            outputs.append(
                {
                    "index": record["index"],
                    "model": name,
                    "chosen_distortion": engine.compute_distortion(record["chosen"], [prediction]),
                    "rejected_distortion": engine.compute_distortion(record["rejected"], [prediction]) if record["rejected"] else 0.0,
                    "prompt_distortion": engine.compute_distortion(record["prompt"], [prediction]),
                    "prediction_preview": prediction[:160],
                }
            )
        print(f"[{name}] processed batch {batch_index}/{total_batches}")
    return outputs


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)
    output_path = Path(args.output_json)
    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    requested_base_model = args.base_model
    resolved_base_model, resolution_note = resolve_base_model(requested_base_model)
    records = load_records(data_path)
    engine = IEDM()

    print(f"Running base benchmark: requested={requested_base_model} resolved={resolved_base_model}")
    base_model, base_tokenizer = load_model_and_tokenizer(resolved_base_model, is_adapter=False)
    base_results = benchmark_track(
        "base",
        base_model,
        base_tokenizer,
        records,
        engine,
        args.batch_size,
        args.max_new_tokens,
    )
    del base_model
    torch.cuda.empty_cache()

    print(f"Running DPO benchmark: {adapter_dir}")
    adapter_label = adapter_dir.name
    dpo_model, dpo_tokenizer = load_model_and_tokenizer(str(adapter_dir), is_adapter=True)
    dpo_results = benchmark_track(
        adapter_label,
        dpo_model,
        dpo_tokenizer,
        records,
        engine,
        args.batch_size,
        args.max_new_tokens,
    )

    base_by_index = {row["index"]: row for row in base_results}
    dpo_by_index = {row["index"]: row for row in dpo_results}
    per_example: list[dict] = []
    deltas: list[float] = []
    dpo_better = 0
    base_better = 0
    ties = 0

    for record in records:
        index = record["index"]
        base_row = base_by_index[index]
        dpo_row = dpo_by_index[index]
        delta = base_row["chosen_distortion"] - dpo_row["chosen_distortion"]
        deltas.append(delta)
        if delta > 0:
            dpo_better += 1
        elif delta < 0:
            base_better += 1
        else:
            ties += 1
        per_example.append(
            {
                "index": index,
                "domain": record["domain"],
                "source_id": record["source_id"],
                "base_chosen_distortion": base_row["chosen_distortion"],
                "dpo_chosen_distortion": dpo_row["chosen_distortion"],
                "delta_base_minus_dpo": delta,
                "base_preview": base_row["prediction_preview"],
                "dpo_preview": dpo_row["prediction_preview"],
            }
        )

    summary = {
        "metadata": {
            "data_path": str(data_path),
            "num_records": len(records),
            "requested_base_model": requested_base_model,
            "resolved_base_model": resolved_base_model,
            "resolution_note": resolution_note,
            "adapter_dir": str(adapter_dir),
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
        },
        "summary": {
            "base": summarize([row["chosen_distortion"] for row in base_results]),
            adapter_label: summarize([row["chosen_distortion"] for row in dpo_results]),
            "delta_base_minus_dpo": summarize(deltas),
            "dpo_better_count": dpo_better,
            "base_better_count": base_better,
            "tie_count": ties,
        },
        "per_example": per_example,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))
    print(f"Saved benchmark summary to {output_path}")


if __name__ == "__main__":
    main()
