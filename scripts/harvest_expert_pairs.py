from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


DEFAULT_KERNEL_ROOT = Path("/tmp/orggpt_kernel_src")
DEFAULT_OUTPUT_ROOT = Path("/tmp/orggpt_dpo_raw")
DEFAULT_NODE_IDS = ("01", "02", "03", "04", "05")
BASE_POLICY = "Mandate a 20% reduction in cloud computing expenditures."
CONSTITUTION = ["Do not implement cuts that disrupt revenue services."]


class DummyIEDM:
    def compute_distortion(self, _root: str, _level) -> float:
        return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest real expert DPO pairs from Kaggle kernel sources.")
    parser.add_argument("--kernel-root", default=str(DEFAULT_KERNEL_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--records-per-node", type=int, default=20)
    parser.add_argument("--node-ids", nargs="+", default=list(DEFAULT_NODE_IDS))
    return parser.parse_args()


def load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


async def collect_records(module: ModuleType, target_count: int, node_id: str) -> tuple[list[dict], dict]:
    prompt = f"Role: VP. Rephrase policy for subordinates: {BASE_POLICY}"
    runner = module.AsyncSimulationRunner(DummyIEDM(), BASE_POLICY, CONSTITUTION)
    records: list[dict] = []
    attempts = 0

    try:
        while len(records) < target_count:
            batch_size = max(1, target_count - len(records))
            baseline_batch, audited_batch = await asyncio.gather(
                runner.run_surge(batch_size, False),
                runner.run_surge(batch_size, True),
            )
            attempts += batch_size
            for baseline_trial, audited_trial in zip(baseline_batch, audited_batch):
                if not baseline_trial.draft or not audited_trial.draft:
                    continue

                if audited_trial.is_compliant:
                    chosen_text = audited_trial.draft
                    rejected_text = baseline_trial.draft
                    chosen_source = "audited"
                else:
                    chosen_text = baseline_trial.draft
                    rejected_text = audited_trial.draft
                    chosen_source = "baseline"

                if chosen_text.strip() == rejected_text.strip():
                    continue

                records.append(
                    {
                        "system": "You are an internal compliance assistant.",
                        "prompt": prompt,
                        "chosen": chosen_text,
                        "rejected": rejected_text,
                        "node_id": node_id,
                        "trial_id": len(records) + 1,
                        "chosen_source": chosen_source,
                        "baseline_is_compliant": bool(baseline_trial.is_compliant),
                        "audited_is_compliant": bool(audited_trial.is_compliant),
                    }
                )
                if len(records) >= target_count:
                    break
    finally:
        await runner._aio_client.aclose()

    summary = {
        "node_id": node_id,
        "target_count": target_count,
        "collected_count": len(records),
        "attempted_trials": attempts,
    }
    return records, summary


async def harvest_node(kernel_root: Path, output_root: Path, node_id: str, records_per_node: int) -> dict:
    script_path = kernel_root / f"node_{node_id}" / f"orggpt-v3-1-expert-harvest-node-{node_id}.py"
    if not script_path.exists():
        raise FileNotFoundError(f"Kernel source not found: {script_path}")

    module = load_module(script_path, f"orggpt_harvest_node_{node_id}")
    records, summary = await collect_records(module, records_per_node, node_id)

    output_dir = output_root / f"node_{node_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


async def main() -> None:
    args = parse_args()
    kernel_root = Path(args.kernel_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    for node_id in args.node_ids:
        summary = await harvest_node(kernel_root, output_root, node_id, args.records_per_node)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))

    manifest = {
        "total_nodes": len(summaries),
        "records_per_node": args.records_per_node,
        "total_records": sum(item["collected_count"] for item in summaries),
        "summaries": summaries,
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(main())
