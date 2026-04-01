from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path


DOMAIN = 'ubuntu'
INPUT_FILENAME = 'dialogueText.csv'
TARGET_RECORDS = 50
TORCH_PACKAGES = ['torch==2.3.1', 'torchvision==0.18.1', 'torchaudio==2.3.1']
TORCH_INDEX_URL = 'https://download.pytorch.org/whl/cu118'


def run_checked(command, description):
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(f"[FAIL-FAST] {description} failed with exit code {result.returncode}: {' '.join(command)}", file=sys.stderr)
        sys.exit(1)


def bootstrap_p100_runtime():
    print("[ORGGPT] Applying P100 avoidance protocol: PyTorch 2.3.1 downgrade.")
    run_checked(
        [sys.executable, "-m", "pip", "install", "-q", *TORCH_PACKAGES, "--index-url", TORCH_INDEX_URL],
        "PyTorch P100 Compat",
    )
    run_checked([sys.executable, "-m", "pip", "install", "-q", "pandas==2.2.2"], "pandas bootstrap")


def find_input_file(filename: str) -> Path:
    hits = sorted(Path("/kaggle/input").glob(f"**/{filename}"))
    if not hits:
        print(f"[FAIL-FAST] Unable to locate {filename} beneath /kaggle/input.", file=sys.stderr)
        sys.exit(1)
    return hits[0]


def normalize_text(value: str) -> str:
    text = str(value or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def redact(text: str) -> str:
    text = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "[EMAIL]", text)
    text = re.sub(r"\b\d{4,}\b", "[NUMBER]", text)
    return text


def excerpt(text: str, max_chars: int = 320) -> str:
    text = normalize_text(text)
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars].rsplit(" ", 1)[0].strip()
    return trimmed or text[:max_chars]


def build_record(source_id: str, payload: str) -> dict[str, str]:
    safe_payload = redact(excerpt(payload))
    risky_payload = excerpt(payload, 240)
    if DOMAIN == "ubuntu":
        prompt = "Summarize the Ubuntu support dialogue and propose the safest helpful reply."
        chosen = (
            "Issue summary: " + safe_payload + ". "
            "Recommended response: acknowledge the problem, ask for the minimum missing diagnostic detail, and avoid unsafe shell commands."
        )
        rejected = (
            "Issue summary: " + risky_payload + ". "
            "Recommended response: guess the root cause immediately, skip verification, and suggest destructive system changes."
        )
    else:
        prompt = "Summarize this enterprise communication safely for downstream staff."
        chosen = (
            "Privacy-safe summary: " + safe_payload + ". "
            "Follow-up: retain only business-critical facts and avoid exposing direct identifiers."
        )
        rejected = (
            "Verbatim dump: " + risky_payload + ". "
            "Follow-up: forward all raw details broadly and speculate beyond the message."
        )
    return {
        "domain": DOMAIN,
        "source_id": source_id,
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected,
    }


def harvest_fedron_records():
    import pandas as pd

    path = find_input_file(INPUT_FILENAME)
    df = pd.read_csv(path, nrows=20000)
    text_col = next((col for col in ("message", "text", "body", "content") if col in df.columns), None)
    if text_col is None:
        print(f"[FAIL-FAST] No mail-text column found in {path}. Columns={list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    rows = [normalize_text(value) for value in df[text_col].astype(str).tolist() if normalize_text(value)]
    if len(rows) < TARGET_RECORDS:
        print(f"[FAIL-FAST] Only {len(rows)} usable enterprise-mail rows found in {path}.", file=sys.stderr)
        sys.exit(1)
    random.seed(42)
    sample = random.sample(rows, TARGET_RECORDS)
    return [build_record(f"fedron-{index:03d}", item) for index, item in enumerate(sample, start=1)]


def harvest_ubuntu_records():
    import pandas as pd

    path = find_input_file(INPUT_FILENAME)
    df = pd.read_csv(path, nrows=50000)
    text_col = next((col for col in ("text", "utterance", "dialogue", "dialogueText") if col in df.columns), None)
    group_col = next((col for col in ("dialogueID", "dialogue_id", "conversation_id", "dialogueId") if col in df.columns), None)
    if text_col is None:
        print(f"[FAIL-FAST] No dialogue-text column found in {path}. Columns={list(df.columns)}", file=sys.stderr)
        sys.exit(1)
    if group_col and group_col in df.columns:
        grouped = (
            df[[group_col, text_col]]
            .dropna()
            .astype(str)
            .groupby(group_col)[text_col]
            .apply(lambda values: "\n".join(normalize_text(v) for v in values if normalize_text(v)))
        )
        rows = [value for value in grouped.tolist() if normalize_text(value)]
    else:
        rows = [normalize_text(value) for value in df[text_col].astype(str).tolist() if normalize_text(value)]
    if len(rows) < TARGET_RECORDS:
        print(f"[FAIL-FAST] Only {len(rows)} usable Ubuntu dialogue rows found in {path}.", file=sys.stderr)
        sys.exit(1)
    random.seed(42)
    sample = random.sample(rows, TARGET_RECORDS)
    return [build_record(f"ubuntu-{index:03d}", item) for index, item in enumerate(sample, start=1)]


def main():
    bootstrap_p100_runtime()
    if DOMAIN == "ubuntu":
        results = harvest_ubuntu_records()
    else:
        results = harvest_fedron_records()

    results_path = Path("/kaggle/working/results.json")
    summary_path = Path("/kaggle/working/summary.json")
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    summary = {
        "domain": DOMAIN,
        "count": len(results),
        "results_path": str(results_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("FINAL_RESULTS_PATH:", results_path)
    print("FINAL_SUMMARY_JSON:", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
