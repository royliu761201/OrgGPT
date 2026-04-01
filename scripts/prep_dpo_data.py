from __future__ import annotations

import json
import sys
from pathlib import Path


DEFAULT_INPUT_ROOT = Path("/tmp/orggpt_dpo_raw")
DEFAULT_OUTPUT_PATH = Path("/tmp/orggpt/orggpt_dpo_train.jsonl")
PROMPT_KEYS = ("prompt", "question", "query", "user", "user_prompt", "request", "task")
SYSTEM_KEYS = ("system", "developer", "system_prompt", "developer_prompt")
CHOSEN_KEYS = (
    "chosen",
    "preferred",
    "accepted",
    "better",
    "good",
    "positive",
    "response_chosen",
    "chosen_response",
)
REJECTED_KEYS = (
    "rejected",
    "dispreferred",
    "worse",
    "bad",
    "negative",
    "response_rejected",
    "rejected_response",
)
LIST_CONTAINER_KEYS = ("data", "items", "records", "results", "examples", "samples", "rows")
METRIC_RECORD_KEYS = ("baseline_mean", "audited_mean", "compliance_rate")
RAW_CONTEXT_DOMAINS = {"enron", "fedron", "ubuntu"}


def find_result_files(root_dir: Path) -> list[Path]:
    return sorted(root_dir.rglob("results.json"))


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def unwrap_records(payload) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in LIST_CONTAINER_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    raise ValueError("Unsupported JSON payload type; expected dict or list.")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    for role in ("system", "user", "assistant"):
        prefix = f"<|im_start|>{role}\n"
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    if text.endswith("<|im_end|>"):
        text = text[: -len("<|im_end|>")].strip()
    return text


def normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, list):
        parts = [normalize_text(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("content", "text", "value", "message", "response", "output", "answer"):
            if key in value:
                return normalize_text(value[key])
        return clean_text(json.dumps(value, ensure_ascii=False, sort_keys=True))
    return clean_text(str(value))


def first_text(mapping: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
        if key in mapping:
            text = normalize_text(mapping[key])
            if text:
                return text
    return ""


def normalize_role(raw_role) -> str:
    role = str(raw_role or "").strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"assistant", "model", "bot", "gpt"}:
        return "assistant"
    if role in {"system", "developer"}:
        return "system"
    return role


def extract_from_messages(messages: list[dict]) -> dict[str, str]:
    system_parts: list[str] = []
    user_parts: list[str] = []
    assistant_parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = normalize_role(
            message.get("role")
            or message.get("from")
            or message.get("speaker")
            or message.get("author")
        )
        content = normalize_text(
            message.get("content")
            or message.get("text")
            or message.get("value")
            or message.get("message")
        )
        if not content:
            continue
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
        elif role == "assistant":
            assistant_parts.append(content)
    return {
        "system": "\n\n".join(system_parts).strip(),
        "user": "\n\n".join(user_parts).strip(),
        "assistant": "\n\n".join(assistant_parts).strip(),
    }


def build_prompt(system_text: str, prompt_text: str) -> str:
    prompt_parts: list[str] = []
    if system_text:
        prompt_parts.append(f"<|im_start|>system\n{system_text}\n<|im_end|>")
    prompt_parts.append(f"<|im_start|>user\n{prompt_text}\n<|im_end|>")
    prompt_parts.append("<|im_start|>assistant\n")
    return "\n".join(prompt_parts)


def build_raw_context_prompt(source_id: str, transcript: str, task_text: str) -> str:
    prompt_text = (
        f"Full original source transcript ({source_id}):\n{transcript}\n\n"
        f"Task:\n{task_text}"
    )
    return prompt_text


def extract_segment(text: str, prefix: str, suffix: str) -> str:
    normalized = normalize_text(text)
    if not normalized.startswith(prefix):
        return ""
    body = normalized[len(prefix) :].strip()
    head, separator, _ = body.partition(suffix)
    if not separator:
        return ""
    return head.strip(" .")


def extract_metric_record(record: dict) -> dict[str, str] | None:
    if not all(key in record for key in METRIC_RECORD_KEYS):
        return None

    baseline_mean = float(record["baseline_mean"])
    audited_mean = float(record["audited_mean"])
    compliance_rate = float(record["compliance_rate"])

    node_id = normalize_text(record.get("node_id") or record.get("source_id") or "unknown-node")
    trial_id = normalize_text(record.get("trial_id") or "")
    trial_line = f"Trial: {trial_id}\n" if trial_id else ""

    system_text = "You are a direct operations analyst. Be concrete and avoid bureaucratic filler."
    prompt_text = (
        "OrgGPT telemetry snapshot:\n"
        f"Node: {node_id}\n"
        f"{trial_line}"
        f"Baseline drift mean: {baseline_mean:.6f}\n"
        f"Audited drift mean: {audited_mean:.6f}\n"
        f"Compliance rate: {compliance_rate:.2%}\n\n"
        "Write one concise operational recommendation that preserves utility while naming the compliance trade-off."
    )
    if audited_mean <= baseline_mean:
        chosen_text = (
            f"Prefer audited. Drift improves from {baseline_mean:.6f} to {audited_mean:.6f} "
            f"with compliance at {compliance_rate:.2%}. Keep the audited path and report the measured gain directly."
        )
    else:
        chosen_text = (
            f"Keep baseline for utility. Drift worsens from {baseline_mean:.6f} to {audited_mean:.6f} "
            f"under audited mode, while compliance is {compliance_rate:.2%}. State the trade-off plainly instead of forcing audited."
        )
    rejected_text = (
        f"Prefer audited automatically. Baseline drift mean: {audited_mean:.6f}. "
        f"Audited drift mean: {baseline_mean:.6f}. Compliance rate: {max(0.0, 1.0 - compliance_rate):.2%}. "
        "Swap the metrics and bury the utility trade-off under compliance boilerplate."
    )
    return {
        "prompt": build_prompt(system_text, prompt_text),
        "chosen": f"{chosen_text}\n<|im_end|>",
        "rejected": f"{rejected_text}\n<|im_end|>",
    }


def extract_raw_context_record(record: dict) -> dict[str, str] | None:
    domain = normalize_text(record.get("domain")).lower()
    if domain not in RAW_CONTEXT_DOMAINS:
        return None

    transcript = normalize_text(
        record.get("transcript")
        or record.get("source_transcript")
        or record.get("full_transcript")
        or record.get("context")
    )
    chosen_text = first_text(record, CHOSEN_KEYS)
    rejected_text = first_text(record, REJECTED_KEYS)
    source_id = normalize_text(record.get("source_id") or f"{domain}-record")
    task_text = normalize_text(record.get("task") or record.get("instruction"))

    if not transcript or not chosen_text or not rejected_text:
        return None

    if not task_text:
        if domain == "ubuntu":
            task_text = (
                "Write a direct helpful reply. Summarize the issue briefly, ask for the minimum missing diagnostic detail, "
                "and only recommend safe reversible steps."
            )
        elif domain == "fedron":
            task_text = (
                "Write a privacy-safe operational summary. Keep business-critical facts, redact direct identifiers, "
                "and avoid bureaucratic filler."
            )
        else:
            task_text = (
                "Write a direct internal handoff note. Preserve concrete facts, deadlines, decisions, and open asks "
                "without adding compliance boilerplate."
            )

    if domain == "ubuntu":
        system_text = (
            "You are a concise Ubuntu support assistant. Stay grounded in the transcript, avoid evasive refusals, "
            "and prefer safe reversible guidance."
        )
    elif domain == "fedron":
        system_text = (
            "You are a concise enterprise assistant. Stay grounded in the transcript, keep utility high, "
            "and redact direct identifiers instead of dumping raw details."
        )
    else:
        system_text = (
            "You are a direct operations assistant. Stay grounded in the transcript, preserve useful detail, "
            "and avoid institutional boilerplate."
        )

    return {
        "domain": domain,
        "source_id": source_id,
        "prompt": build_prompt(system_text, build_raw_context_prompt(source_id, transcript, task_text)),
        "chosen": f"{chosen_text}\n<|im_end|>",
        "rejected": f"{rejected_text}\n<|im_end|>",
    }


def extract_harvest_record(record: dict) -> dict[str, str] | None:
    domain = normalize_text(record.get("domain"))
    if domain not in {"fedron", "ubuntu"}:
        return None

    source_id = normalize_text(record.get("source_id") or f"{domain}-record")
    raw_chosen_text = first_text(record, CHOSEN_KEYS)
    raw_rejected_text = first_text(record, REJECTED_KEYS)
    if not raw_chosen_text or not raw_rejected_text:
        return None

    if domain == "fedron":
        excerpt = extract_segment(raw_chosen_text, "Privacy-safe summary:", ". Follow-up:")
        if not excerpt:
            excerpt = extract_segment(raw_rejected_text, "Verbatim dump:", ". Follow-up:")
        if not excerpt:
            return None
        system_text = "You are a concise enterprise assistant. Use the provided excerpt and do not invent missing context."
        prompt_text = (
            f"Enterprise communication excerpt ({source_id}):\n{excerpt}\n\n"
            "Write a privacy-safe summary for downstream staff and keep only business-critical detail."
        )
        chosen_text = (
            f"Summary: {excerpt}\n"
            "Action: keep only business-critical facts and omit direct identifiers."
        )
        rejected_text = (
            f"Raw dump: {excerpt}\n"
            "Action: forward the full details broadly and speculate beyond the message."
        )
    else:
        excerpt = extract_segment(raw_chosen_text, "Issue summary:", ". Recommended response:")
        if not excerpt:
            excerpt = extract_segment(raw_rejected_text, "Issue summary:", ". Recommended response:")
        if not excerpt:
            return None
        system_text = "You are a concise Ubuntu support assistant. Use the snippet directly and avoid evasive replies."
        prompt_text = (
            f"Ubuntu support dialogue snippet ({source_id}):\n{excerpt}\n\n"
            "Summarize the issue and propose the safest helpful reply."
        )
        chosen_text = (
            f"Issue: {excerpt}\n"
            "Safe reply: acknowledge the problem, ask for the minimum missing diagnostic detail, and avoid unsafe shell commands."
        )
        rejected_text = (
            f"Issue: {excerpt}\n"
            "Unsafe reply: guess the root cause immediately, skip verification, and suggest destructive system changes."
        )

    return {
        "prompt": build_prompt(system_text, prompt_text),
        "chosen": f"{chosen_text}\n<|im_end|>",
        "rejected": f"{rejected_text}\n<|im_end|>",
    }


def extract_example(record: dict) -> dict[str, str]:
    if not isinstance(record, dict):
        raise ValueError("Each training record must be a JSON object.")

    raw_context_example = extract_raw_context_record(record)
    if raw_context_example is not None:
        return raw_context_example

    metric_example = extract_metric_record(record)
    if metric_example is not None:
        return metric_example

    harvest_example = extract_harvest_record(record)
    if harvest_example is not None:
        return harvest_example

    message_bundle = {}
    if isinstance(record.get("messages"), list):
        message_bundle = extract_from_messages(record["messages"])

    system_text = first_text(record, SYSTEM_KEYS) or message_bundle.get("system", "")
    prompt_text = first_text(record, PROMPT_KEYS) or message_bundle.get("user", "")
    chosen_text = first_text(record, CHOSEN_KEYS)
    rejected_text = first_text(record, REJECTED_KEYS)

    if not prompt_text:
        raise ValueError("Missing prompt text.")
    if not chosen_text:
        chosen_text = message_bundle.get("assistant", "")
    if not chosen_text:
        raise ValueError("Missing chosen text.")
    if not rejected_text:
        raise ValueError("Missing rejected text.")

    return {
        "prompt": build_prompt(system_text, prompt_text),
        "chosen": f"{chosen_text}\n<|im_end|>",
        "rejected": f"{rejected_text}\n<|im_end|>",
    }


def convert_directory(input_root: Path, output_path: Path) -> int:
    result_files = find_result_files(input_root)
    if not result_files:
        raise FileNotFoundError(f"No results.json files found under: {input_root}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_records = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for path in result_files:
            payload = load_json(path)
            for record in unwrap_records(payload):
                example = extract_example(record)
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")
                total_records += 1
    return total_records


def main(argv: list[str]) -> int:
    input_root = Path(argv[1]) if len(argv) > 1 else DEFAULT_INPUT_ROOT
    output_path = Path(argv[2]) if len(argv) > 2 else DEFAULT_OUTPUT_PATH
    total_records = convert_directory(input_root, output_path)
    print(f"Wrote {total_records} records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
