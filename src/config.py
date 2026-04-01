from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Dict, Iterable, List, Optional


class SecretMasker:
    """
    Redacts registered secret values from log strings before they are emitted.
    """

    def __init__(self, secrets: Optional[Iterable[Optional[str]]] = None):
        self._secrets: List[str] = []
        if secrets:
            for secret in secrets:
                self.register(secret)

    @classmethod
    def from_environment(cls, *env_vars: str) -> "SecretMasker":
        return cls(os.environ.get(env_var) for env_var in env_vars)

    def register(self, secret: Optional[str]) -> None:
        if not secret:
            return
        if secret in self._secrets:
            return
        self._secrets.append(secret)
        self._secrets.sort(key=len, reverse=True)

    def mask(self, value: Any) -> str:
        text = str(value)
        for secret in self._secrets:
            text = text.replace(secret, "[REDACTED]")
        return text


def _kaggle_path(root: str, *parts: str) -> str:
    path = PurePosixPath(root.rstrip("/"))
    for part in parts:
        path = path / part.strip("/")
    return str(path)


@dataclass
class OrgGPTConfig:
    """
    OrgGPT global configuration for Kaggle-native execution.
    """

    device: str = "cpu"
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY"))
    wandb_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("WANDB_API_KEY"))
    use_wandb: bool = field(
        default_factory=lambda: os.environ.get("USE_WANDB", "1").lower() not in {"0", "false", "no"}
    )
    wandb_project: str = field(default_factory=lambda: os.getenv("WANDB_PROJECT", "orggpt_neurips"))

    blue_team_model: str = "gemini-3-flash-preview"
    red_team_model: str = "gemini-3-flash-preview"
    embedding_model: str = "text-embedding-004"

    kaggle_input_root: str = field(default_factory=lambda: os.getenv("KAGGLE_INPUT_ROOT", "/kaggle/input"))
    enron_dataset_slug: str = field(default_factory=lambda: os.getenv("ORGGPT_ENRON_DATASET", "enron-email-dataset"))
    enron_filename: str = field(default_factory=lambda: os.getenv("ORGGPT_ENRON_FILENAME", "emails.csv"))
    bpic_dataset_slug: str = field(default_factory=lambda: os.getenv("ORGGPT_BPIC_DATASET", "bpi-challenge-2015"))
    bpic_filename: str = field(default_factory=lambda: os.getenv("ORGGPT_BPIC_FILENAME", "BPI_Challenge_2015.xes"))
    enron_path: str = field(init=False)
    bpic_path: str = field(init=False)
    secret_masker: SecretMasker = field(init=False)

    def __post_init__(self) -> None:
        kaggle_root = self.kaggle_input_root.rstrip("/")
        self.enron_path = os.getenv(
            "ORGGPT_ENRON_PATH",
            _kaggle_path(kaggle_root, self.enron_dataset_slug, self.enron_filename),
        )
        self.bpic_path = os.getenv(
            "ORGGPT_BPIC_PATH",
            _kaggle_path(kaggle_root, self.bpic_dataset_slug, self.bpic_filename),
        )
        self.secret_masker = SecretMasker.from_environment("GEMINI_API_KEY", "WANDB_API_KEY")
        self.secret_masker.register(self.gemini_api_key)
        self.secret_masker.register(self.wandb_api_key)

    def require_gemini_api_key(self) -> str:
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        return self.gemini_api_key

    @property
    def DPO_CONFIG(self) -> Dict[str, Any]:
        return DPO_CONFIG

    @property
    def DPO_V11_CONFIG(self) -> Dict[str, Any]:
        return DPO_V11_CONFIG

    @property
    def DPO_V12_CONFIG(self) -> Dict[str, Any]:
        return DPO_V12_CONFIG

    @property
    def DPO_V13_CONFIG(self) -> Dict[str, Any]:
        return DPO_V13_CONFIG


CONFIG = OrgGPTConfig()

# ==========================================
# 14B QLoRA DPO Hyperparameter Registry (Git SSoT)
# ==========================================
DPO_CONFIG = {
    "lr": 5e-5,
    "beta": 0.1,
    "max_len": 2048,
    "batch_size": 4,
}

DPO_V11_CONFIG = {
    "lr": 3e-5,
    "beta": 0.05,
    "max_len": 2048,
    "batch_size": 4,
}

DPO_V12_CONFIG = {
    "lr": 3e-5,
    "beta": 0.05,
    "max_len": 2048,
    "batch_size": 4,
}

DPO_V13_CONFIG = {
    "lr": 2e-5,
    "beta": 0.05,
    "max_len": 2048,
    "batch_size": 4,
}

# ==========================================
# Experiment Matrix Registry (Git SSoT)
# ==========================================
EXPERIMENT_MATRIX = {
    "orggpt_enron_baseline": {
        "dataset": "enron",
        "method": "baseline",
        "hierarchy_depth": 3,
        "blue_team_risk_aversion": 0.1,
        "red_team_strictness": 0.0,
        "desc": "Baseline semantic drift without Red Team Constitutional Auditor",
    },
    "orggpt_enron_audited": {
        "dataset": "enron",
        "method": "constitutional",
        "hierarchy_depth": 3,
        "blue_team_risk_aversion": 0.5,
        "red_team_strictness": 1.0,
        "desc": "Semantic drift controlled by Gemini Red Team",
    },
    "orggpt_bpic_baseline": {
        "dataset": "bpic",
        "method": "baseline",
        "limit": 5000,
        "red_team_strictness": 0.0,
        "desc": "Workflow routing delay baseline on BPIC logs",
    },
    "orggpt_bpic_audited": {
        "dataset": "bpic",
        "method": "constitutional",
        "limit": 5000,
        "red_team_strictness": 1.0,
        "desc": "Workflow routing delay controlled by institutional constraints on BPIC logs",
    },
}

CLAIM_TO_EXPERIMENT_MAP = {
    "Table 1: Semantic Entropy Decay ($H_\\ell$) is reduced by Red Team": [
        "orggpt_enron_baseline",
        "orggpt_enron_audited",
    ],
    "Figure 2: Workflow routing delays under Institutional Constraints": [
        "orggpt_bpic_baseline",
        "orggpt_bpic_audited",
    ],
}
