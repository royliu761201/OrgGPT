from __future__ import annotations
import os
import operator
import sys

# 0. Emergency Dependency pulse (Industrial-Grade Pulse)
if '/kaggle/' in os.getcwd() or '/kaggle/' in (__file__ or ''):
    print("--- 📦 Building Remote Dependencies (Industrial-Grade Pulse) ---")
    os.system("pip install -q google-genai numpy scipy scikit-learn")

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Optional

import google.genai as genai
from google.genai import types as genai_types
import numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================
# 1. ORGGPT CORE CONFIG & SECURITY (INLINED)
# ==========================================

class SecretMasker:
    def __init__(self, secrets: Optional[Iterable[Optional[str]]] = None):
        self._secrets: List[str] = []
        if secrets:
            for secret in secrets:
                self.register(secret)

    @classmethod
    def from_environment(cls, *env_vars: str) -> "SecretMasker":
        return cls(os.environ.get(env_var) for env_var in env_vars)

    def register(self, secret: Optional[str]) -> None:
        if not secret: return
        if secret in self._secrets: return
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
    device: str = "cpu"
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("GEMINI_API_KEY"))
    wandb_api_key: Optional[str] = field(default_factory=lambda: os.environ.get("WANDB_API_KEY"))
    
    blue_team_model: str = "gemini-3-flash-preview"
    red_team_model: str = "gemini-3-flash-preview"
    
    kaggle_input_root: str = field(default_factory=lambda: os.getenv("KAGGLE_INPUT_ROOT", "/kaggle/input"))
    enron_dataset_slug: str = "enron-email-dataset"
    enron_filename: str = "emails.csv"
    
    def _get_secret(self, key: str) -> Optional[str]:
        val = os.environ.get(key)
        if val: return val
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret(key)
        except: return None

    def __post_init__(self) -> None:
        if not self.gemini_api_key: self.gemini_api_key = self._get_secret("GEMINI_API_KEY")
        if not self.wandb_api_key: self.wandb_api_key = self._get_secret("WANDB_API_KEY")
        self.secret_masker = SecretMasker.from_environment("GEMINI_API_KEY", "WANDB_API_KEY")
        self.secret_masker.register(self.gemini_api_key)
        self.secret_masker.register(self.wandb_api_key)
        
        root = self.kaggle_input_root.rstrip("/")
        self.enron_path = os.getenv("ORGGPT_ENRON_PATH", _kaggle_path(root, self.enron_dataset_slug, self.enron_filename))
        if not os.path.exists(self.enron_path):
            potential = list(Path(root).glob(f"**/{self.enron_filename}"))
            if potential: self.enron_path = str(potential[0])

    def require_gemini_api_key(self) -> str:
        if not self.gemini_api_key:
            err_msg = (
                "\n❌ CRITICAL: GEMINI_API_KEY is missing in the Kaggle environment!\n"
                "👉 INDUSTRIAL-GRADE FIX:\n"
                "1. Go to Kaggle Kernel Editor (Web UI)\n"
                "2. Click 'Add-ons' -> 'Secrets'\n"
                "3. Use '[Add a new secret]' with Label: 'GEMINI_API_KEY' and Value: [Your Key]\n"
                "4. Ensure context checkbox is CHECKED for this notebook.\n"
                "5. Restart the surge.\n"
            )
            print(err_msg)
            raise ValueError("GEMINI_API_KEY is not set.")
        return self.gemini_api_key

CONFIG = OrgGPTConfig()
GEMINI_GENERATE_TIMEOUT_SECONDS = 30
GEMINI_GENERATE_TIMEOUT_MS = GEMINI_GENERATE_TIMEOUT_SECONDS * 1000


def _require_positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field_name} must be an integer, got bool.")
    try:
        integer_value = operator.index(value)
    except TypeError as exc:
        raise TypeError(
            f"{field_name} must be an integer, got {type(value).__name__}."
        ) from exc
    if integer_value < 1:
        raise ValueError(f"{field_name} must be at least 1, got {integer_value}.")
    return integer_value

# ==========================================
class IEDM:
    def __init__(self, vectorizer, smoothing=1e-12):
        self.vectorizer = vectorizer
        self.smoothing = smoothing

    def _to_probability_matrix(self, matrix: csr_matrix) -> np.ndarray:
        dense = np.asarray(matrix.toarray(), dtype=np.float64)
        dense = np.atleast_2d(np.clip(dense, a_min=0.0, a_max=None))
        smoothed = dense + self.smoothing
        row_sums = np.asarray(smoothed.sum(axis=1, keepdims=True), dtype=np.float64)
        return smoothed / row_sums

    def compute_distortion(self, v_root: str, V_level: Iterable[str]) -> float:
        child_texts = np.asarray([str(v) for v in V_level], dtype=object)
        if child_texts.size == 0:
            return 0.0

        root_text = np.asarray([str(v_root)], dtype=object)
        corpus = np.concatenate((root_text, child_texts))
        matrix = self.vectorizer.transform(corpus.tolist()).tocsr()
        distributions = np.asarray(self._to_probability_matrix(matrix), dtype=np.float64)
        if distributions.shape[0] < 2:
            return 0.0

        root_distribution = np.asarray(distributions[0], dtype=np.float64)
        child_distributions = np.atleast_2d(
            np.asarray(distributions[1:], dtype=np.float64)
        )
        root_distributions = np.broadcast_to(root_distribution, child_distributions.shape)
        js_distances = np.asarray(
            jensenshannon(root_distributions, child_distributions, axis=1, base=2.0),
            dtype=np.float64,
        )
        return float(np.mean(np.square(js_distances))) if js_distances.size else 0.0

# ==========================================
# 3. ASYNC SIMULATION HARNESS
# ==========================================

@dataclass
class TrialResult:
    trial_id: int
    draft: str
    distortion: Optional[float]
    is_compliant: bool

class AsyncSimulationRunner:
    def __init__(
        self,
        iedm: IEDM,
        base_policy: str,
        constitution: list[str],
        max_concurrency: int = 5,
    ):
        self.iedm = iedm
        self.base_policy = base_policy
        self.constitution = constitution
        self.max_concurrency = _require_positive_int(
            max_concurrency, field_name="max_concurrency"
        )
        self._semaphore = asyncio.Semaphore(self.max_concurrency)
        self._client = genai.Client(
            api_key=CONFIG.require_gemini_api_key(),
            http_options=genai_types.HttpOptions(timeout=GEMINI_GENERATE_TIMEOUT_MS),
        )
        self._aio_client = self._client.aio

    async def _generate(self, model: str, prompt: str) -> str:
        async with self._semaphore:
            try:
                res = await asyncio.wait_for(
                    self._aio_client.models.generate_content(
                        model=model,
                        contents=prompt,
                    ),
                    timeout=GEMINI_GENERATE_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"Gemini generate_content timed out after "
                    f"{GEMINI_GENERATE_TIMEOUT_SECONDS} seconds."
                ) from exc
            return res.text or ""

    async def run_trial(self, trial_id: int, audited: bool) -> TrialResult:
        prompt = f"Role: VP. Rephrase policy for subordinates: {self.base_policy}"
        try:
            draft = await self._generate(CONFIG.blue_team_model, prompt)
            is_compliant = True
            if audited:
                aud_prompt = f"Rules: {self.constitution}\nAction: {draft}\nReturn JSON strictly: {{\"is_compliant\": true|false}}"
                aud_res = await self._generate(CONFIG.red_team_model, aud_prompt)
                try:
                    cleaned = aud_res.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                    is_compliant = json.loads(cleaned).get("is_compliant", False)
                except: is_compliant = False
            # V2.4: Use global IEDM instance (no re-fitting)
            dist = self.iedm.compute_distortion(self.base_policy, [draft]) if is_compliant else None
            return TrialResult(trial_id, draft, dist, is_compliant)
        except Exception as e:
            print(f"Error during trial {trial_id}: {e}")
            return TrialResult(trial_id, "", None, False)

    async def run_surge(self, num_trials: int, audited: bool):
        tasks = [self.run_trial(i, audited) for i in range(num_trials)]
        return await asyncio.gather(*tasks)

async def main_surge(num_trials: int = 50):
    print("--- 🛰️  V2.4 ENRON OPTIMIZER IGNITION ---")
    
    # 1. Establish Global Context (Industrial-Grade Pulse)
    import pandas as pd
    print(f"--- 📦 Loading Enron Reference Space: {CONFIG.enron_path} ---")
    try:
        df_sample = pd.read_csv(CONFIG.enron_path, nrows=20000).sample(5000)
        global_vec = TfidfVectorizer(max_features=2000, stop_words='english')
        global_vec.fit(df_sample['message'].fillna(""))
        iedm = IEDM(vectorizer=global_vec)
        print("✅ Global Semantic Context: ESTABLISHED")
    except Exception as e:
        print(f"⚠️ Warning: Enron contextual fit failed ({e}). Falling back to local space.")
        local_vec = TfidfVectorizer(max_features=2000, stop_words='english')
        local_vec.fit(["Policy context placeholder"]) # Minimal fallback
        iedm = IEDM(vectorizer=local_vec)

    base_policy = "Mandate a 20% reduction in cloud computing expenditures."
    constitution = ["Do not implement cuts that disrupt revenue services."]
    runner = AsyncSimulationRunner(iedm, base_policy, constitution)
    
    try:
        baseline, audited = await asyncio.gather(
            runner.run_surge(num_trials, False),
            runner.run_surge(num_trials, True)
        )
    finally:
        await runner._aio_client.aclose()

    b_drifts = [r.distortion for r in baseline if r.distortion is not None]
    a_drifts = [r.distortion for r in audited if r.distortion is not None]
    
    results = {
        "baseline_mean": float(np.mean(b_drifts)) if b_drifts else 0.0,
        "audited_mean": float(np.mean(a_drifts)) if a_drifts else 0.0,
        "compliance_rate": float(len(a_drifts)/num_trials)
    }

    results_path = Path("/kaggle/working/results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(results, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    
    print(f"\nFINAL_RESULTS_JSON: {json.dumps(results)}")
    print(f"FINAL_RESULTS_PATH: {results_path}")
    print(f"FINAL - Baseline Drift: {results['baseline_mean']:.4f}")
    print(f"FINAL - Audited Drift: {results['audited_mean']:.4f}")
    print(f"FINAL - Compliance: {results['compliance_rate']*100:.1f}%")

if __name__ == "__main__":
    asyncio.run(main_surge(num_trials=50))
