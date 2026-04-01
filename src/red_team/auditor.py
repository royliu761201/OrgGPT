from typing import List, Dict, Any
from .evaluator import GeminiComplianceJudge

FORBIDDEN_KNOWLEDGE_KEYS = {
    "directive",
    "target_directive",
    "root_policy",
    "original_policy",
    "reference_policy",
}

class ConstitutionalAuditor:
    """
    Red Team: Institutional Hard-rules and Auditor.
    """
    def __init__(self, ruleset: Dict[str, Any]):
        forbidden_keys = FORBIDDEN_KNOWLEDGE_KEYS.intersection(ruleset)
        if forbidden_keys:
            raise ValueError(
                f"ConstitutionalAuditor must remain blind to the target directive; remove {sorted(forbidden_keys)}."
            )
        self.rules = ruleset.get("constitution", [])
        self.strictness = ruleset.get("red_team_strictness", 1.0)
        self.judge = GeminiComplianceJudge() if self.strictness > 0 else None

    def get_legal_actions(self, candidate_actions: List[str]) -> List[str]:
        """
        Uses a blind constitutional judge to filter candidate actions strictly.
        """
        if self.strictness == 0.0 or not self.judge:
            return candidate_actions

        legal_actions = []
        for action in candidate_actions:
            is_compliant = self.judge.evaluate_compliance(action, self.rules)
            if is_compliant:
                legal_actions.append(action)
        return legal_actions

    async def get_legal_actions_async(self, candidate_actions: List[str]) -> List[str]:
        if self.strictness == 0.0 or not self.judge:
            return candidate_actions

        import asyncio
        tasks = [self.judge.evaluate_compliance_async(action, self.rules) for action in candidate_actions]
        results = await asyncio.gather(*tasks)
        return [action for action, is_compliant in zip(candidate_actions, results) if is_compliant]
