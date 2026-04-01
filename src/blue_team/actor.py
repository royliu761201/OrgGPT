import os
import json
import google.genai as genai
from typing import List, Dict, Any

from ..config import CONFIG

class CognitiveActor:
    """
    Blue Team Actor: Boundedly rational LLM kernel (Gemini 3.1 Pro)
    """
    def __init__(self, role_profile: Dict[str, Any]):
        self.profile = role_profile
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment.")
        self.client = genai.Client(api_key=self.api_key)
        
        self.risk_aversion = role_profile.get("risk_aversion", 0.5)
        self.role_name = role_profile.get("role_name", "Employee")
        
    def decide(self, local_state: str, legal_actions_list: List[str]) -> str:
        """
        Takes the current local state (e.g., a received policy or directive)
        and explicitly chooses an action from the Red-Team's legal subset.
        """
        system_prompt = f"""You are a {self.role_name} in a large hierarchy.
Your risk aversion score is {self.risk_aversion} (0=reckless, 1=paralyzed safe).
You must review the current state and select exactly ONE action from the LEGAL ACTIONS list.
You must return your decision as a raw JSON object: {{"selected_action": "<exact_string_from_list>", "reasoning": "..."}}
"""

        user_prompt = f"""CURRENT STATE / DIRECTIVE:
{local_state}

LEGAL ACTIONS YOU CAN TAKE:
{json.dumps(legal_actions_list, indent=2)}

Make your decision:
"""
        prompt = f"{system_prompt}\n\n{user_prompt}"
        response = self.client.models.generate_content(
            model=CONFIG.blue_team_model,
            contents=prompt,
        )
        
        raw_text = response.text.strip().removeprefix('```json').removesuffix('```').strip()
        try:
            decision = json.loads(raw_text)
        except json.JSONDecodeError:
            raise ValueError(f"Blue Team LLM failed to return valid JSON: {raw_text}")
        
        selected = decision.get("selected_action")
        if selected not in legal_actions_list:
            raise ValueError(f"Blue Team hallucinated an illegal action: {selected}. Allowed: {legal_actions_list}")
        
        return selected

    def rephrase_policy(self, incoming_policy: str) -> str:
        """
        Simulates semantic distortion/drift when passing information down the tree.
        """
        instructions = f"""You are a {self.role_name}. 
Rewrite the following policy for your subordinates. Simplify and contextualize it,
but note that your risk aversion ({self.risk_aversion}) might cause you to omit dangerous details.
POLICY: {incoming_policy}
"""
        res = self.client.models.generate_content(
            model=CONFIG.blue_team_model,
            contents=instructions,
        )
        return res.text

    async def rephrase_policy_async(self, incoming_policy: str) -> str:
        instructions = f"""You are a {self.role_name}. 
Rewrite the following policy for your subordinates. Simplify and contextualize it.
POLICY: {incoming_policy}
"""
        res = await self.client.aio.models.generate_content(
            model=CONFIG.blue_team_model,
            contents=instructions,
        )
        return res.text
