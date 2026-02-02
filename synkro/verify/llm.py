"""LLM helpers for verification routing and judging.

Uses Synkro's LLM class for multi-provider support (OpenAI, Anthropic, Google, Cerebras, etc.)
"""

import json

from synkro.llm import LLM
from synkro.models import Model
from synkro.utils.model_detection import get_default_models

ROUTE_PROMPT = """\
You are a verification router. Given an AI response and a verifier, \
decide if the verifier is relevant and extract the required parameters.

Only trigger if the response makes a concrete, checkable factual claim \
matching the verifier's intent.

Return ONLY valid JSON (no markdown, no extra text):
{"trigger": true, "params": {"param_name": "extracted_value"}}
or
{"trigger": false, "params": {}}"""

VERIFY_PROMPT = """\
You are a factual accuracy verifier. You receive:
1. An AI response (the trace)
2. Ground truth data from a database

Compare the AI response against the database.

Return ONLY valid JSON (no markdown, no extra text):
{"verdict": "pass", "confidence": 0.95, "reasoning": "explanation"}
or
{"verdict": "violation", "confidence": 0.95, "reasoning": "explanation"}

Guidelines:
- verdict must be exactly "pass" or "violation"
- confidence is a float from 0.0 to 1.0
- violation = AI response contradicts the database
- pass = AI response is consistent with the database
- Vague but not wrong (e.g. "about 45" vs DB 47) = pass"""


def _parse_json(text: str) -> dict:
    """Parse JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json) and last line (```)
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


class VerifierLLM:
    """LLM client for verification routing and judging.

    Uses Synkro's LLM class for multi-provider support.
    Auto-detects available provider if models not specified.
    """

    def __init__(
        self,
        route_model: Model | str | None = None,
        judge_model: Model | str | None = None,
    ):
        """Initialize with optional models.

        Args:
            route_model: Model for routing (fast, cheap). Auto-detected if not provided.
            judge_model: Model for judging (accurate). Auto-detected if not provided.
        """
        # Auto-detect models if not specified
        if route_model is None or judge_model is None:
            default_gen, default_grade = get_default_models()
            if route_model is None:
                route_model = default_gen
            if judge_model is None:
                judge_model = default_grade

        self.route_model = route_model
        self.judge_model = judge_model

        # Create LLM instances
        self._route_llm = LLM(model=route_model, temperature=0)
        self._judge_llm = LLM(model=judge_model, temperature=0)

    @property
    def route_model_name(self) -> str:
        """Get the route model name."""
        return self._route_llm.model

    @property
    def judge_model_name(self) -> str:
        """Get the judge model name."""
        return self._judge_llm.model

    async def route(self, response_text: str, verifier_desc: dict) -> dict:
        """Decide if verifier is relevant and extract params.

        Args:
            response_text: The AI response to check
            verifier_desc: Dict with 'intent' and 'params' keys

        Returns:
            {"trigger": bool, "params": {...}} or {"trigger": False}
        """
        user_msg = (
            f"AI Response:\n{response_text}\n\n" f"Verifier:\n{json.dumps(verifier_desc, indent=2)}"
        )
        prompt = f"{ROUTE_PROMPT}\n\n{user_msg}"

        result = await self._route_llm.generate(prompt)
        return _parse_json(result)

    async def judge(self, response_text: str, evidence: dict) -> dict:
        """Judge if response matches ground truth.

        Args:
            response_text: The AI response to verify
            evidence: Dict with ground truth data from database

        Returns:
            {"verdict": "pass"|"violation", "confidence": float, "reasoning": str}
        """
        user_msg = (
            f"AI Response (trace):\n{response_text}\n\n"
            f"Ground truth:\n{json.dumps(evidence, default=str, indent=2)}"
        )
        prompt = f"{VERIFY_PROMPT}\n\n{user_msg}"

        result = await self._judge_llm.generate(prompt)
        return _parse_json(result)
