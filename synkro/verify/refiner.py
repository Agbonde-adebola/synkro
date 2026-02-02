"""Refinement of traces that failed SQL verification."""

import json

from synkro.llm.client import LLM
from synkro.verify.types import Report, Verdict

REFINE_PROMPT = """\
You are fixing factual errors in an AI response.

The response made claims that contradict the database. Fix ONLY the incorrect facts.
Keep everything else (tone, structure, reasoning) the same.

ORIGINAL RESPONSE:
{response}

ERRORS FOUND:
{errors}

CORRECT DATA FROM DATABASE:
{ground_truth}

Generate a corrected response that:
1. Replaces incorrect facts with correct ones from the database
2. Keeps all other content unchanged
3. Maintains the same tone and structure

Respond with ONLY the corrected text, no JSON or markdown."""


class SQLRefiner:
    """Refines traces that failed SQL verification.

    Takes violations and regenerates responses with correct facts from the database.

    Example:
        refiner = SQLRefiner()
        fixed = await refiner.refine(messages, report)
        # or
        fixed_batch = await refiner.refine_batch(all_messages, all_reports)
    """

    def __init__(self, llm: LLM | None = None):
        """Initialize refiner.

        Args:
            llm: LLM client (creates default if not provided)
        """
        self.llm = llm or LLM(temperature=0.3)

    async def refine(
        self,
        messages: list[dict],
        report: Report,
    ) -> list[dict]:
        """Refine a trace that failed SQL verification.

        Args:
            messages: Original OpenAI-format messages
            report: Verification report with violations

        Returns:
            New messages with corrected assistant response
        """
        violations = [r for r in report.results if r.value == Verdict.VIOLATION]

        if not violations:
            return messages

        # Build error summary from violations
        errors = []
        ground_truth = []

        for v in violations:
            errors.append(f"- {v.name}: {v.comment}")
            if v.metadata.get("ground_truth"):
                ground_truth.append(
                    f"{v.name}:\n{json.dumps(v.metadata['ground_truth'], indent=2, default=str)}"
                )

        prompt = REFINE_PROMPT.format(
            response=report.response,
            errors="\n".join(errors),
            ground_truth="\n\n".join(ground_truth) if ground_truth else "N/A",
        )

        corrected = await self.llm.generate(prompt)

        # Replace assistant message with corrected version
        new_messages = []
        for msg in messages:
            if msg.get("role") == "assistant":
                new_messages.append({"role": "assistant", "content": corrected.strip()})
            else:
                new_messages.append(msg)

        return new_messages

    async def refine_batch(
        self,
        all_messages: list[list[dict]],
        reports: list[Report],
    ) -> list[list[dict]]:
        """Refine multiple traces.

        Args:
            all_messages: List of message lists
            reports: Corresponding verification reports

        Returns:
            List of refined message lists
        """
        refined = []

        for messages, report in zip(all_messages, reports):
            if not report.passed:
                fixed = await self.refine(messages, report)
                refined.append(fixed)
            else:
                refined.append(messages)

        return refined
