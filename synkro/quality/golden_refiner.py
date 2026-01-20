"""Golden Refiner - Refines traces that failed verification.

Refines traces with Logic Map context to fix:
- Skipped rules
- Hallucinated rules
- Contradictions
- DAG violations
"""

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.prompts.golden_templates import GOLDEN_REFINE_PROMPT
from synkro.schemas import GoldenTraceOutput
from synkro.types.core import Message, Trace
from synkro.types.logic_map import GoldenScenario, LogicMap, VerificationResult


class GoldenRefiner:
    """
    Refiner that uses Logic Map context to fix verification failures.

    Addresses specific issues:
    1. Skipped Rules: Adds evaluation of missed rules
    2. Hallucinated Rules: Removes references to non-existent rules
    3. Contradictions: Resolves logical inconsistencies
    4. DAG Violations: Reorders reasoning to follow dependencies

    Examples:
        >>> refiner = GoldenRefiner(llm=LLM(model=OpenAI.GPT_4O_MINI))
        >>> refined = await refiner.refine(trace, logic_map, scenario, verification)
    """

    def __init__(
        self,
        llm: LLM | None = None,
        model: Model = OpenAI.GPT_4O_MINI,
    ):
        """
        Initialize the Golden Refiner.

        Args:
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM
        """
        self.llm = llm or LLM(model=model, temperature=0.5)

    async def refine(
        self,
        trace: Trace,
        logic_map: LogicMap,
        scenario: GoldenScenario,
        verification: VerificationResult,
    ) -> Trace:
        """
        Refine a trace that failed verification.

        Args:
            trace: The original trace that failed
            logic_map: The Logic Map (ground truth)
            scenario: The golden scenario
            verification: The verification result with issues

        Returns:
            Refined trace with issues addressed
        """
        # Format inputs for prompt
        logic_map_str = self._format_logic_map(logic_map)
        original_trace_str = self._format_trace(trace)
        verification_str = self._format_verification(verification)

        # Build prompt
        prompt = GOLDEN_REFINE_PROMPT.format(
            original_trace=original_trace_str,
            verification_result=verification_str,
            logic_map=logic_map_str,
            scenario_description=scenario.description,
            skipped_rules=", ".join(verification.skipped_rules)
            if verification.skipped_rules
            else "None",
            hallucinated_rules=", ".join(verification.hallucinated_rules)
            if verification.hallucinated_rules
            else "None",
            contradictions="; ".join(verification.contradictions)
            if verification.contradictions
            else "None",
        )

        # Generate refined trace
        result = await self.llm.generate_structured(prompt, GoldenTraceOutput)

        # Convert to Trace
        messages = [Message(role=m.role, content=m.content) for m in result.messages]

        # Preserve scenario from original trace
        return Trace(
            messages=messages,
            scenario=trace.scenario,
        )

    def _format_logic_map(self, logic_map: LogicMap) -> str:
        """Format Logic Map for refinement prompt."""
        lines = []
        lines.append("RULES:")
        for rule in logic_map.rules:
            deps = f" [depends on: {', '.join(rule.dependencies)}]" if rule.dependencies else ""
            lines.append(f"  {rule.rule_id} ({rule.category.value}): {rule.text}{deps}")
            lines.append(f"    IF: {rule.condition}")
            lines.append(f"    THEN: {rule.action}")

        lines.append("\nDEPENDENCY ORDER:")
        for root_id in logic_map.root_rules:
            chain = logic_map.get_chain(root_id)
            if chain:
                chain_str = " -> ".join(r.rule_id for r in chain)
                lines.append(f"  {chain_str}")

        return "\n".join(lines)

    def _format_trace(self, trace: Trace) -> str:
        """Format trace for refinement prompt."""
        lines = []
        for msg in trace.messages:
            role = msg.role.upper()
            content = msg.content or "(no content)"

            # Handle tool calls
            if msg.tool_calls:
                tool_info = []
                for tc in msg.tool_calls:
                    if hasattr(tc, "function"):
                        tool_info.append(f"  - {tc.function.name}({tc.function.arguments})")
                    elif isinstance(tc, dict):
                        func = tc.get("function", {})
                        tool_info.append(
                            f"  - {func.get('name', 'unknown')}({func.get('arguments', '{}')})"
                        )
                content = "Tool calls:\n" + "\n".join(tool_info)

            lines.append(f"[{role}]: {content}")

        return "\n\n".join(lines)

    def _format_verification(self, verification: VerificationResult) -> str:
        """Format verification result for refinement prompt."""
        lines = []
        lines.append(f"Passed: {verification.passed}")

        if verification.issues:
            lines.append(f"Issues: {'; '.join(verification.issues)}")

        if verification.skipped_rules:
            lines.append(f"Skipped Rules: {', '.join(verification.skipped_rules)}")

        if verification.hallucinated_rules:
            lines.append(f"Hallucinated Rules: {', '.join(verification.hallucinated_rules)}")

        if verification.contradictions:
            lines.append(f"Contradictions: {'; '.join(verification.contradictions)}")

        if verification.rules_verified:
            lines.append(f"Rules Verified: {', '.join(verification.rules_verified)}")

        return "\n".join(lines)


__all__ = ["GoldenRefiner"]
