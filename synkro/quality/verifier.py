"""Trace Verifier - The Auditor.

Verifies generated traces against the Logic Map to ensure:
- No skipped rules
- No hallucinated rules
- No contradictions
- DAG compliance

This is Stage 4 of the Golden Trace pipeline.
"""

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.prompts.golden_templates import VERIFICATION_PROMPT
from synkro.schemas import VerificationOutput
from synkro.types.core import GradeResult, Trace
from synkro.types.logic_map import GoldenScenario, LogicMap, VerificationResult


class TraceVerifier:
    """
    The Auditor - Verifies traces against the Logic Map.

    Performs strict verification to ensure:
    1. No Skipped Rules: All target rules were evaluated
    2. No Hallucinated Rules: Only valid rules were cited
    3. No Contradictions: Reasoning is internally consistent
    4. DAG Compliance: Dependency order was followed
    5. Outcome Alignment: Response matches expected outcome

    Examples:
        >>> verifier = TraceVerifier(llm=LLM(model=OpenAI.GPT_4O))
        >>> result = await verifier.verify(trace, logic_map, scenario)
        >>> if result.passed:
        ...     print("Trace verified successfully")
    """

    def __init__(
        self,
        llm: LLM | None = None,
        model: Model = OpenAI.GPT_4O,
    ):
        """
        Initialize the Trace Verifier.

        Args:
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM (default: GPT-4O for accuracy)
        """
        self.llm = llm or LLM(model=model, temperature=0.1)

    async def verify(
        self,
        trace: Trace,
        logic_map: LogicMap,
        scenario: GoldenScenario,
        reasoning_chain: list | None = None,
        rules_applied: list[str] | None = None,
        rules_excluded: list[str] | None = None,
    ) -> VerificationResult:
        """
        Verify a trace against the Logic Map.

        Args:
            trace: The trace to verify
            logic_map: The Logic Map (ground truth)
            scenario: The golden scenario
            reasoning_chain: Optional reasoning chain from generation
            rules_applied: Optional list of rules claimed applied
            rules_excluded: Optional list of rules claimed excluded

        Returns:
            VerificationResult with pass/fail and detailed issues
        """
        # Format inputs for prompt
        logic_map_str = self._format_logic_map(logic_map)
        trace_messages_str = self._format_trace_messages(trace)
        reasoning_str = (
            self._format_reasoning_chain(reasoning_chain) if reasoning_chain else "Not provided"
        )

        # Build prompt
        prompt = VERIFICATION_PROMPT.format(
            logic_map=logic_map_str,
            scenario_type=scenario.scenario_type.value.upper(),
            scenario_description=scenario.description,
            target_rule_ids=", ".join(scenario.target_rule_ids),
            expected_outcome=scenario.expected_outcome,
            trace_messages=trace_messages_str,
            reasoning_chain=reasoning_str,
            rules_applied=", ".join(rules_applied) if rules_applied else "Not specified",
            rules_excluded=", ".join(rules_excluded) if rules_excluded else "Not specified",
        )

        # Generate structured output
        result = await self.llm.generate_structured(prompt, VerificationOutput)

        # Convert to domain model
        return VerificationResult(
            passed=result.passed,
            issues=result.issues,
            skipped_rules=result.skipped_rules,
            hallucinated_rules=result.hallucinated_rules,
            contradictions=result.contradictions,
            rules_verified=result.rules_verified,
        )

    def _format_logic_map(self, logic_map: LogicMap) -> str:
        """Format Logic Map for verification prompt."""
        lines = []
        lines.append("RULES:")
        for rule in logic_map.rules:
            deps = f" [depends on: {', '.join(rule.dependencies)}]" if rule.dependencies else ""
            lines.append(f"  {rule.rule_id} ({rule.category.value}): {rule.text}{deps}")
            lines.append(f"    IF: {rule.condition}")
            lines.append(f"    THEN: {rule.action}")

        lines.append("\nROOT RULES (Entry Points):")
        lines.append(f"  {', '.join(logic_map.root_rules)}")

        return "\n".join(lines)

    def _format_trace_messages(self, trace: Trace) -> str:
        """Format trace messages for verification prompt."""
        lines = []
        for i, msg in enumerate(trace.messages):
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

            # Handle tool responses
            if msg.tool_call_id:
                role = f"TOOL (call_id: {msg.tool_call_id})"

            lines.append(f"[{role}] {content}")

        return "\n\n".join(lines)

    def _format_reasoning_chain(self, reasoning_chain: list) -> str:
        """Format reasoning chain for verification prompt."""
        lines = []
        for i, step in enumerate(reasoning_chain, 1):
            if isinstance(step, str):
                # Handle string format (raw LLM output)
                lines.append(f"Step {i}: {step}")
            elif hasattr(step, "rule_id"):
                applies = "APPLIES" if step.applies else "DOES NOT APPLY"
                lines.append(f"Step {i}: {step.rule_id} - {applies}")
                lines.append(f"  Rule: {step.rule_text}")
                lines.append(f"  Reasoning: {step.reasoning}")
                if step.exclusions:
                    lines.append(f"  Excludes: {', '.join(step.exclusions)}")
            else:
                # Handle dict format
                applies = "APPLIES" if step.get("applies", False) else "DOES NOT APPLY"
                lines.append(f"Step {i}: {step.get('rule_id', 'unknown')} - {applies}")
                lines.append(f"  Reasoning: {step.get('reasoning', 'N/A')}")

        return "\n".join(lines)

    async def verify_and_grade(
        self,
        trace: Trace,
        logic_map: LogicMap,
        scenario: GoldenScenario,
    ) -> tuple[VerificationResult, GradeResult]:
        """
        Verify a trace and convert to GradeResult for pipeline compatibility.

        Args:
            trace: The trace to verify
            logic_map: The Logic Map
            scenario: The golden scenario

        Returns:
            Tuple of (VerificationResult, GradeResult)
        """
        # Extract reasoning chain metadata from trace (if present)
        reasoning_chain = getattr(trace, "reasoning_chain", None)
        rules_applied = getattr(trace, "rules_applied", None)
        rules_excluded = getattr(trace, "rules_excluded", None)

        verification = await self.verify(
            trace,
            logic_map,
            scenario,
            reasoning_chain=reasoning_chain,
            rules_applied=rules_applied,
            rules_excluded=rules_excluded,
        )

        # Convert to GradeResult for pipeline compatibility
        grade = GradeResult(
            passed=verification.passed,
            issues=verification.issues,
            feedback=self._create_feedback(verification),
        )

        return verification, grade

    def _create_feedback(self, verification: VerificationResult) -> str:
        """Create feedback string from verification result."""
        if verification.passed:
            return f"Verified. Rules correctly applied: {', '.join(verification.rules_verified)}"

        feedback_parts = []

        if verification.skipped_rules:
            feedback_parts.append(f"Skipped rules: {', '.join(verification.skipped_rules)}")

        if verification.hallucinated_rules:
            feedback_parts.append(
                f"Hallucinated rules: {', '.join(verification.hallucinated_rules)}"
            )

        if verification.contradictions:
            feedback_parts.append(f"Contradictions: {'; '.join(verification.contradictions)}")

        if verification.issues:
            feedback_parts.append(f"Other issues: {'; '.join(verification.issues)}")

        return " | ".join(feedback_parts) if feedback_parts else "Verification failed"


__all__ = ["TraceVerifier"]
