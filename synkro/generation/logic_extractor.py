"""Logic Extractor - The Cartographer.

Extracts a Logic Map (DAG of rules) from a policy document.
This is Stage 1 of the Golden Trace pipeline.
"""

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.prompts.golden_templates import LOGIC_EXTRACTION_PROMPT
from synkro.schemas import LogicMapOutput
from synkro.types.logic_map import LogicMap, Rule, RuleCategory


class LogicExtractor:
    """
    The Cartographer - Extracts a Logic Map from policy documents.

    The Logic Map is a Directed Acyclic Graph (DAG) of rules that enables:
    - Grounded scenario generation with rule references
    - Chain-of-thought reasoning with rule citations
    - Verification that traces don't skip or hallucinate rules

    Examples:
        >>> extractor = LogicExtractor(llm=LLM(model=OpenAI.GPT_4O))
        >>> logic_map = await extractor.extract(policy_text)
        >>> print(f"Extracted {len(logic_map.rules)} rules")
    """

    def __init__(
        self,
        llm: LLM | None = None,
        model: Model = OpenAI.GPT_4O,
    ):
        """
        Initialize the Logic Extractor.

        Args:
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM (default: GPT-4O for accuracy)
        """
        self.llm = llm or LLM(model=model, temperature=0.3)

    async def extract(self, policy_text: str) -> LogicMap:
        """
        Extract a Logic Map from a policy document.

        Args:
            policy_text: The policy document text

        Returns:
            LogicMap with extracted rules as a DAG

        Raises:
            ValueError: If extraction fails or produces invalid DAG
        """
        # Format the prompt
        prompt = LOGIC_EXTRACTION_PROMPT.format(policy_text=policy_text)

        # Generate structured output
        result = await self.llm.generate_structured(prompt, LogicMapOutput)

        # Convert to domain model
        logic_map = self._convert_to_logic_map(result)

        # Validate DAG properties
        if not logic_map.validate_dag():
            raise ValueError("Extracted rules contain circular dependencies")

        return logic_map

    def _convert_to_logic_map(self, output: LogicMapOutput) -> LogicMap:
        """Convert schema output to domain model."""
        rules = []
        for rule_out in output.rules:
            # Convert category string to enum
            category = RuleCategory(rule_out.category)

            rule = Rule(
                rule_id=rule_out.rule_id,
                text=rule_out.text,
                condition=rule_out.condition,
                action=rule_out.action,
                dependencies=rule_out.dependencies,
                category=category,
            )
            rules.append(rule)

        return LogicMap(
            rules=rules,
            root_rules=output.root_rules,
        )

    async def extract_with_retry(
        self,
        policy_text: str,
        max_retries: int = 2,
    ) -> LogicMap:
        """
        Extract with retry on validation failure.

        Args:
            policy_text: The policy document text
            max_retries: Maximum retry attempts

        Returns:
            LogicMap with extracted rules

        Raises:
            ValueError: If extraction fails after all retries
        """
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return await self.extract(policy_text)
            except ValueError as e:
                last_error = e
                if attempt < max_retries:
                    # Add hint for next attempt
                    continue

        raise ValueError(
            f"Failed to extract valid Logic Map after {max_retries + 1} attempts: {last_error}"
        )


__all__ = ["LogicExtractor"]
