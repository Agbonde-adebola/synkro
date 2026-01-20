"""Planning for trace generation across categories."""

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.prompts.templates import POLICY_COMPLEXITY_PROMPT, POLICY_PLANNING_PROMPT
from synkro.schemas import PolicyComplexity, PolicyPlan
from synkro.types.core import Category, Plan


class Planner:
    """
    Plans how to distribute trace generation across categories.

    The planner analyzes the policy and creates an optimal distribution
    of scenarios across different categories to ensure comprehensive
    coverage.

    Examples:
        >>> planner = Planner()
        >>> plan = await planner.plan(policy, target_traces=100)
        >>> for cat in plan.categories:
        ...     print(f"{cat.name}: {cat.traces} traces")
    """

    def __init__(self, llm: LLM | None = None, model: Model = OpenAI.GPT_4O):
        """
        Initialize the planner.

        Args:
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM
        """
        self.llm = llm or LLM(model=model)

    async def analyze_complexity(self, policy_text: str) -> PolicyComplexity:
        """
        Analyze policy complexity to determine optimal conversation turns.

        Uses the PolicyComplexity schema to assess:
        - Variable count (rules, conditions, exceptions)
        - Complexity level (simple, conditional, complex)
        - Recommended turns (1-6)

        Args:
            policy_text: The policy text to analyze

        Returns:
            PolicyComplexity with recommended turns and complexity level
        """
        prompt = f"""{POLICY_COMPLEXITY_PROMPT}

POLICY:
{policy_text}

Analyze the policy complexity and recommend conversation turns."""

        try:
            return await self.llm.generate_structured(prompt, PolicyComplexity)
        except Exception:
            # Default to simple single-turn
            return PolicyComplexity(
                variable_count=1,
                complexity_level="simple",
                recommended_turns=1,
                reasoning="Default - unable to analyze complexity",
            )

    async def plan(self, policy_text: str, target_traces: int, analyze_turns: bool = True) -> Plan:
        """
        Create a generation plan for the policy.

        Analyzes the policy and determines optimal category distribution
        and conversation turn count.

        Args:
            policy_text: The policy text to analyze
            target_traces: Target number of traces to generate
            analyze_turns: Whether to analyze policy for turn recommendation

        Returns:
            Plan object with categories, reasoning, and turn recommendations
        """
        prompt = f"""{POLICY_PLANNING_PROMPT}

POLICY:
{policy_text}

TARGET TRACES: {target_traces}

Analyze the policy and create a plan with categories for generating training data."""

        # Analyze complexity for turn recommendations
        complexity = None
        if analyze_turns:
            complexity = await self.analyze_complexity(policy_text)

        try:
            # Use structured output for reliable planning
            parsed = await self.llm.generate_structured(prompt, PolicyPlan)

            # Convert to typed objects
            categories = [
                Category(
                    name=c.name,
                    description=c.description,
                    count=c.traces,
                )
                for c in parsed.categories
            ]

            return Plan(
                categories=categories,
                reasoning=parsed.reasoning,
                recommended_turns=complexity.recommended_turns if complexity else 1,
                complexity_level=complexity.complexity_level if complexity else "simple",
            )
        except Exception:
            # Fallback plan
            third = target_traces // 3
            remainder = target_traces - (third * 3)
            return Plan(
                categories=[
                    Category(name="Happy Path", description="Clear success cases", count=third),
                    Category(name="Edge Cases", description="Ambiguous situations", count=third),
                    Category(
                        name="Violations",
                        description="Clear failure cases",
                        count=third + remainder,
                    ),
                ],
                reasoning="Default plan - unable to parse LLM response",
                recommended_turns=complexity.recommended_turns if complexity else 1,
                complexity_level=complexity.complexity_level if complexity else "simple",
            )
