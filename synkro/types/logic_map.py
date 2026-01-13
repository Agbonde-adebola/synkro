"""Logic Map types for Golden Trace generation.

The Logic Map represents a policy as a directed acyclic graph (DAG) of rules,
enabling grounded reasoning and verification of generated traces.
"""

import json
from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ScenarioType(str, Enum):
    """Types of scenarios for balanced dataset generation."""

    POSITIVE = "positive"      # Happy path - user meets all criteria
    NEGATIVE = "negative"      # Violation - user fails one criterion
    EDGE_CASE = "edge_case"    # Boundary - user at exact limit
    IRRELEVANT = "irrelevant"  # Not covered by policy


class RuleCategory(str, Enum):
    """Categories of rules extracted from policy."""

    CONSTRAINT = "constraint"      # Must/must not conditions
    PERMISSION = "permission"      # Allowed/can do
    PROCEDURE = "procedure"        # Step-by-step processes
    EXCEPTION = "exception"        # Special cases/overrides


class Rule(BaseModel):
    """
    A single rule extracted from the policy document.

    Rules form nodes in the Logic Map DAG, with dependencies
    indicating which rules must be evaluated first.

    Examples:
        >>> rule = Rule(
        ...     rule_id="R001",
        ...     text="Refunds are allowed within 30 days of purchase",
        ...     condition="purchase date is within 30 days",
        ...     action="allow refund",
        ...     dependencies=[],
        ...     category=RuleCategory.PERMISSION,
        ... )
    """

    rule_id: str = Field(
        description="Unique identifier (e.g., 'R001', 'R002')"
    )
    text: str = Field(
        description="Exact rule text from the policy"
    )
    condition: str = Field(
        description="The 'if' part - when this rule applies"
    )
    action: str = Field(
        description="The 'then' part - what happens when rule applies"
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Rule IDs that must be evaluated before this rule"
    )
    category: RuleCategory = Field(
        description="Type of rule (constraint, permission, procedure, exception)"
    )

    def __hash__(self) -> int:
        return hash(self.rule_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rule):
            return False
        return self.rule_id == other.rule_id


class LogicMap(BaseModel):
    """
    Directed Acyclic Graph (DAG) of rules extracted from a policy.

    The Logic Map is the "Map of Truth" that enables:
    - Grounded scenario generation with rule references
    - Chain-of-thought reasoning with rule citations
    - Verification that traces don't skip or hallucinate rules

    Examples:
        >>> logic_map = LogicMap(
        ...     rules=[rule1, rule2, rule3],
        ...     root_rules=["R001"],  # Entry points
        ... )
        >>> print(logic_map.get_rule("R001"))
    """

    rules: list[Rule] = Field(
        description="All rules extracted from the policy"
    )
    root_rules: list[str] = Field(
        default_factory=list,
        description="Rule IDs with no dependencies (entry points)"
    )

    def get_rule(self, rule_id: str) -> Rule | None:
        """Get a rule by its ID."""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None

    def get_dependents(self, rule_id: str) -> list[Rule]:
        """Get all rules that depend on the given rule."""
        return [r for r in self.rules if rule_id in r.dependencies]

    def get_dependencies(self, rule_id: str) -> list[Rule]:
        """Get all rules that the given rule depends on."""
        rule = self.get_rule(rule_id)
        if not rule:
            return []
        return [r for r in self.rules if r.rule_id in rule.dependencies]

    def get_chain(self, rule_id: str) -> list[Rule]:
        """
        Get the full dependency chain for a rule (topologically sorted).

        Returns all rules that must be evaluated before the given rule,
        in the order they should be evaluated.
        """
        visited = set()
        chain = []

        def visit(rid: str):
            if rid in visited:
                return
            visited.add(rid)
            rule = self.get_rule(rid)
            if rule:
                for dep_id in rule.dependencies:
                    visit(dep_id)
                chain.append(rule)

        visit(rule_id)
        return chain

    def validate_dag(self) -> bool:
        """Verify the rules form a valid DAG (no cycles)."""
        # Track visit state: 0=unvisited, 1=visiting, 2=visited
        state = {r.rule_id: 0 for r in self.rules}

        def has_cycle(rule_id: str) -> bool:
            if state.get(rule_id, 0) == 1:  # Currently visiting = cycle
                return True
            if state.get(rule_id, 0) == 2:  # Already visited = ok
                return False

            state[rule_id] = 1  # Mark as visiting
            rule = self.get_rule(rule_id)
            if rule:
                for dep_id in rule.dependencies:
                    if has_cycle(dep_id):
                        return True
            state[rule_id] = 2  # Mark as visited
            return False

        for rule in self.rules:
            if has_cycle(rule.rule_id):
                return False
        return True

    def get_rules_by_category(self, category: RuleCategory) -> list[Rule]:
        """Get all rules of a specific category."""
        return [r for r in self.rules if r.category == category]

    def to_display_string(self) -> str:
        """Generate a human-readable representation of the Logic Map."""
        lines = [f"Logic Map ({len(self.rules)} rules)"]
        lines.append("=" * 40)

        # Show root rules first
        lines.append("\nRoot Rules (Entry Points):")
        for rid in self.root_rules:
            rule = self.get_rule(rid)
            if rule:
                lines.append(f"  {rid}: {rule.text[:60]}...")

        # Show dependency chains
        lines.append("\nDependency Chains:")
        processed = set()
        for rule in self.rules:
            if rule.rule_id not in processed and rule.dependencies:
                chain = " -> ".join(r.rule_id for r in self.get_chain(rule.rule_id))
                lines.append(f"  {chain}")
                processed.update(r.rule_id for r in self.get_chain(rule.rule_id))

        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """
        Save the Logic Map to a JSON file.

        Args:
            path: File path to save to (e.g., "logic_map.json")

        Examples:
            >>> logic_map.save("logic_map.json")
            >>> # Later, reload it
            >>> logic_map = LogicMap.load("logic_map.json")
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "LogicMap":
        """
        Load a Logic Map from a JSON file.

        Args:
            path: File path to load from

        Returns:
            LogicMap instance

        Examples:
            >>> logic_map = LogicMap.load("logic_map.json")
            >>> print(f"Loaded {len(logic_map.rules)} rules")
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)


class ReasoningStep(BaseModel):
    """
    A single step in the Chain-of-Thought reasoning.

    Each step references exactly one rule and explains how it applies
    (or doesn't apply) to the current scenario.
    """

    rule_id: str = Field(
        description="The rule being evaluated in this step"
    )
    rule_text: str = Field(
        description="The text of the rule"
    )
    applies: bool = Field(
        description="Whether this rule applies to the scenario"
    )
    reasoning: str = Field(
        description="Explanation of why the rule does/doesn't apply"
    )
    exclusions: list[str] = Field(
        default_factory=list,
        description="Rule IDs that are excluded because this rule applies"
    )


class GoldenScenario(BaseModel):
    """
    A scenario with explicit type and rule targeting.

    Extends the base Scenario concept with:
    - Explicit scenario type (positive, negative, edge_case, irrelevant)
    - Target rule IDs that this scenario is designed to test
    - Expected outcome based on the rules
    - Sub-category IDs for coverage tracking
    """

    description: str = Field(
        description="The user's request or question"
    )
    context: str = Field(
        default="",
        description="Additional context for the scenario"
    )
    category: str = Field(
        default="",
        description="The policy category this scenario belongs to"
    )
    scenario_type: ScenarioType = Field(
        description="Type of scenario (positive, negative, edge_case, irrelevant)"
    )
    target_rule_ids: list[str] = Field(
        default_factory=list,
        description="Rule IDs this scenario is designed to test"
    )
    expected_outcome: str = Field(
        default="",
        description="Expected response behavior based on rules"
    )
    sub_category_ids: list[str] = Field(
        default_factory=list,
        description="Sub-category IDs this scenario covers (for coverage tracking)"
    )

    def to_base_scenario(self) -> "Scenario":
        """Convert to base Scenario type for compatibility, preserving eval fields."""
        from synkro.types.core import Scenario
        return Scenario(
            description=self.description,
            context=self.context,
            category=self.category,
            scenario_type=self.scenario_type.value if self.scenario_type else None,
            target_rule_ids=self.target_rule_ids,
            expected_outcome=self.expected_outcome,
        )


class VerificationResult(BaseModel):
    """
    Result of verifying a trace against the Logic Map.

    The Auditor produces this to indicate whether a trace
    correctly applies all relevant rules without hallucination.
    """

    passed: bool = Field(
        description="Whether the trace passed verification"
    )
    issues: list[str] = Field(
        default_factory=list,
        description="List of issues found (if any)"
    )
    skipped_rules: list[str] = Field(
        default_factory=list,
        description="Rule IDs that should have been applied but weren't"
    )
    hallucinated_rules: list[str] = Field(
        default_factory=list,
        description="Rule IDs cited that don't exist or don't apply"
    )
    contradictions: list[str] = Field(
        default_factory=list,
        description="Logical contradictions found in the trace"
    )
    rules_verified: list[str] = Field(
        default_factory=list,
        description="Rule IDs that were correctly applied"
    )


__all__ = [
    "ScenarioType",
    "RuleCategory",
    "Rule",
    "LogicMap",
    "ReasoningStep",
    "GoldenScenario",
    "VerificationResult",
]
