"""Standalone HITL functions for editing rules and scenarios.

These functions provide standalone access to the Human-in-the-Loop editing
capabilities, usable outside the main pipeline context.

Examples:
    >>> # Edit rules with natural language
    >>> new_logic_map, summary = await synkro.edit_rules(
    ...     logic_map,
    ...     "add rule: overtime requires director approval",
    ...     policy_text
    ... )
    >>> print(summary)

    >>> # Edit scenarios
    >>> new_scenarios, distribution, summary = await synkro.edit_scenarios(
    ...     scenarios,
    ...     "add 5 edge cases for boundary conditions",
    ...     policy_text,
    ...     logic_map
    ... )
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from synkro.llm.client import LLM
from synkro.models import Model

if TYPE_CHECKING:
    from synkro.types.logic_map import GoldenScenario, LogicMap


async def edit_rules_async(
    logic_map: "LogicMap",
    instruction: str,
    policy_text: str,
    model: Model,
    base_url: str | None = None,
    conversation_history: str = "No previous feedback in this session.",
    llm: LLM | None = None,
) -> tuple["LogicMap", str]:
    """
    Edit a Logic Map using natural language instructions.

    This function wraps the LogicMapEditor to provide a simple interface
    for modifying rules. Supports operations like:
    - "add a rule for overtime approval"
    - "remove R005"
    - "merge R002 and R003"
    - "split R001 into separate rules"
    - "rename R004 to be more specific"

    Args:
        logic_map: Current Logic Map to edit
        instruction: Natural language instruction (e.g., "add rule: overtime needs approval")
        policy_text: Original policy text for context
        model: Model to use for editing (e.g., Google.GEMINI_2_5_PRO)
        base_url: Optional API base URL for local providers
        conversation_history: Previous feedback for multi-turn context

    Returns:
        Tuple of (modified LogicMap, changes summary string)

    Examples:
        >>> # Add a new rule
        >>> new_map, summary = await edit_rules(
        ...     logic_map,
        ...     "add rule: expenses over $100 need VP approval",
        ...     policy_text,
        ...     model=Google.GEMINI_2_5_PRO
        ... )
        >>> print(summary)  # "Added R006: Expenses over $100 need VP approval"

        >>> # Merge rules
        >>> new_map, summary = await edit_rules(
        ...     logic_map,
        ...     "merge R002 and R003 since they cover the same condition",
        ...     policy_text,
        ...     model=Google.GEMINI_2_5_PRO
        ... )

        >>> # Remove a rule
        >>> new_map, summary = await edit_rules(
        ...     logic_map,
        ...     "remove R004 - it's redundant",
        ...     policy_text,
        ...     model=Google.GEMINI_2_5_PRO
        ... )
    """
    from synkro.interactive.logic_map_editor import LogicMapEditor

    # Use provided LLM or create one
    if llm is None:
        llm = LLM(model=model, base_url=base_url, temperature=0.3)
    editor = LogicMapEditor(llm=llm)

    # Apply the edit
    new_logic_map, summary = await editor.refine(
        logic_map=logic_map,
        user_feedback=instruction,
        policy_text=policy_text,
        conversation_history=conversation_history,
    )

    return new_logic_map, summary


def edit_rules(
    logic_map: "LogicMap",
    instruction: str,
    policy_text: str,
    model: Model,
    base_url: str | None = None,
    conversation_history: str = "No previous feedback in this session.",
) -> tuple["LogicMap", str]:
    """
    Edit a Logic Map using natural language (sync wrapper).

    See edit_rules_async for full documentation.
    """
    return asyncio.run(
        edit_rules_async(logic_map, instruction, policy_text, model, base_url, conversation_history)
    )


async def edit_scenarios_async(
    scenarios: list["GoldenScenario"],
    instruction: str,
    policy_text: str,
    logic_map: "LogicMap",
    model: Model,
    distribution: dict[str, int] | None = None,
    base_url: str | None = None,
    conversation_history: str = "No previous feedback in this session.",
    llm: LLM | None = None,
) -> tuple[list["GoldenScenario"], dict[str, int], str]:
    """
    Edit scenarios using natural language instructions.

    This function wraps the ScenarioEditor to provide a simple interface
    for modifying scenarios. Supports operations like:
    - "add 5 more edge cases"
    - "delete S3"
    - "add a scenario for overtime requests"
    - "more negative cases for rule R002"
    - "replace S1 with a clearer example"

    Args:
        scenarios: Current list of scenarios to edit
        instruction: Natural language instruction
        policy_text: Original policy text for context
        logic_map: Logic Map for rule references
        model: Model to use for editing (e.g., Google.GEMINI_2_5_PRO)
        distribution: Current type distribution (calculated if not provided)
        base_url: Optional API base URL for local providers
        conversation_history: Previous feedback for multi-turn context

    Returns:
        Tuple of (modified scenarios list, updated distribution, changes summary)

    Examples:
        >>> # Add edge cases
        >>> new_scenarios, dist, summary = await edit_scenarios(
        ...     scenarios,
        ...     "add 5 edge cases for boundary conditions",
        ...     policy_text,
        ...     logic_map,
        ...     model=Google.GEMINI_2_5_PRO
        ... )
        >>> print(f"Now have {dist['edge_case']} edge cases")

        >>> # Delete a scenario
        >>> new_scenarios, dist, summary = await edit_scenarios(
        ...     scenarios,
        ...     "delete S3 - it's too similar to S1",
        ...     policy_text,
        ...     logic_map,
        ...     model=Google.GEMINI_2_5_PRO
        ... )

        >>> # Add scenarios targeting a specific rule
        >>> new_scenarios, dist, summary = await edit_scenarios(
        ...     scenarios,
        ...     "add 3 negative cases that violate R004",
        ...     policy_text,
        ...     logic_map,
        ...     model=Google.GEMINI_2_5_PRO
        ... )
    """
    from synkro.interactive.scenario_editor import ScenarioEditor

    # Calculate current distribution if not provided
    if distribution is None:
        distribution = {}
        for s in scenarios:
            stype = (
                s.scenario_type.value if hasattr(s.scenario_type, "value") else str(s.scenario_type)
            )
            distribution[stype] = distribution.get(stype, 0) + 1

    # Use provided LLM or create one
    if llm is None:
        llm = LLM(model=model, base_url=base_url, temperature=0.3)
    editor = ScenarioEditor(llm=llm)

    # Apply the edit
    new_scenarios, new_distribution, summary = await editor.refine(
        scenarios=scenarios,
        distribution=distribution,
        user_feedback=instruction,
        policy_text=policy_text,
        logic_map=logic_map,
        conversation_history=conversation_history,
    )

    return new_scenarios, new_distribution, summary


def edit_scenarios(
    scenarios: list["GoldenScenario"],
    instruction: str,
    policy_text: str,
    logic_map: "LogicMap",
    model: Model,
    distribution: dict[str, int] | None = None,
    base_url: str | None = None,
    conversation_history: str = "No previous feedback in this session.",
) -> tuple[list["GoldenScenario"], dict[str, int], str]:
    """
    Edit scenarios using natural language (sync wrapper).

    See edit_scenarios_async for full documentation.
    """
    return asyncio.run(
        edit_scenarios_async(
            scenarios,
            instruction,
            policy_text,
            logic_map,
            model,
            distribution,
            base_url,
            conversation_history,
        )
    )


__all__ = [
    "edit_rules",
    "edit_rules_async",
    "edit_scenarios",
    "edit_scenarios_async",
]
