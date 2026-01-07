"""Human-in-the-Loop session state management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synkro.types.logic_map import LogicMap, GoldenScenario


@dataclass
class HITLSession:
    """
    Tracks state of an interactive Logic Map and Scenario editing session.

    Supports undo/reset operations and maintains edit history for both
    Logic Map changes and scenario changes.

    Example:
        >>> session = HITLSession(original_logic_map=logic_map)
        >>> session.apply_change("Added rule R009", new_logic_map)
        >>> session.set_scenarios(scenarios, distribution)
        >>> session.apply_scenario_change("Added S21", new_scenarios, new_distribution)
        >>> session.undo()  # Reverts last change (rule or scenario)
        >>> session.reset()  # Reverts to original
    """

    original_logic_map: "LogicMap"
    current_logic_map: "LogicMap" = field(init=False)
    history: list[tuple[str, "LogicMap"]] = field(default_factory=list)

    # Scenario tracking
    original_scenarios: list["GoldenScenario"] | None = field(default=None)
    current_scenarios: list["GoldenScenario"] | None = field(default=None)
    original_distribution: dict[str, int] | None = field(default=None)
    current_distribution: dict[str, int] | None = field(default=None)
    scenario_history: list[tuple[str, list["GoldenScenario"], dict[str, int]]] = field(
        default_factory=list
    )

    def __post_init__(self) -> None:
        """Initialize current_logic_map from original."""
        self.current_logic_map = self.original_logic_map

    def set_scenarios(
        self,
        scenarios: list["GoldenScenario"],
        distribution: dict[str, int],
    ) -> None:
        """
        Initialize scenarios after they're generated.

        Args:
            scenarios: List of generated golden scenarios
            distribution: Type distribution dict
        """
        self.original_scenarios = scenarios
        self.current_scenarios = scenarios
        self.original_distribution = distribution
        self.current_distribution = distribution

    def apply_change(self, feedback: str, new_map: "LogicMap") -> None:
        """
        Record a Logic Map change in history and update current state.

        Args:
            feedback: The user feedback that triggered this change
            new_map: The new Logic Map after applying the change
        """
        self.history.append((feedback, self.current_logic_map))
        self.current_logic_map = new_map

    def apply_scenario_change(
        self,
        feedback: str,
        new_scenarios: list["GoldenScenario"],
        new_distribution: dict[str, int],
    ) -> None:
        """
        Record a scenario change in history and update current state.

        Args:
            feedback: The user feedback that triggered this change
            new_scenarios: The updated scenario list
            new_distribution: The updated distribution dict
        """
        if self.current_scenarios is not None and self.current_distribution is not None:
            self.scenario_history.append(
                (feedback, self.current_scenarios, self.current_distribution)
            )
        self.current_scenarios = new_scenarios
        self.current_distribution = new_distribution

    def undo(self) -> tuple["LogicMap | None", list["GoldenScenario"] | None, dict[str, int] | None]:
        """
        Undo the last change (either rule or scenario).

        Returns:
            Tuple of (logic_map or None, scenarios or None, distribution or None)
            indicating which was restored
        """
        # Check which has the most recent change
        rule_has_history = len(self.history) > 0
        scenario_has_history = len(self.scenario_history) > 0

        if not rule_has_history and not scenario_has_history:
            return None, None, None

        # For simplicity, undo the most recent change of either type
        # In practice, we could track a unified history with timestamps
        if scenario_has_history:
            _, prev_scenarios, prev_distribution = self.scenario_history.pop()
            self.current_scenarios = prev_scenarios
            self.current_distribution = prev_distribution
            return None, prev_scenarios, prev_distribution

        if rule_has_history:
            _, previous_map = self.history.pop()
            self.current_logic_map = previous_map
            return previous_map, None, None

        return None, None, None

    def reset(self) -> tuple["LogicMap", list["GoldenScenario"] | None, dict[str, int] | None]:
        """
        Reset to the original state, clearing all history.

        Returns:
            Tuple of (original LogicMap, original scenarios, original distribution)
        """
        self.history.clear()
        self.scenario_history.clear()
        self.current_logic_map = self.original_logic_map
        self.current_scenarios = self.original_scenarios
        self.current_distribution = self.original_distribution
        return self.original_logic_map, self.original_scenarios, self.original_distribution

    @property
    def can_undo(self) -> bool:
        """Check if undo is available."""
        return len(self.history) > 0 or len(self.scenario_history) > 0

    @property
    def change_count(self) -> int:
        """Number of changes made in this session (rules + scenarios)."""
        return len(self.history) + len(self.scenario_history)

    @property
    def rule_change_count(self) -> int:
        """Number of rule changes made in this session."""
        return len(self.history)

    @property
    def scenario_change_count(self) -> int:
        """Number of scenario changes made in this session."""
        return len(self.scenario_history)

    @property
    def has_scenarios(self) -> bool:
        """Check if scenarios have been initialized."""
        return self.current_scenarios is not None
