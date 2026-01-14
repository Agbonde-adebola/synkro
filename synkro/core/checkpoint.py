"""Checkpoint manager for resumable generation."""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field
from rich.console import Console

if TYPE_CHECKING:
    from synkro.types.core import Trace
    from synkro.types.logic_map import LogicMap, GoldenScenario

console = Console()

# Hash length for policy matching (first 16 chars of SHA256)
HASH_LENGTH = 16


class CheckpointData(BaseModel):
    """Data stored in a checkpoint file."""

    # Metadata
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    policy_hash: str = ""
    target_traces: int = 0
    dataset_type: str = ""

    # Stage 1: Logic Map
    logic_map_data: dict | None = None

    # Stage 2: Scenarios
    scenarios_data: list[dict] = Field(default_factory=list)
    scenario_distribution: dict[str, int] = Field(default_factory=dict)

    # Stage 3: Generated traces
    traces_data: list[dict] = Field(default_factory=list)
    completed_scenario_indices: list[int] = Field(default_factory=list)

    # Stage 4: Verified traces
    verified_traces_data: list[dict] = Field(default_factory=list)
    verification_complete: bool = False


class CheckpointManager:
    """
    Manages checkpoints for resumable generation.

    Saves progress after each stage and allows resuming from the last
    successful checkpoint.

    Examples:
        >>> manager = CheckpointManager("./checkpoints")
        >>> manager.save_logic_map(logic_map, policy_hash, 100, "messages")
        >>> manager.save_scenarios(scenarios, distribution)
        >>> manager.save_trace(trace, scenario_index)

        >>> # Resume from checkpoint
        >>> checkpoint = manager.load()
        >>> if checkpoint:
        ...     logic_map = checkpoint.get_logic_map()
        ...     completed = checkpoint.completed_scenario_indices
    """

    def __init__(self, checkpoint_dir: str | Path):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "checkpoint.json"
        self._data: CheckpointData | None = None

    def _load_or_create(self) -> CheckpointData:
        """Load existing checkpoint or create new one."""
        if self._data is not None:
            return self._data

        if self.checkpoint_file.exists():
            with open(self.checkpoint_file) as f:
                data = json.load(f)
            self._data = CheckpointData.model_validate(data)
        else:
            self._data = CheckpointData()

        return self._data

    def _save(self) -> None:
        """Save checkpoint to disk."""
        if self._data is None:
            return

        with open(self.checkpoint_file, "w") as f:
            json.dump(self._data.model_dump(), f, indent=2)

    def has_checkpoint(self) -> bool:
        """Check if a checkpoint exists."""
        return self.checkpoint_file.exists()

    def load(self) -> CheckpointData | None:
        """Load checkpoint if it exists."""
        if not self.has_checkpoint():
            return None
        return self._load_or_create()

    def matches_config(self, policy_hash: str, target_traces: int, dataset_type: str) -> bool:
        """Check if checkpoint matches the current generation config."""
        data = self._load_or_create()
        return (
            data.policy_hash == policy_hash
            and data.target_traces == target_traces
            and data.dataset_type == dataset_type
        )

    def save_logic_map(
        self,
        logic_map: "LogicMap",
        policy_hash: str,
        target_traces: int,
        dataset_type: str,
    ) -> None:
        """Save Logic Map (Stage 1 complete)."""
        data = self._load_or_create()
        data.policy_hash = policy_hash
        data.target_traces = target_traces
        data.dataset_type = dataset_type
        data.logic_map_data = logic_map.model_dump()
        self._save()
        console.print("[dim]💾 Checkpoint: Logic Map saved[/dim]")

    def save_scenarios(
        self,
        scenarios: list["GoldenScenario"],
        distribution: dict[str, int],
    ) -> None:
        """Save scenarios (Stage 2 complete)."""
        data = self._load_or_create()
        data.scenarios_data = [s.model_dump() for s in scenarios]
        data.scenario_distribution = distribution
        self._save()
        console.print("[dim]💾 Checkpoint: Scenarios saved[/dim]")

    def save_trace(self, trace: "Trace", scenario_index: int) -> None:
        """Save a generated trace (incremental Stage 3)."""
        data = self._load_or_create()
        data.traces_data.append(trace.model_dump())
        data.completed_scenario_indices.append(scenario_index)
        self._save()

    def save_traces_batch(self, traces: list["Trace"], indices: list[int]) -> None:
        """Save a batch of traces at once."""
        data = self._load_or_create()
        for trace, idx in zip(traces, indices):
            data.traces_data.append(trace.model_dump())
            data.completed_scenario_indices.append(idx)
        self._save()
        console.print(f"[dim]💾 Checkpoint: {len(traces)} traces saved[/dim]")

    def save_verified_traces(self, traces: list["Trace"]) -> None:
        """Save verified traces (Stage 4 complete)."""
        data = self._load_or_create()
        data.verified_traces_data = [t.model_dump() for t in traces]
        data.verification_complete = True
        self._save()
        console.print("[dim]💾 Checkpoint: Verification complete[/dim]")

    def get_logic_map(self) -> "LogicMap | None":
        """Retrieve Logic Map from checkpoint."""
        from synkro.types.logic_map import LogicMap

        data = self._load_or_create()
        if data.logic_map_data:
            return LogicMap.model_validate(data.logic_map_data)
        return None

    def get_scenarios(self) -> list["GoldenScenario"]:
        """Retrieve scenarios from checkpoint."""
        from synkro.types.logic_map import GoldenScenario

        data = self._load_or_create()
        return self._validate_list(data.scenarios_data, GoldenScenario)

    def get_traces(self) -> list["Trace"]:
        """Retrieve traces from checkpoint."""
        from synkro.types.core import Trace

        data = self._load_or_create()
        return self._validate_list(data.traces_data, Trace)

    def get_verified_traces(self) -> list["Trace"]:
        """Retrieve verified traces from checkpoint."""
        from synkro.types.core import Trace

        data = self._load_or_create()
        return self._validate_list(data.verified_traces_data, Trace)

    def _validate_list(self, data_list: list[dict], model_class: type) -> list:
        """Helper to validate a list of dicts into model instances."""
        return [model_class.model_validate(item) for item in data_list]

    def get_pending_scenario_indices(self, total: int) -> list[int]:
        """Get indices of scenarios that haven't been processed yet."""
        data = self._load_or_create()
        completed = set(data.completed_scenario_indices)
        return [i for i in range(total) if i not in completed]

    def clear(self) -> None:
        """Clear the checkpoint."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
        self._data = None
        console.print("[dim]🗑️ Checkpoint cleared[/dim]")

    @property
    def stage(self) -> str:
        """Get the current stage based on checkpoint data."""
        data = self._load_or_create()

        if data.verification_complete:
            return "complete"
        if data.traces_data:
            return "traces"  # In progress or done
        if data.scenarios_data:
            return "scenarios"
        if data.logic_map_data:
            return "logic_map"
        return "start"

    def summary(self) -> str:
        """Get a summary of the checkpoint status."""
        data = self._load_or_create()

        lines = [
            f"Checkpoint Status",
            f"=================",
            f"Stage: {self.stage}",
            f"Target traces: {data.target_traces}",
            f"Logic Map: {'✓' if data.logic_map_data else '✗'}",
            f"Scenarios: {len(data.scenarios_data)}",
            f"Traces: {len(data.traces_data)}/{data.target_traces}",
            f"Verified: {'✓' if data.verification_complete else '✗'}",
        ]

        return "\n".join(lines)


def hash_policy(policy_text: str) -> str:
    """Create a hash of policy text for checkpoint matching."""
    return hashlib.sha256(policy_text.encode()).hexdigest()[:HASH_LENGTH]


__all__ = ["CheckpointManager", "CheckpointData", "hash_policy"]
