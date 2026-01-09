"""Progress reporting abstraction for generation pipeline.

This module provides a Protocol for progress reporting and implementations
for different use cases (rich console, silent for testing, etc.).

Enhanced for Golden Trace pipeline with:
- Logic Map logging
- Scenario type distribution
- Per-trace category/type logging
"""

from typing import Protocol, TYPE_CHECKING, Callable

from synkro.types.core import Plan, Scenario, Trace, GradeResult

if TYPE_CHECKING:
    from synkro.types.logic_map import LogicMap, GoldenScenario


class ProgressReporter(Protocol):
    """
    Protocol for reporting generation progress.
    
    Implement this to customize how progress is displayed or logged.
    
    Examples:
        >>> # Use silent reporter for testing
        >>> generator = Generator(reporter=SilentReporter())
        
        >>> # Use rich reporter for CLI (default)
        >>> generator = Generator(reporter=RichReporter())
    """
    
    def on_start(self, traces: int, model: str, dataset_type: str) -> None:
        """Called when generation starts."""
        ...
    
    def on_plan_complete(self, plan: Plan) -> None:
        """Called when planning phase completes."""
        ...
    
    def on_scenario_progress(self, completed: int, total: int) -> None:
        """Called during scenario generation."""
        ...
    
    def on_response_progress(self, completed: int, total: int) -> None:
        """Called during response generation."""
        ...
    
    def on_responses_complete(self, traces: list[Trace]) -> None:
        """Called when all responses are generated."""
        ...
    
    def on_grading_progress(self, completed: int, total: int) -> None:
        """Called during grading."""
        ...
    
    def on_grading_complete(self, traces: list[Trace], pass_rate: float) -> None:
        """Called when grading completes."""
        ...
    
    def on_refinement_start(self, iteration: int, failed_count: int) -> None:
        """Called when a refinement iteration starts."""
        ...
    
    def on_grading_skipped(self) -> None:
        """Called when grading is skipped."""
        ...
    
    def on_complete(
        self,
        dataset_size: int,
        elapsed_seconds: float,
        pass_rate: float | None,
        total_cost: float | None = None,
        generation_calls: int | None = None,
        grading_calls: int | None = None,
        scenario_calls: int | None = None,
        response_calls: int | None = None,
    ) -> None:
        """Called when generation is complete."""
        ...

    def on_logic_map_complete(self, logic_map: "LogicMap") -> None:
        """Called when logic extraction completes (Stage 1)."""
        ...

    def on_golden_scenarios_complete(
        self, scenarios: list["GoldenScenario"], distribution: dict[str, int]
    ) -> None:
        """Called when golden scenarios are generated (Stage 2)."""
        ...



class _NoOpContextManager:
    """No-op context manager for SilentReporter spinner."""
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


class SilentReporter:
    """
    No-op reporter for testing and embedding.

    Use this when you don't want any console output.

    Examples:
        >>> generator = Generator(reporter=SilentReporter())
        >>> dataset = generator.generate(policy)  # No console output
    """

    def spinner(self, message: str = "Thinking..."):
        """No-op spinner for silent mode."""
        return _NoOpContextManager()

    def on_start(self, traces: int, model: str, dataset_type: str) -> None:
        pass
    
    def on_plan_complete(self, plan: Plan) -> None:
        pass
    
    def on_scenario_progress(self, completed: int, total: int) -> None:
        pass
    
    def on_response_progress(self, completed: int, total: int) -> None:
        pass
    
    def on_responses_complete(self, traces: list[Trace]) -> None:
        pass
    
    def on_grading_progress(self, completed: int, total: int) -> None:
        pass
    
    def on_grading_complete(self, traces: list[Trace], pass_rate: float) -> None:
        pass
    
    def on_refinement_start(self, iteration: int, failed_count: int) -> None:
        pass
    
    def on_grading_skipped(self) -> None:
        pass

    def on_complete(
        self,
        dataset_size: int,
        elapsed_seconds: float,
        pass_rate: float | None,
        total_cost: float | None = None,
        generation_calls: int | None = None,
        grading_calls: int | None = None,
        scenario_calls: int | None = None,
        response_calls: int | None = None,
    ) -> None:
        pass

    def on_logic_map_complete(self, logic_map) -> None:
        pass

    def on_golden_scenarios_complete(self, scenarios, distribution) -> None:
        pass



class RichReporter:
    """
    Rich console reporter with progress bars and formatted output.

    This is the default reporter that provides the familiar synkro CLI experience.
    """

    def __init__(self):
        from rich.console import Console
        self.console = Console()
        self._progress = None
        self._current_task = None

    def spinner(self, message: str = "Thinking..."):
        """Context manager that shows a loading spinner.

        Usage:
            with reporter.spinner("Thinking..."):
                await some_llm_call()
        """
        from rich.status import Status
        self.console.print()  # Add space above spinner
        return Status(f"[cyan]{message}[/cyan]", spinner="dots", console=self.console)
    
    def on_start(self, traces: int, model: str, dataset_type: str) -> None:
        self.console.print()
        self.console.print(
            f"[cyan]⚡ Generating {traces} traces[/cyan] "
            f"[dim]| {dataset_type.upper()} | {model}[/dim]"
        )
    
    def on_plan_complete(self, plan: Plan) -> None:
        from rich.table import Table
        
        self.console.print(f"[green]📋 Planning[/green] [dim]{len(plan.categories)} categories[/dim]")
        
        cat_table = Table(title="Categories", show_header=True, header_style="bold cyan")
        cat_table.add_column("Name")
        cat_table.add_column("Description")
        cat_table.add_column("Count", justify="right")
        for cat in plan.categories:
            cat_table.add_row(cat.name, cat.description, str(cat.count))
        self.console.print(cat_table)
        self.console.print()
    
    def on_scenario_progress(self, completed: int, total: int) -> None:
        pass

    def on_response_progress(self, completed: int, total: int) -> None:
        pass

    def on_grading_progress(self, completed: int, total: int) -> None:
        pass  # Progress shown in on_grading_complete
    
    def on_grading_complete(self, traces: list[Trace], pass_rate: float) -> None:
        self.console.print(f"[green]⚖️  Grading[/green] [dim]{pass_rate:.0f}% passed[/dim]")
        for idx, trace in enumerate(traces, 1):
            scenario_preview = trace.scenario.description[:40] + "..." if len(trace.scenario.description) > 40 else trace.scenario.description
            if trace.grade and trace.grade.passed:
                self.console.print(f"  [dim]#{idx}[/dim] [cyan]{scenario_preview}[/cyan] [green]✓ Passed[/green]")
            else:
                issues = ", ".join(trace.grade.issues[:2]) if trace.grade and trace.grade.issues else "No specific issues"
                issues_preview = issues[:40] + "..." if len(issues) > 40 else issues
                self.console.print(f"  [dim]#{idx}[/dim] [cyan]{scenario_preview}[/cyan] [red]✗ Failed[/red] [dim]{issues_preview}[/dim]")
    
    def on_refinement_start(self, iteration: int, failed_count: int) -> None:
        self.console.print(f"  [yellow]↻ Refining {failed_count} failed traces (iteration {iteration})...[/yellow]")
    
    def on_grading_skipped(self) -> None:
        self.console.print(f"  [dim]⚖️  Grading skipped[/dim]")
    
    def on_complete(
        self,
        dataset_size: int,
        elapsed_seconds: float,
        pass_rate: float | None,
        total_cost: float | None = None,
        generation_calls: int | None = None,
        grading_calls: int | None = None,
        scenario_calls: int | None = None,
        response_calls: int | None = None,
    ) -> None:
        from rich.panel import Panel
        from rich.table import Table

        elapsed_str = f"{int(elapsed_seconds) // 60}m {int(elapsed_seconds) % 60}s" if elapsed_seconds >= 60 else f"{int(elapsed_seconds)}s"

        self.console.print()
        summary = Table.grid(padding=(0, 2))
        summary.add_column(style="green")
        summary.add_column()
        summary.add_row("✅ Done!", f"Generated {dataset_size} traces in {elapsed_str}")
        if pass_rate is not None:
            summary.add_row("📊 Quality:", f"{pass_rate:.0f}% passed verification")
        if total_cost is not None and total_cost > 0:
            summary.add_row("💰 Cost:", f"${total_cost:.4f}")
        if scenario_calls is not None and response_calls is not None:
            calls_str = f"{scenario_calls} scenario + {response_calls} response"
            if grading_calls is not None and grading_calls > 0:
                calls_str += f" + {grading_calls} grading"
            summary.add_row("🔄 LLM Calls:", calls_str)
        elif generation_calls is not None:
            calls_str = f"{generation_calls} generation"
            if grading_calls is not None and grading_calls > 0:
                calls_str += f" + {grading_calls} grading"
            summary.add_row("🔄 LLM Calls:", calls_str)
        self.console.print(Panel(summary, border_style="green", title="[green]Complete[/green]"))
        self.console.print()

    def on_logic_map_complete(self, logic_map) -> None:
        """Logic Map details shown in HITL session."""
        pass

    def on_golden_scenarios_complete(self, scenarios, distribution) -> None:
        """Scenario details shown in HITL session."""
        pass

    def on_responses_complete(self, traces: list[Trace]) -> None:
        """Enhanced to show category and type for each trace."""
        self.console.print(f"\n[green]✍️  Traces[/green] [dim]{len(traces)} generated[/dim]")

        # Group by category
        by_category = {}
        for trace in traces:
            cat = trace.scenario.category or "uncategorized"
            by_category.setdefault(cat, []).append(trace)

        for cat_name, cat_traces in by_category.items():
            self.console.print(f"\n  [cyan]📁 {cat_name}[/cyan] ({len(cat_traces)} traces)")

            for trace in cat_traces[:3]:  # Show first 3 per category
                # Try to get scenario type if available
                scenario_type = getattr(trace.scenario, 'scenario_type', None)
                if scenario_type:
                    type_indicator = {
                        "positive": "[green]✓[/green]",
                        "negative": "[red]✗[/red]",
                        "edge_case": "[yellow]⚡[/yellow]",
                        "irrelevant": "[dim]○[/dim]"
                    }.get(scenario_type if isinstance(scenario_type, str) else scenario_type.value, "[white]?[/white]")
                else:
                    type_indicator = "[white]•[/white]"

                user_preview = trace.user_message[:50] + "..." if len(trace.user_message) > 50 else trace.user_message
                self.console.print(f"    {type_indicator} [blue]{user_preview}[/blue]")

            if len(cat_traces) > 3:
                self.console.print(f"    [dim]... and {len(cat_traces) - 3} more[/dim]")


class CallbackReporter:
    """
    Reporter that invokes user-provided callbacks for progress events.

    Use this when you need programmatic access to progress events
    (e.g., updating a progress bar, logging to a file, etc.)

    Examples:
        >>> def on_progress(event: str, data: dict):
        ...     print(f"{event}: {data}")
        ...
        >>> reporter = CallbackReporter(on_progress=on_progress)
        >>> generator = Generator(reporter=reporter)

        >>> # With specific event handlers
        >>> reporter = CallbackReporter(
        ...     on_start=lambda traces, model, dtype: print(f"Starting {traces} traces"),
        ...     on_complete=lambda size, elapsed, rate: print(f"Done! {size} traces"),
        ... )
    """

    def __init__(
        self,
        on_progress: "Callable[[str, dict], None] | None" = None,
        on_start: "Callable[[int, str, str], None] | None" = None,
        on_plan_complete: "Callable[[Plan], None] | None" = None,
        on_scenario_progress: "Callable[[int, int], None] | None" = None,
        on_scenarios_complete: "Callable[[list[Scenario]], None] | None" = None,
        on_response_progress: "Callable[[int, int], None] | None" = None,
        on_responses_complete: "Callable[[list[Trace]], None] | None" = None,
        on_grading_progress: "Callable[[int, int], None] | None" = None,
        on_grading_complete: "Callable[[list[Trace], float], None] | None" = None,
        on_complete: "Callable[[int, float, float | None], None] | None" = None,
    ):
        """
        Initialize the callback reporter.

        Args:
            on_progress: Generic callback for all events. Receives (event_name, data_dict).
            on_start: Called when generation starts (traces, model, dataset_type)
            on_plan_complete: Called when planning completes (plan)
            on_scenario_progress: Called during scenario generation (completed, total)
            on_scenarios_complete: Called when scenarios are done (scenarios list)
            on_response_progress: Called during response generation (completed, total)
            on_responses_complete: Called when responses are done (traces list)
            on_grading_progress: Called during grading (completed, total)
            on_grading_complete: Called when grading is done (traces, pass_rate)
            on_complete: Called when generation completes (dataset_size, elapsed, pass_rate)
        """
        self._on_progress = on_progress
        self._on_start = on_start
        self._on_plan_complete = on_plan_complete
        self._on_scenario_progress = on_scenario_progress
        self._on_scenarios_complete = on_scenarios_complete
        self._on_response_progress = on_response_progress
        self._on_responses_complete = on_responses_complete
        self._on_grading_progress = on_grading_progress
        self._on_grading_complete = on_grading_complete
        self._on_complete_cb = on_complete

    def _emit(self, event: str, data: dict) -> None:
        """Emit an event to the generic callback."""
        if self._on_progress:
            self._on_progress(event, data)

    def spinner(self, message: str = "Thinking..."):
        """No-op spinner for callback mode."""
        return _NoOpContextManager()

    def on_start(self, traces: int, model: str, dataset_type: str) -> None:
        self._emit("start", {"traces": traces, "model": model, "dataset_type": dataset_type})
        if self._on_start:
            self._on_start(traces, model, dataset_type)

    def on_plan_complete(self, plan: Plan) -> None:
        self._emit("plan_complete", {"categories": len(plan.categories)})
        if self._on_plan_complete:
            self._on_plan_complete(plan)

    def on_scenario_progress(self, completed: int, total: int) -> None:
        self._emit("scenario_progress", {"completed": completed, "total": total})
        if self._on_scenario_progress:
            self._on_scenario_progress(completed, total)

    def on_scenarios_complete(self, scenarios: list[Scenario]) -> None:
        self._emit("scenarios_complete", {"count": len(scenarios)})
        if self._on_scenarios_complete:
            self._on_scenarios_complete(scenarios)

    def on_response_progress(self, completed: int, total: int) -> None:
        self._emit("response_progress", {"completed": completed, "total": total})
        if self._on_response_progress:
            self._on_response_progress(completed, total)

    def on_responses_complete(self, traces: list[Trace]) -> None:
        self._emit("responses_complete", {"count": len(traces)})
        if self._on_responses_complete:
            self._on_responses_complete(traces)

    def on_grading_progress(self, completed: int, total: int) -> None:
        self._emit("grading_progress", {"completed": completed, "total": total})
        if self._on_grading_progress:
            self._on_grading_progress(completed, total)

    def on_grading_complete(self, traces: list[Trace], pass_rate: float) -> None:
        self._emit("grading_complete", {"count": len(traces), "pass_rate": pass_rate})
        if self._on_grading_complete:
            self._on_grading_complete(traces, pass_rate)

    def on_refinement_start(self, iteration: int, failed_count: int) -> None:
        self._emit("refinement_start", {"iteration": iteration, "failed_count": failed_count})

    def on_grading_skipped(self) -> None:
        self._emit("grading_skipped", {})

    def on_complete(
        self,
        dataset_size: int,
        elapsed_seconds: float,
        pass_rate: float | None,
        total_cost: float | None = None,
        generation_calls: int | None = None,
        grading_calls: int | None = None,
        scenario_calls: int | None = None,
        response_calls: int | None = None,
    ) -> None:
        self._emit("complete", {
            "dataset_size": dataset_size,
            "elapsed_seconds": elapsed_seconds,
            "pass_rate": pass_rate,
            "total_cost": total_cost,
            "generation_calls": generation_calls,
            "grading_calls": grading_calls,
            "scenario_calls": scenario_calls,
            "response_calls": response_calls,
        })
        if self._on_complete_cb:
            self._on_complete_cb(dataset_size, elapsed_seconds, pass_rate)

    def on_logic_map_complete(self, logic_map) -> None:
        self._emit("logic_map_complete", {"rules_count": len(logic_map.rules)})

    def on_golden_scenarios_complete(self, scenarios, distribution) -> None:
        self._emit("golden_scenarios_complete", {"count": len(scenarios), "distribution": distribution})


__all__ = ["ProgressReporter", "SilentReporter", "RichReporter", "CallbackReporter"]

