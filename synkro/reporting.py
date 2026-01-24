"""Progress reporting abstraction for generation pipeline.

This module provides a Protocol for progress reporting and implementations
for different use cases (rich console, silent for testing, etc.).

Enhanced for Golden Trace pipeline with:
- Logic Map logging
- Scenario type distribution
- Per-trace category/type logging
"""

from typing import TYPE_CHECKING, Callable, Protocol

# Re-export Callable so it's available for type hints within strings
from synkro.types.core import Plan, Scenario, Trace

if TYPE_CHECKING:
    from synkro.interactive.live_display import LiveProgressDisplay
    from synkro.types.coverage import CoverageReport, SubCategoryTaxonomy
    from synkro.types.logic_map import GoldenScenario, LogicMap


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
        refinement_calls: int | None = None,
        hitl_calls: int | None = None,
        coverage_calls: int | None = None,
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

    def on_taxonomy_extracted(self, taxonomy: "SubCategoryTaxonomy") -> None:
        """Called when sub-category taxonomy is extracted for coverage tracking."""
        ...

    def on_coverage_calculated(self, report: "CoverageReport") -> None:
        """Called when coverage metrics are calculated."""
        ...

    def on_coverage_improved(
        self,
        before: "CoverageReport",
        after: "CoverageReport",
        added_scenarios: int,
    ) -> None:
        """Called when coverage is improved via HITL commands."""
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
        refinement_calls: int | None = None,
        hitl_calls: int | None = None,
        coverage_calls: int | None = None,
    ) -> None:
        pass

    def on_logic_map_complete(self, logic_map) -> None:
        pass

    def on_golden_scenarios_complete(self, scenarios, distribution) -> None:
        pass

    def on_taxonomy_extracted(self, taxonomy) -> None:
        pass

    def on_coverage_calculated(self, report) -> None:
        pass

    def on_coverage_improved(self, before, after, added_scenarios) -> None:
        pass


class RichReporter:
    """
    Rich console reporter with live-updating display, spinners, and colors.

    Uses LiveProgressDisplay for a polished, single-panel UI that updates in-place
    with spinners, progress bars, and emoji indicators. This is the default reporter
    that provides the familiar synkro CLI experience.
    """

    def __init__(self):
        from rich.console import Console

        from synkro.interactive.live_display import LiveProgressDisplay

        self.console = Console()
        self._display = LiveProgressDisplay()
        self._start_time: float | None = None
        self._model: str = ""
        self._traces_target: int = 0
        self._cost_source: "Callable[[], float] | None" = None
        self._final_traces: list[Trace] | None = None
        self._show_traces: bool = True  # Display traces after generation

    def set_cost_source(self, cost_fn: "Callable[[], float]") -> None:
        """Set a function that returns the current total cost.

        This enables live cost updates in the display. The function will
        be called on each display refresh to get the latest cost.

        Args:
            cost_fn: A callable that returns the current total cost as a float.
        """
        self._cost_source = cost_fn
        # Also set on display for live rendering updates
        self._display.set_cost_source(cost_fn)

    @property
    def display(self) -> "LiveProgressDisplay":
        """Get the live display instance for HITL integration."""
        return self._display

    def spinner(self, message: str = "Thinking..."):
        """Context manager that shows a loading spinner.

        When the Live display is active, this returns a no-op and updates the phase
        instead of showing a separate spinner (which would stack panels).
        """
        # If live display is running, don't show separate spinner - update phase instead
        if self._display.is_active:
            self._display.update_phase(message.replace("...", ""))
            return _NoOpContextManager()

        from rich.status import Status

        return Status(f"[cyan]{message}[/cyan]", spinner="dots", console=self.console)

    def on_start(self, traces: int, model: str, dataset_type: str) -> None:
        import time

        self._start_time = time.time()
        self._model = model
        self._traces_target = traces
        self._display.start(model=model, dataset_type=dataset_type, traces_target=traces)
        self._display.update_phase("Starting")
        self._display.add_event(f"START: Generating {traces} {dataset_type} traces")

    def on_plan_complete(self, plan: Plan) -> None:
        """Update phase to show planning is complete."""
        cat_names = ", ".join(c.name for c in plan.categories[:3])
        if len(plan.categories) > 3:
            cat_names += f" +{len(plan.categories) - 3}"
        self._display.update_phase("Planning Complete")
        self._display.add_activity(f"{len(plan.categories)} categories: {cat_names}")
        self._display.add_event(f"PLAN: {len(plan.categories)} categories defined")
        self._update_elapsed()

    def on_scenario_progress(self, completed: int, total: int) -> None:
        self._display.update_progress(completed, total)
        self._update_elapsed()

    def on_response_progress(self, completed: int, total: int) -> None:
        self._display.update_phase("Generating Traces")
        self._display.update_progress(completed, total)
        if completed == 1 or completed == total or completed % 5 == 0:
            self._display.add_event(f"TRACE: Processing {completed}/{total}...")
        self._update_elapsed()

    def on_grading_progress(self, completed: int, total: int) -> None:
        self._display.update_phase("Verifying")
        self._display.update_progress(completed, total)
        self._update_elapsed()

    def on_grading_complete(self, traces: list[Trace], pass_rate: float) -> None:
        self._display._state.pass_rate = pass_rate
        self._display.update_phase("Verification Complete")
        passed = sum(1 for t in traces if t.grade and t.grade.passed)
        self._display.add_activity(f"{passed}/{len(traces)} passed ({pass_rate:.0f}%)")
        self._display.add_event(f"VERIFY: {passed}/{len(traces)} passed ({pass_rate:.0f}%)")
        self._update_elapsed()
        # Store traces for display after completion
        self._final_traces = traces

    def on_refinement_start(self, iteration: int, failed_count: int) -> None:
        self._display.update_phase("Refining")
        self._display.add_activity(f"Iteration {iteration}: {failed_count} failed traces")
        self._update_elapsed()

    def on_grading_skipped(self) -> None:
        self._display.add_activity("Grading skipped")

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
        refinement_calls: int | None = None,
        hitl_calls: int | None = None,
        coverage_calls: int | None = None,
    ) -> None:
        self._display.set_complete(
            traces_count=dataset_size,
            elapsed_seconds=elapsed_seconds,
            cost=total_cost or 0,
            pass_rate=pass_rate,
        )
        self._display.stop()

        # Display traces after completion
        if self._show_traces and self._final_traces:
            self._display_traces(self._final_traces)

    def on_logic_map_complete(self, logic_map) -> None:
        """Store full logic map for section rendering."""
        self._display.set_logic_map(logic_map)
        self._display.update_phase("Rules Extracted")
        self._display.add_activity(f"{len(logic_map.rules)} rules found")
        self._display.add_event(f"RULES: Extracted {len(logic_map.rules)} rules from policy")
        self._update_elapsed()

    def on_golden_scenarios_complete(self, scenarios, distribution) -> None:
        """Store full scenarios for section rendering."""
        self._display.set_scenarios(scenarios, distribution)
        self._display.update_phase("Scenarios Generated")
        self._display.add_activity(f"{len(scenarios)} scenarios created")
        self._display.add_event(f"SCEN: Generated {len(scenarios)} golden scenarios")
        self._update_elapsed()

    def on_responses_complete(self, traces: list[Trace]) -> None:
        """Update display with trace count."""
        self._display._state.traces_count = len(traces)
        self._display.update_phase("Traces Complete")
        self._display.add_activity(f"{len(traces)} traces synthesized")
        self._display.add_event(f"DONE: Synthesized {len(traces)} traces")
        self._update_elapsed()

    def on_taxonomy_extracted(self, taxonomy) -> None:
        """Taxonomy extraction is part of coverage phase."""
        self._display.add_activity(f"{len(taxonomy.sub_categories)} sub-categories")
        self._update_elapsed()

    def on_coverage_calculated(self, report) -> None:
        """Store coverage for section rendering - displayed in live panel."""
        # Store coverage data in live display - it will be shown as a section
        self._display.set_coverage(report)
        self._display.add_activity(f"Coverage: {report.overall_coverage_percent:.0f}%")
        self._display.add_event(f"COV: Coverage at {report.overall_coverage_percent:.0f}%")
        self._update_elapsed()

        # Don't print suggestions while Live display is active - they go in the panel
        # Suggestions will be shown after generation completes

    def on_coverage_improved(self, before, after, added_scenarios) -> None:
        """Display coverage improvement."""
        self._display.add_activity(
            f"Coverage: {before.overall_coverage_percent:.0f}% → {after.overall_coverage_percent:.0f}% (+{added_scenarios})"
        )

    def _update_elapsed(self) -> None:
        """Update elapsed time and cost in the display."""
        import time

        if self._start_time:
            elapsed = time.time() - self._start_time
            self._display._state.elapsed_seconds = elapsed

        # Update cost from source if available
        if self._cost_source:
            self._display._state.cost = self._cost_source()

    def _display_traces(self, traces: list[Trace], max_display: int = 5) -> None:
        """Display traces in a readable format (up to max_display)."""
        from rich.panel import Panel

        self.console.print()  # Add spacing after summary

        display_traces = traces[:max_display]
        total = len(traces)

        for idx, trace in enumerate(display_traces, 1):
            # Build trace header
            status = (
                "[green]PASS[/green]" if trace.grade and trace.grade.passed else "[red]FAIL[/red]"
            )
            category = trace.scenario.category or "uncategorized"
            scenario_type = trace.scenario.scenario_type or "unknown"

            self.console.print(
                f"[bold cyan]━━━ Trace {idx}/{total} ━━━[/bold cyan] {status} | {category} | {scenario_type}"
            )

            # User message
            self.console.print("[bold yellow]User:[/bold yellow]")
            self.console.print(Panel(trace.user_message, border_style="yellow", padding=(0, 1)))

            # Assistant response
            self.console.print("[bold green]Assistant:[/bold green]")
            self.console.print(Panel(trace.assistant_message, border_style="green", padding=(0, 1)))

            # Grade feedback if failed
            if trace.grade and not trace.grade.passed and trace.grade.issues:
                self.console.print("[bold red]Issues:[/bold red]")
                for issue in trace.grade.issues:
                    self.console.print(f"  [red]•[/red] {issue}")

            self.console.print()  # Spacing between traces

        # Show message if there are more traces
        if total > max_display:
            self.console.print(
                f"[dim]... and {total - max_display} more traces (use dataset.display() to see all)[/dim]\n"
            )


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
        refinement_calls: int | None = None,
        hitl_calls: int | None = None,
        coverage_calls: int | None = None,
    ) -> None:
        self._emit(
            "complete",
            {
                "dataset_size": dataset_size,
                "elapsed_seconds": elapsed_seconds,
                "pass_rate": pass_rate,
                "total_cost": total_cost,
                "generation_calls": generation_calls,
                "grading_calls": grading_calls,
                "scenario_calls": scenario_calls,
                "response_calls": response_calls,
                "refinement_calls": refinement_calls,
                "hitl_calls": hitl_calls,
                "coverage_calls": coverage_calls,
            },
        )
        if self._on_complete_cb:
            self._on_complete_cb(dataset_size, elapsed_seconds, pass_rate)

    def on_logic_map_complete(self, logic_map) -> None:
        self._emit("logic_map_complete", {"rules_count": len(logic_map.rules)})

    def on_golden_scenarios_complete(self, scenarios, distribution) -> None:
        self._emit(
            "golden_scenarios_complete", {"count": len(scenarios), "distribution": distribution}
        )

    def on_taxonomy_extracted(self, taxonomy) -> None:
        self._emit("taxonomy_extracted", {"sub_categories_count": len(taxonomy.sub_categories)})

    def on_coverage_calculated(self, report) -> None:
        self._emit(
            "coverage_calculated",
            {
                "overall_coverage_percent": report.overall_coverage_percent,
                "covered_count": report.covered_count,
                "partial_count": report.partial_count,
                "uncovered_count": report.uncovered_count,
                "gaps_count": len(report.gaps),
            },
        )

    def on_coverage_improved(self, before, after, added_scenarios) -> None:
        self._emit(
            "coverage_improved",
            {
                "before_percent": before.overall_coverage_percent,
                "after_percent": after.overall_coverage_percent,
                "added_scenarios": added_scenarios,
            },
        )


class FileLoggingReporter:
    """
    Reporter that logs events to a file while delegating to another reporter for display.

    This allows you to have both CLI output (via RichReporter) and file logging simultaneously.
    All events are written to a timestamped log file in a structured format.

    Examples:
        >>> # Log to file while showing rich CLI output
        >>> reporter = FileLoggingReporter()  # Uses RichReporter by default
        >>> pipeline = create_pipeline(reporter=reporter)

        >>> # Custom log directory
        >>> reporter = FileLoggingReporter(log_dir="./logs")

        >>> # Wrap a different reporter
        >>> reporter = FileLoggingReporter(delegate=SilentReporter(), log_dir="./logs")

        >>> # Disable console output entirely (file only)
        >>> reporter = FileLoggingReporter(delegate=SilentReporter())
    """

    def __init__(
        self,
        delegate: "ProgressReporter | None" = None,
        log_dir: str = ".",
        log_filename: str | None = None,
    ):
        """
        Initialize the file logging reporter.

        Args:
            delegate: Reporter to forward events to for display (default: RichReporter)
            log_dir: Directory to write log files (default: current directory)
            log_filename: Custom log filename. If None, uses timestamped name like
                         'synkro_log_2024-01-15_1430.log'
        """
        import os
        from datetime import datetime

        self._delegate = delegate if delegate is not None else RichReporter()
        self._log_dir = log_dir

        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Generate timestamped filename if not provided
        if log_filename is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
            log_filename = f"synkro_log_{timestamp}.log"

        self._log_path = os.path.join(log_dir, log_filename)
        self._start_time: float | None = None

        # Write header to log file
        self._write_log("=== Synkro Generation Log ===")
        self._write_log(f"Started: {datetime.now().isoformat()}")
        self._write_log(f"Log file: {self._log_path}")
        self._write_log("=" * 50)

    def _write_log(self, message: str) -> None:
        """Write a message to the log file."""
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {message}\n")

    def _format_duration(self, seconds: float) -> str:
        """Format seconds into human-readable duration."""
        if seconds >= 60:
            return f"{int(seconds) // 60}m {int(seconds) % 60}s"
        return f"{seconds:.1f}s"

    @property
    def log_path(self) -> str:
        """Return the path to the log file."""
        return self._log_path

    @property
    def display(self):
        """Forward display access to delegate if available (needed for HITL)."""
        if hasattr(self._delegate, "display"):
            return self._delegate.display
        return None

    @property
    def console(self):
        """Forward console access to delegate if available."""
        if hasattr(self._delegate, "console"):
            return self._delegate.console
        return None

    def spinner(self, message: str = "Thinking..."):
        """Forward spinner to delegate."""
        return self._delegate.spinner(message)

    def on_start(self, traces: int, model: str, dataset_type: str) -> None:
        import time

        self._start_time = time.time()
        self._write_log(f"STARTED: Generating {traces} traces")
        self._write_log(f"  Model: {model}")
        self._write_log(f"  Dataset type: {dataset_type}")
        self._delegate.on_start(traces, model, dataset_type)

    def on_plan_complete(self, plan: Plan) -> None:
        self._write_log(f"PLAN COMPLETE: {len(plan.categories)} categories")
        for cat in plan.categories:
            self._write_log(f"  - {cat.name}: {cat.description} (count: {cat.count})")
        self._delegate.on_plan_complete(plan)

    def on_scenario_progress(self, completed: int, total: int) -> None:
        if completed == total or completed % 10 == 0:  # Log every 10 or on completion
            self._write_log(f"SCENARIO PROGRESS: {completed}/{total}")
        self._delegate.on_scenario_progress(completed, total)

    def on_response_progress(self, completed: int, total: int) -> None:
        if completed == total or completed % 10 == 0:  # Log every 10 or on completion
            self._write_log(f"RESPONSE PROGRESS: {completed}/{total}")
        self._delegate.on_response_progress(completed, total)

    def on_responses_complete(self, traces: list[Trace]) -> None:
        self._write_log(f"RESPONSES COMPLETE: {len(traces)} traces generated")

        # Group by category
        by_category: dict[str, list[Trace]] = {}
        for trace in traces:
            cat = trace.scenario.category or "uncategorized"
            by_category.setdefault(cat, []).append(trace)

        for cat_name, cat_traces in by_category.items():
            self._write_log(f"  Category '{cat_name}': {len(cat_traces)} traces")
            for trace in cat_traces:
                scenario_type = getattr(trace.scenario, "scenario_type", "unknown")
                if hasattr(scenario_type, "value"):
                    scenario_type = scenario_type.value
                user_preview = trace.user_message[:80].replace("\n", " ")
                self._write_log(f"    [{scenario_type}] {user_preview}")

        self._delegate.on_responses_complete(traces)

    def on_grading_progress(self, completed: int, total: int) -> None:
        if completed == total or completed % 10 == 0:
            self._write_log(f"GRADING PROGRESS: {completed}/{total}")
        self._delegate.on_grading_progress(completed, total)

    def on_grading_complete(self, traces: list[Trace], pass_rate: float) -> None:
        self._write_log(f"GRADING COMPLETE: {pass_rate:.1f}% passed")
        passed = sum(1 for t in traces if t.grade and t.grade.passed)
        failed = len(traces) - passed
        self._write_log(f"  Passed: {passed}, Failed: {failed}")

        for idx, trace in enumerate(traces, 1):
            status = "PASS" if (trace.grade and trace.grade.passed) else "FAIL"
            scenario_preview = trace.scenario.description[:60].replace("\n", " ")
            self._write_log(f"  #{idx} [{status}] {scenario_preview}")
            if trace.grade and not trace.grade.passed and trace.grade.issues:
                for issue in trace.grade.issues[:3]:
                    self._write_log(f"       Issue: {issue}")

        self._delegate.on_grading_complete(traces, pass_rate)

    def on_refinement_start(self, iteration: int, failed_count: int) -> None:
        self._write_log(
            f"REFINEMENT: Starting iteration {iteration} for {failed_count} failed traces"
        )
        self._delegate.on_refinement_start(iteration, failed_count)

    def on_grading_skipped(self) -> None:
        self._write_log("GRADING: Skipped")
        self._delegate.on_grading_skipped()

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
        refinement_calls: int | None = None,
        hitl_calls: int | None = None,
        coverage_calls: int | None = None,
    ) -> None:
        self._write_log("=" * 50)
        self._write_log(
            f"COMPLETE: Generated {dataset_size} traces in {self._format_duration(elapsed_seconds)}"
        )
        if pass_rate is not None:
            self._write_log(f"  Quality: {pass_rate:.1f}% passed")
        if total_cost is not None and total_cost > 0:
            self._write_log(f"  Cost: ${total_cost:.4f}")
        if scenario_calls is not None:
            self._write_log(f"  Scenario calls: {scenario_calls}")
        if coverage_calls is not None and coverage_calls > 0:
            self._write_log(f"  Coverage calls: {coverage_calls}")
        if hitl_calls is not None and hitl_calls > 0:
            self._write_log(f"  HITL calls: {hitl_calls}")
        if response_calls is not None:
            self._write_log(f"  Response calls: {response_calls}")
        if refinement_calls is not None and refinement_calls > 0:
            self._write_log(f"  Refinement calls: {refinement_calls}")
        if grading_calls is not None:
            self._write_log(f"  Grading calls: {grading_calls}")
        self._write_log(f"Log saved to: {self._log_path}")
        self._write_log("=" * 50)

        self._delegate.on_complete(
            dataset_size,
            elapsed_seconds,
            pass_rate,
            total_cost,
            generation_calls,
            grading_calls,
            scenario_calls,
            response_calls,
            refinement_calls,
            hitl_calls,
            coverage_calls,
        )

        # Print log file location to console
        if hasattr(self._delegate, "console"):
            self._delegate.console.print(f"[dim]📝 Log saved: {self._log_path}[/dim]")

    def on_logic_map_complete(self, logic_map) -> None:
        self._write_log(f"LOGIC MAP: Extracted {len(logic_map.rules)} rules")
        for rule in logic_map.rules:
            rule_type = getattr(rule, "category", "unknown")
            if hasattr(rule_type, "value"):
                rule_type = rule_type.value
            rule_text = getattr(rule, "text", str(rule))[:60]
            self._write_log(f"  [{rule.rule_id}] ({rule_type}) {rule_text}")
        self._delegate.on_logic_map_complete(logic_map)

    def on_golden_scenarios_complete(self, scenarios, distribution) -> None:
        self._write_log(f"SCENARIOS: Generated {len(scenarios)} golden scenarios")
        self._write_log(f"  Distribution: {distribution}")
        for scenario in scenarios:
            scenario_type = getattr(scenario, "scenario_type", "unknown")
            if hasattr(scenario_type, "value"):
                scenario_type = scenario_type.value
            preview = (
                scenario.user_message[:60].replace("\n", " ")
                if hasattr(scenario, "user_message")
                else str(scenario)[:60]
            )
            self._write_log(f"  [{scenario_type}] {preview}")
        self._delegate.on_golden_scenarios_complete(scenarios, distribution)

    def on_taxonomy_extracted(self, taxonomy) -> None:
        self._write_log(f"TAXONOMY: Extracted {len(taxonomy.sub_categories)} sub-categories")
        self._delegate.on_taxonomy_extracted(taxonomy)

    def on_coverage_calculated(self, report) -> None:
        self._write_log(f"COVERAGE: {report.overall_coverage_percent:.0f}% overall")
        self._write_log(
            f"  Covered: {report.covered_count}, Partial: {report.partial_count}, Uncovered: {report.uncovered_count}"
        )
        self._delegate.on_coverage_calculated(report)

    def on_coverage_improved(self, before, after, added_scenarios) -> None:
        self._write_log(
            f"COVERAGE IMPROVED: {before.overall_coverage_percent:.0f}% → {after.overall_coverage_percent:.0f}% (+{added_scenarios} scenarios)"
        )
        self._delegate.on_coverage_improved(before, after, added_scenarios)


__all__ = [
    "ProgressReporter",
    "SilentReporter",
    "RichReporter",
    "CallbackReporter",
    "FileLoggingReporter",
]
