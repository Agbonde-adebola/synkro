"""Live-updating display using Rich's Live component.

Provides a polished, single-panel UI that updates in-place with spinners,
colors, styled progress bars, and indicators.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from synkro.types.coverage import CoverageReport
    from synkro.types.logic_map import GoldenScenario, LogicMap


@dataclass
class DisplayState:
    """State for the live display."""

    phase: str = "IDLE"
    phase_message: str = ""
    progress_current: int = 0
    progress_total: int = 0
    latest_activity: str = ""
    elapsed_seconds: float = 0.0
    cost: float = 0.0
    model: str = ""
    is_complete: bool = False

    # Summary counts
    rules_count: int = 0
    scenarios_count: int = 0
    traces_count: int = 0
    pass_rate: float | None = None

    # Type distribution for scenarios/traces
    positive_count: int = 0
    negative_count: int = 0
    edge_count: int = 0
    irrelevant_count: int = 0

    # Collected IDs for display
    rule_ids: list[str] = field(default_factory=list)
    output_file: str = ""

    # Full data for section rendering
    logic_map: "LogicMap | None" = None
    coverage_percent: float | None = None
    coverage_sub_categories: int = 0
    covered_count: int = 0
    partial_count: int = 0
    uncovered_count: int = 0

    # Event log for activity feed
    events: list[str] = field(default_factory=list)


class _NoOpContextManager:
    """No-op context manager for HITL spinner."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class LiveProgressDisplay:
    """Polished live-updating display with spinners, colors, and progress bars."""

    # Braille spinner frames for smooth animation
    SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self) -> None:
        self.console = Console()
        self._live: Live | None = None
        self._state = DisplayState()
        self._hitl_mode = False
        self._start_time: float | None = None
        self._frame_idx = 0
        self._is_active = False  # Track if display is in active mode

    @property
    def is_active(self) -> bool:
        """Check if the live display is currently active (should suppress external prints)."""
        return self._is_active and not self._hitl_mode

    @property
    def state(self) -> DisplayState:
        """Get the current display state."""
        return self._state

    def __rich__(self) -> Panel:
        """Rich protocol method - called by Live on each refresh to get renderable."""
        return self._render()

    def _render(self) -> Panel:
        """Render the current state as a styled grid-based Panel."""
        s = self._state

        # Advance spinner frame
        self._frame_idx += 1

        # Update elapsed time
        if self._start_time and not s.is_complete:
            s.elapsed_seconds = time.time() - self._start_time

        # Build the full layout
        content_parts: list = []

        # Header: subtitle with model name
        if s.model:
            subtitle = Text(f"Creating traces with {s.model}", style="dim")
            content_parts.append(subtitle)
            content_parts.append(Text(""))

        if s.is_complete:
            # Completion view - simple summary
            content_parts.extend(self._render_complete_view())
        else:
            # Active view - grid layout
            content_parts.extend(self._render_active_view())

        return Panel(
            Group(*content_parts),
            title="[bold cyan]SYNKRO[/bold cyan]",
            subtitle=self._render_status_bar(),
            border_style="green" if s.is_complete else "cyan",
            padding=(0, 1),
        )

    def _render_active_view(self) -> list:
        """Render the active (non-complete) view with grid layout."""
        s = self._state
        content_parts: list = []

        # Create main grid table (2 columns) - Status on RIGHT, Content on LEFT
        grid = Table.grid(padding=(0, 2))
        grid.add_column("left", ratio=3)  # Rules + Scenarios + Coverage
        grid.add_column("right", ratio=1)  # Status box (prominent)

        # Build left column content (Rules + Scenarios + Coverage combined)
        left_content = self._render_content_column()

        # Build right column content (Status box - bigger and more visible)
        right_content = self._render_status_box()

        grid.add_row(left_content, right_content)
        content_parts.append(grid)

        # Events section (full width)
        if s.events:
            content_parts.append(Text(""))
            content_parts.append(self._render_events())

        return content_parts

    def _render_content_column(self) -> Group:
        """Render Rules + Scenarios + Coverage as a single unified content area."""
        s = self._state
        lines: list = []

        # Rules section
        if s.logic_map and s.rules_count > 0:
            rules_header = Text(f"─── Rules ({s.rules_count}) ───", style="bold white")
            lines.append(rules_header)

            # Group by category
            categories: dict[str, list[str]] = {}
            for rule in s.logic_map.rules:
                cat = rule.category.value if hasattr(rule.category, "value") else str(rule.category)
                categories.setdefault(cat, []).append(rule.rule_id)

            for cat, ids in sorted(categories.items()):
                line = Text()
                line.append(f"{cat} ", style="cyan")
                line.append(f"({len(ids)}): ", style="dim")
                ids_display = ", ".join(ids[:5])
                line.append(ids_display, style="white")
                if len(ids) > 5:
                    line.append(f" (+{len(ids) - 5})", style="dim")
                lines.append(line)

            lines.append(Text(""))

        # Scenarios section
        if s.scenarios_count > 0:
            scen_header = Text("─── Scenarios ───", style="bold white")
            lines.append(scen_header)

            dist1 = Text()
            dist1.append(f"[+] {s.positive_count} positive  ", style="green")
            dist1.append(f"[-] {s.negative_count} negative", style="red")
            lines.append(dist1)

            dist2 = Text()
            dist2.append(f"[!] {s.edge_count} edge  ", style="yellow")
            dist2.append(f"[o] {s.irrelevant_count} irrelevant", style="dim")
            lines.append(dist2)

            lines.append(Text(""))

        # Coverage section (integrated into content, not separate box)
        if s.coverage_percent is not None:
            cov_header = Text("─── Coverage ───", style="bold white")
            lines.append(cov_header)

            cov_style = "green" if s.coverage_percent >= 70 else "yellow"
            cov_line = Text()
            cov_line.append(f"{s.coverage_percent:.0f}%", style=f"bold {cov_style}")
            cov_line.append(" overall  ", style="dim")
            cov_line.append(f"{s.covered_count}", style="green")
            cov_line.append(" covered  ", style="dim")
            cov_line.append(f"{s.partial_count}", style="yellow")
            cov_line.append(" partial  ", style="dim")
            cov_line.append(f"{s.uncovered_count}", style="red")
            cov_line.append(" uncovered", style="dim")
            lines.append(cov_line)

        # If nothing to show yet, display placeholder
        if not lines:
            lines.append(Text("Initializing...", style="dim"))

        return Group(*lines)

    def _render_status_box(self) -> Panel:
        """Render prominent Status box for right column."""
        s = self._state

        # Create status table with bigger, more visible styling
        status_table = Table.grid(padding=(0, 2))
        status_table.add_column("label", style="bold white")
        status_table.add_column("value", style="bold cyan")

        # Phase with emphasis
        phase_style = "bold yellow" if "Awaiting" in s.phase else "bold cyan"
        status_table.add_row("Phase", Text(s.phase, style=phase_style))

        # Progress (if applicable)
        if s.progress_total > 0:
            progress_text = f"{s.progress_current}/{s.progress_total}"
            status_table.add_row("Progress", Text(progress_text, style="bold white"))

        # Elapsed time
        status_table.add_row(
            "Elapsed", Text(self._format_time(s.elapsed_seconds), style="bold white")
        )

        # Cost
        status_table.add_row("Cost", Text(f"${s.cost:.4f}", style="bold white"))

        return Panel(
            status_table,
            title="[bold cyan]STATUS[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )

    def _render_events(self) -> Panel:
        """Render scrolling events log."""
        s = self._state
        # Keep last 4 events
        recent_events = s.events[-4:] if s.events else []
        lines = [Text(e, style="dim") for e in recent_events]

        if not lines:
            lines = [Text("No events yet...", style="dim")]

        return Panel(
            Group(*lines),
            title="[dim]Events[/dim]",
            border_style="dim",
            padding=(0, 1),
        )

    def _render_status_bar(self) -> Text:
        """Render bottom status bar with spinner."""
        s = self._state

        if s.is_complete:
            bar = Text()
            bar.append("✓ ", style="green")
            bar.append("Complete", style="bold green")
            bar.append(f"  {s.traces_count} traces", style="dim")
            bar.append(f"  {self._format_time(s.elapsed_seconds)}", style="dim")
            return bar

        spinner = self.SPINNER_FRAMES[self._frame_idx % len(self.SPINNER_FRAMES)]
        bar = Text()
        bar.append(f"{spinner} ", style="cyan")
        bar.append(s.phase, style="bold")
        if s.progress_total > 0:
            bar.append(f"  {s.progress_current}/{s.progress_total}", style="dim")
        bar.append(f"  {self._format_time(s.elapsed_seconds)}", style="dim")
        return bar

    def _render_complete_view(self) -> list:
        """Render the completion summary view."""
        s = self._state
        lines: list = []

        lines.append(Text(""))

        # Main summary
        summary = Text()
        summary.append(f"Generated {s.traces_count} traces", style="bold white")
        summary.append(f" in {self._format_time(s.elapsed_seconds)}", style="dim")
        lines.append(summary)

        lines.append(Text(""))

        # Breakdown
        lines.append(Text(f"├─ {s.rules_count} rules extracted", style="dim"))

        dist_str = (
            f"({s.positive_count}+ {s.negative_count}- {s.edge_count}! {s.irrelevant_count}o)"
        )
        lines.append(Text(f"├─ {s.scenarios_count} scenarios {dist_str}", style="dim"))

        lines.append(Text(f"├─ {s.traces_count} traces synthesized", style="dim"))

        if s.pass_rate is not None:
            rate_style = "green" if s.pass_rate >= 80 else "yellow" if s.pass_rate >= 50 else "red"
            rate_line = Text("└─ ")
            rate_line.append(f"{s.pass_rate:.0f}%", style=rate_style)
            rate_line.append(" passed verification", style="dim")
            lines.append(rate_line)
        else:
            lines.append(Text(f"└─ Cost: ${s.cost:.4f}", style="dim"))

        if s.output_file:
            lines.append(Text(""))
            lines.append(Text(f"Output: {s.output_file}", style="cyan"))

        lines.append(Text(""))

        return lines

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time."""
        if seconds >= 60:
            return f"{int(seconds) // 60}m {int(seconds) % 60}s"
        return f"{seconds:.0f}s"

    def _render_input_section(self) -> Panel:
        """Render the commands and input hint section."""
        lines: list = []

        # Commands row
        cmd_line = Text()
        cmd_line.append("Commands: ", style="dim")
        cmd_line.append("done", style="cyan")
        cmd_line.append(" · ", style="dim")
        cmd_line.append("undo", style="cyan")
        cmd_line.append(" · ", style="dim")
        cmd_line.append("reset", style="cyan")
        cmd_line.append(" · ", style="dim")
        cmd_line.append("show rules", style="cyan")
        cmd_line.append(" · ", style="dim")
        cmd_line.append("show coverage", style="cyan")
        cmd_line.append(" · ", style="dim")
        cmd_line.append("help", style="cyan")
        lines.append(cmd_line)

        # Feedback examples
        feedback_line = Text()
        feedback_line.append("Feedback: ", style="dim")
        feedback_line.append(
            '"add rule for..." "remove R005" "improve coverage" "shorter"', style="yellow"
        )
        lines.append(feedback_line)

        return Panel(
            Group(*lines),
            title="[dim]Input[/dim]",
            border_style="dim",
            padding=(0, 1),
        )

    def _render_with_input(self) -> Group:
        """Render the full panel with input section below."""
        main_panel = self._render()
        input_section = self._render_input_section()
        return Group(main_panel, input_section)

    def prompt_for_input(self, prompt: str = "synkro> ") -> str:
        """
        Pause the live display, show panel with input section, and get user input.

        Returns the user's input string.
        """
        # Stop live display if running
        if self._live:
            self._live.stop()
            self._live = None

        # Render the panel with input section
        self.console.print(self._render_with_input())

        # Get input
        try:
            user_input = self.console.input(f"[cyan]{prompt}[/cyan]")
        except (KeyboardInterrupt, EOFError):
            user_input = "done"
            self.console.print()

        return user_input.strip()

    def resume_live(self) -> None:
        """Resume the live display after input."""
        if not self._live and not self._hitl_mode:
            self._live = Live(
                self,  # Pass self - Rich calls __rich__() on each refresh
                console=self.console,
                refresh_per_second=10,
                transient=True,
            )
            self._live.start()

    def start(self, model: str = "") -> None:
        """Start the live display with auto-animating spinner."""
        self._state = DisplayState(model=model)
        self._start_time = time.time()
        self._frame_idx = 0
        self._is_active = True  # Mark as active
        # Pass self - Rich calls __rich__() on each refresh for animation
        self._live = Live(
            self,  # Rich calls __rich__() on each refresh
            console=self.console,
            refresh_per_second=10,  # Higher rate for smooth spinner
            transient=True,  # Replace in place, don't stack
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display and print final panel."""
        self._is_active = False  # Mark as inactive
        if self._live:
            self._live.stop()
            self._live = None
            # Print final panel since transient=True clears it
            self.console.print(self._render())

    def update_phase(self, phase: str, message: str = "") -> None:
        """Update the current phase."""
        self._state.phase = phase
        self._state.phase_message = message
        self._state.progress_current = 0
        self._state.progress_total = 0
        self._refresh()

    def update_progress(self, current: int, total: int) -> None:
        """Update progress within the current phase."""
        self._state.progress_current = current
        self._state.progress_total = total
        self._refresh()

    def add_activity(self, text: str) -> None:
        """Add latest activity message."""
        self._state.latest_activity = text
        self._refresh()

    def add_rule(self, rule_id: str) -> None:
        """Add a discovered rule ID."""
        if rule_id not in self._state.rule_ids:
            self._state.rule_ids.append(rule_id)
        self._state.rules_count = len(self._state.rule_ids)
        self._refresh()

    def update_distribution(
        self,
        positive: int = 0,
        negative: int = 0,
        edge: int = 0,
        irrelevant: int = 0,
    ) -> None:
        """Update scenario type distribution."""
        self._state.positive_count = positive
        self._state.negative_count = negative
        self._state.edge_count = edge
        self._state.irrelevant_count = irrelevant
        self._refresh()

    def update_metrics(self, elapsed: float, cost: float) -> None:
        """Update elapsed time and cost."""
        self._state.elapsed_seconds = elapsed
        self._state.cost = cost
        self._refresh()

    def set_complete(
        self,
        traces_count: int,
        elapsed_seconds: float,
        cost: float,
        pass_rate: float | None = None,
        output_file: str = "",
    ) -> None:
        """Mark the display as complete with final summary."""
        self._state.is_complete = True
        self._state.traces_count = traces_count
        self._state.elapsed_seconds = elapsed_seconds
        self._state.cost = cost
        self._state.pass_rate = pass_rate
        self._state.output_file = output_file
        self._refresh()

    def set_logic_map(self, logic_map: "LogicMap") -> None:
        """Store logic map for section rendering."""
        self._state.logic_map = logic_map
        self._state.rules_count = len(logic_map.rules)
        self._state.rule_ids = [r.rule_id for r in logic_map.rules]
        self._refresh()

    def set_scenarios(self, scenarios: list, distribution: dict) -> None:
        """Store scenarios for section rendering."""
        self._state.scenarios_count = len(scenarios)
        self._state.positive_count = distribution.get("positive", 0)
        self._state.negative_count = distribution.get("negative", 0)
        self._state.edge_count = distribution.get("edge_case", 0)
        self._state.irrelevant_count = distribution.get("irrelevant", 0)
        self._refresh()

    def set_coverage(self, report: "CoverageReport") -> None:
        """Store coverage for section rendering."""
        self._state.coverage_percent = report.overall_coverage_percent
        self._state.coverage_sub_categories = len(report.sub_category_coverage)
        self._state.covered_count = report.covered_count
        self._state.partial_count = report.partial_count
        self._state.uncovered_count = report.uncovered_count
        self._refresh()

    def add_event(self, event: str) -> None:
        """Add an event to the scrolling log."""
        self._state.events.append(event)
        # Keep last 10 events
        if len(self._state.events) > 10:
            self._state.events = self._state.events[-10:]
        self._refresh()

    def _refresh(self) -> None:
        """Refresh the live display - triggers re-render of the callable."""
        if self._live and not self._hitl_mode:
            # Use refresh() to trigger re-render, NOT update() which replaces the callable
            self._live.refresh()

    # =========================================================================
    # HITL Mode Methods
    # =========================================================================

    def enter_hitl_mode(self) -> None:
        """Pause live display and switch to HITL layout."""
        if self._live:
            self._live.stop()
            self._live = None
        self._hitl_mode = True

    def exit_hitl_mode(self) -> None:
        """Resume live display after HITL."""
        self._hitl_mode = False
        self._frame_idx = 0
        self._live = Live(
            self,  # Rich calls __rich__() on each refresh for animation
            console=self.console,
            refresh_per_second=10,
            transient=True,
        )
        self._live.start()

    def hitl_spinner(self, message: str):
        """Show spinner during HITL operations."""
        from rich.status import Status

        return Status(f"[cyan]{message}[/cyan]", spinner="dots", console=self.console)

    def render_hitl_state(
        self,
        logic_map: "LogicMap",
        scenarios: list["GoldenScenario"],
        coverage: "CoverageReport | None",
        current_turns: int,
        complexity_level: str = "medium",
    ) -> None:
        """Render HITL state as ONE consolidated panel with all sections."""

        content_lines = []

        # =====================================================================
        # HEADER
        # =====================================================================
        header = Text()
        header.append("  SYNKRO HITL", style="bold cyan")
        header.append(" " * 30, style="")
        header.append(self._state.model or "", style="dim")
        content_lines.append(header)
        content_lines.append(Text(""))

        # =====================================================================
        # RULES SECTION
        # =====================================================================
        content_lines.append(Text("  --- Rules ---", style="bold white"))

        # Group rules by category
        rules_by_category: dict[str, list] = {}
        for rule in logic_map.rules:
            cat = rule.category.value if hasattr(rule.category, "value") else str(rule.category)
            rules_by_category.setdefault(cat, []).append(rule)

        # Show categories with counts
        categories = sorted(rules_by_category.items(), key=lambda x: -len(x[1]))
        for cat_name, rules in categories[:4]:
            content_lines.append(Text(f"    {cat_name}: {len(rules)} rules", style="dim"))
        if len(categories) > 4:
            content_lines.append(
                Text(f"    ... +{len(categories) - 4} more categories", style="dim")
            )
        content_lines.append(Text(f"    Total: {len(logic_map.rules)} rules", style="cyan"))
        content_lines.append(Text(""))

        # =====================================================================
        # SCENARIOS SECTION
        # =====================================================================
        if scenarios:
            content_lines.append(Text("  --- Scenarios ---", style="bold white"))

            # Calculate distribution
            dist: dict[str, int] = {}
            for s in scenarios:
                t = (
                    s.scenario_type.value
                    if hasattr(s.scenario_type, "value")
                    else str(s.scenario_type)
                )
                dist[t] = dist.get(t, 0) + 1

            dist_line = Text("    ")
            if dist.get("positive", 0):
                dist_line.append(f"{dist.get('positive', 0)} positive", style="green")
                dist_line.append("  ", style="")
            if dist.get("negative", 0):
                dist_line.append(f"{dist.get('negative', 0)} negative", style="red")
                dist_line.append("  ", style="")
            if dist.get("edge_case", 0):
                dist_line.append(f"{dist.get('edge_case', 0)} edge_case", style="yellow")
                dist_line.append("  ", style="")
            if dist.get("irrelevant", 0):
                dist_line.append(f"{dist.get('irrelevant', 0)} irrelevant", style="dim")
            content_lines.append(dist_line)
            content_lines.append(Text(f"    Total: {len(scenarios)} scenarios", style="cyan"))
            content_lines.append(Text(""))

        # =====================================================================
        # COVERAGE SECTION
        # =====================================================================
        if coverage:
            content_lines.append(Text("  --- Coverage ---", style="bold white"))
            cov_style = (
                "green"
                if coverage.overall_coverage_percent >= 80
                else "yellow"
                if coverage.overall_coverage_percent >= 50
                else "red"
            )
            cov_line = Text("    Overall: ")
            cov_line.append(f"{coverage.overall_coverage_percent:.0f}%", style=cov_style)
            cov_line.append(
                f"  ({coverage.covered_count} covered, {coverage.partial_count} partial, {coverage.uncovered_count} uncovered)",
                style="dim",
            )
            content_lines.append(cov_line)
            if coverage.gaps:
                content_lines.append(Text(f"    Gaps: {len(coverage.gaps)}", style="dim"))
            content_lines.append(Text(""))

        # =====================================================================
        # SETTINGS SECTION
        # =====================================================================
        content_lines.append(Text("  --- Settings ---", style="bold white"))
        settings_line = Text("    ")
        settings_line.append("Complexity: ", style="dim")
        settings_line.append(complexity_level.title(), style="cyan")
        settings_line.append("    Turns: ", style="dim")
        settings_line.append(str(current_turns), style="cyan")
        content_lines.append(settings_line)
        content_lines.append(Text(""))

        # =====================================================================
        # COMMANDS SECTION
        # =====================================================================
        content_lines.append(Text("  --- Commands ---", style="bold white"))
        cmd_line = Text("    ")
        cmd_line.append("done", style="cyan")
        cmd_line.append(" | ", style="dim")
        cmd_line.append("undo", style="cyan")
        cmd_line.append(" | ", style="dim")
        cmd_line.append("reset", style="cyan")
        cmd_line.append(" | ", style="dim")
        cmd_line.append("show R001", style="cyan")
        cmd_line.append(" | ", style="dim")
        cmd_line.append("show S3", style="cyan")
        cmd_line.append(" | ", style="dim")
        cmd_line.append("help", style="cyan")
        content_lines.append(cmd_line)
        content_lines.append(Text(""))

        # =====================================================================
        # FEEDBACK SECTION
        # =====================================================================
        content_lines.append(Text("  --- Feedback Examples ---", style="bold white"))
        content_lines.append(
            Text('    "shorter" "5 turns" "remove R005" "add rule for..."', style="yellow")
        )
        content_lines.append(
            Text('    "add scenario for..." "delete S3" "improve coverage"', style="yellow")
        )
        content_lines.append(Text(""))

        # Build and print the panel
        panel = Panel(
            Group(*content_lines),
            title="[bold cyan]Interactive Session[/bold cyan]",
            border_style="cyan",
            padding=(0, 1),
        )

        self.console.print(panel)

    def hitl_get_input(
        self,
        logic_map: "LogicMap",
        scenarios: list["GoldenScenario"],
        coverage: "CoverageReport | None",
        current_turns: int,
        prompt: str = "synkro> ",
    ) -> str:
        """
        Clear screen, render HITL state, and get user input.

        This is the main entry point for HITL interaction - it combines
        rendering and input into a single clean flow to prevent panel stacking.
        """
        # Clear console to prevent stacking
        self.console.clear()

        # Render the current state
        self.render_hitl_state(logic_map, scenarios, coverage, current_turns)

        # Get input
        try:
            user_input = self.console.input(f"[cyan]{prompt}[/cyan]")
        except (KeyboardInterrupt, EOFError):
            user_input = "done"
            self.console.print()

        return user_input.strip()

    def render_paginated_list(
        self,
        title: str,
        items: list[tuple[str, str]],
        page: int = 1,
        per_page: int = 20,
    ) -> None:
        """Render paginated list with navigation."""
        total_pages = max(1, (len(items) + per_page - 1) // per_page)
        page = max(1, min(page, total_pages))

        start = (page - 1) * per_page
        end = start + per_page
        page_items = items[start:end]

        content_lines: list[Text] = []
        content_lines.append(Text(""))

        for item_id, description in page_items:
            line = Text("  ")
            line.append(item_id, style="cyan")
            line.append(f"  {description[:60]}", style="white")
            if len(description) > 60:
                line.append("...", style="dim")
            content_lines.append(line)

        content_lines.append(Text(""))

        footer = Text("  ")
        footer.append(f"Page {page}/{total_pages} ({per_page} per page)", style="dim")
        footer.append(" | ", style="dim")
        footer.append("n", style="cyan")
        footer.append(" next | ", style="dim")
        footer.append("p", style="cyan")
        footer.append(" prev | ", style="dim")
        footer.append("q", style="cyan")
        footer.append(" back", style="dim")
        content_lines.append(footer)

        panel = Panel(
            Group(*content_lines),
            title=f"[bold]{title} ({len(items)})[/bold]",
            border_style="cyan",
            padding=(0, 1),
        )

        self.console.print(panel)

    def render_rule_detail(
        self,
        rule_id: str,
        logic_map: "LogicMap",
        tested_by: list[str] | None = None,
    ) -> None:
        """Render single rule detail view."""
        rule = logic_map.get_rule(rule_id)
        if not rule:
            self.console.print(f"[red]Rule {rule_id} not found[/red]")
            return

        content_lines: list[Text] = []
        content_lines.append(Text(""))
        content_lines.append(Text(f"  ID:         {rule.rule_id}", style="white"))

        cat = rule.category.value if hasattr(rule.category, "value") else str(rule.category)
        content_lines.append(Text(f"  Category:   {cat}", style="white"))

        text_lines = [rule.text[i : i + 60] for i in range(0, len(rule.text), 60)]
        content_lines.append(Text(f"  Text:       {text_lines[0]}", style="white"))
        for line in text_lines[1:]:
            content_lines.append(Text(f"              {line}", style="white"))

        if rule.condition:
            content_lines.append(Text(f"  Condition:  {rule.condition}", style="dim"))
        if rule.action:
            content_lines.append(Text(f"  Action:     {rule.action}", style="dim"))
        if rule.dependencies:
            content_lines.append(Text(f"  Depends on: {', '.join(rule.dependencies)}", style="dim"))

        if tested_by:
            content_lines.append(Text(""))
            content_lines.append(Text(f"  Tested by:  {', '.join(tested_by[:6])}", style="dim"))
            if len(tested_by) > 6:
                content_lines.append(Text(f"              +{len(tested_by) - 6} more", style="dim"))

        content_lines.append(Text(""))

        panel = Panel(
            Group(*content_lines),
            title=f"[bold]Rule {rule_id}[/bold]",
            border_style="cyan",
            padding=(0, 1),
        )

        self.console.print(panel)

    def render_scenario_detail(
        self,
        scenario_id: str,
        scenarios: list["GoldenScenario"],
    ) -> None:
        """Render single scenario detail view."""
        try:
            idx = int(scenario_id.upper().replace("S", "")) - 1
            if idx < 0 or idx >= len(scenarios):
                self.console.print(
                    f"[red]Scenario {scenario_id} not found (valid: S1-S{len(scenarios)})[/red]"
                )
                return
        except ValueError:
            self.console.print(f"[red]Invalid scenario ID: {scenario_id}[/red]")
            return

        scenario = scenarios[idx]
        content_lines: list[Text] = []
        content_lines.append(Text(""))
        content_lines.append(Text(f"  ID:          S{idx + 1}", style="white"))

        stype = (
            scenario.scenario_type.value
            if hasattr(scenario.scenario_type, "value")
            else str(scenario.scenario_type)
        )
        type_display = stype.replace("_", " ").title()
        content_lines.append(Text(f"  Type:        {type_display}", style="white"))

        desc_lines = [
            scenario.description[i : i + 55] for i in range(0, len(scenario.description), 55)
        ]
        content_lines.append(Text(f"  Description: {desc_lines[0]}", style="white"))
        for line in desc_lines[1:]:
            content_lines.append(Text(f"               {line}", style="white"))

        if scenario.context:
            content_lines.append(Text(f"  Context:     {scenario.context[:55]}", style="dim"))

        if scenario.target_rule_ids:
            content_lines.append(
                Text(f"  Target Rules: {', '.join(scenario.target_rule_ids)}", style="dim")
            )

        if scenario.expected_outcome:
            exp_lines = [
                scenario.expected_outcome[i : i + 55]
                for i in range(0, len(scenario.expected_outcome), 55)
            ]
            content_lines.append(Text(f"  Expected:    {exp_lines[0]}", style="dim"))
            for line in exp_lines[1:]:
                content_lines.append(Text(f"               {line}", style="dim"))

        content_lines.append(Text(""))

        panel = Panel(
            Group(*content_lines),
            title=f"[bold]Scenario S{idx + 1}[/bold]",
            border_style="green",
            padding=(0, 1),
        )

        self.console.print(panel)

    def handle_show_command(
        self,
        command: str,
        logic_map: "LogicMap",
        scenarios: list["GoldenScenario"] | None,
        coverage: "CoverageReport | None",
    ) -> bool:
        """Parse and handle show/find/filter commands. Returns True if handled."""
        parts = command.lower().split()
        if not parts:
            return False

        if parts[0] == "show":
            if len(parts) == 1:
                return False

            target = parts[1]

            if target == "rules":
                items = [(r.rule_id, r.text) for r in logic_map.rules]
                page = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
                self.render_paginated_list("Rules", items, page)
                return True

            elif target == "scenarios" and scenarios:
                items = [(f"S{i + 1}", s.description) for i, s in enumerate(scenarios)]
                page = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
                self.render_paginated_list("Scenarios", items, page)
                return True

            elif target == "gaps" and coverage:
                items = [(f"G{i + 1}", gap) for i, gap in enumerate(coverage.gaps)]
                if not items:
                    self.console.print("[green]No coverage gaps![/green]")
                else:
                    self.render_paginated_list("Coverage Gaps", items)
                return True

            elif target == "coverage" and coverage:
                from rich.table import Table

                table = Table(show_header=True, header_style="bold cyan", title="Coverage")
                table.add_column("Sub-Category")
                table.add_column("Coverage", justify="right")
                table.add_column("Status")

                for cov in coverage.sub_category_coverage:
                    status_icon = {
                        "covered": "[green]+[/green]",
                        "partial": "[yellow]~[/yellow]",
                        "uncovered": "[red]-[/red]",
                    }.get(cov.coverage_status, "?")

                    table.add_row(
                        cov.sub_category_name,
                        f"{cov.coverage_percent:.0f}% ({cov.scenario_count})",
                        status_icon,
                    )

                table.add_row("", "", "", end_section=True)
                table.add_row(
                    f"[bold]Total ({coverage.covered_count}+ {coverage.partial_count}~ {coverage.uncovered_count}-)[/bold]",
                    f"[bold]{coverage.overall_coverage_percent:.0f}%[/bold]",
                    "",
                )
                self.console.print(table)
                return True

            elif target.upper().startswith("R"):
                self.render_rule_detail(target.upper(), logic_map)
                return True

            elif target.upper().startswith("S") and scenarios:
                self.render_scenario_detail(target.upper(), scenarios)
                return True

        elif parts[0] == "find":
            if len(parts) < 2:
                return False

            query = " ".join(parts[1:]).strip("\"'")

            matching_rules = [
                (r.rule_id, r.text)
                for r in logic_map.rules
                if query.lower() in r.text.lower() or query.lower() in r.rule_id.lower()
            ]

            if matching_rules:
                self.render_paginated_list(f"Rules matching '{query}'", matching_rules)
            else:
                self.console.print(f"[dim]No rules matching '{query}'[/dim]")

            if scenarios:
                matching_scenarios = [
                    (f"S{i + 1}", s.description)
                    for i, s in enumerate(scenarios)
                    if query.lower() in s.description.lower()
                ]
                if matching_scenarios:
                    self.render_paginated_list(f"Scenarios matching '{query}'", matching_scenarios)

            return True

        return False


__all__ = ["LiveProgressDisplay", "DisplayState"]
