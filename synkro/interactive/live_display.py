"""Live-updating display using Rich's Live component.

Provides a polished, single-panel UI that updates in-place with spinners,
colors, styled progress bars, and emoji indicators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
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


class _NoOpContextManager:
    """No-op context manager for HITL spinner."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class LiveProgressDisplay:
    """Polished live-updating display with spinners, colors, and progress bars."""

    def __init__(self) -> None:
        self.console = Console()
        self._live: Live | None = None
        self._state = DisplayState()
        self._spinner_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self._frame_idx = 0
        self._hitl_mode = False

    @property
    def state(self) -> DisplayState:
        """Get the current display state."""
        return self._state

    def _render(self) -> Panel:
        """Render the current state as a styled Panel."""
        s = self._state

        # Build content lines
        content_lines: list[Text] = []

        # Header row: branding + model
        header = Text()
        header.append("  ⚡ SYNKRO", style="bold cyan")
        if s.model:
            # Right-align the model name
            padding = 50 - len(s.model)
            header.append(" " * max(padding, 2), style="")
            header.append(s.model, style="dim")
        content_lines.append(header)
        content_lines.append(Text(""))

        # Phase row with spinner (if not complete)
        phase_row = Text()
        if s.is_complete:
            phase_row.append("  ✅ Complete", style="bold green")
            # Add 100% indicator
            phase_row.append("  " + " " * 30 + "100%", style="dim")
        else:
            spinner = self._spinner_frames[self._frame_idx % len(self._spinner_frames)]
            phase_row.append(f"  {spinner} ", style="cyan")
            phase_row.append(s.phase, style="bold cyan")

            # Progress bar
            if s.progress_total > 0:
                pct = (s.progress_current / s.progress_total) * 100
                filled = int(pct / 5)  # 20 char bar
                bar = "█" * filled + "░" * (20 - filled)
                phase_row.append(
                    f"  {bar}  {s.progress_current}/{s.progress_total} {pct:.0f}%", style="cyan"
                )

        content_lines.append(phase_row)
        content_lines.append(Text(""))

        # Summary line (rule IDs or type distribution)
        if s.rule_ids and not s.is_complete:
            ids = ", ".join(s.rule_ids[-5:])  # Last 5
            if len(s.rule_ids) > 5:
                ids += f" (+{len(s.rule_ids) - 5} more)"
            content_lines.append(Text(f"  📜 Found: {ids}", style="white"))
        elif (
            s.positive_count > 0 or s.negative_count > 0 or s.edge_count > 0
        ) and not s.is_complete:
            dist = Text("  ")
            dist.append("✓", style="green")
            dist.append(f" {s.positive_count} positive  ", style="white")
            dist.append("✗", style="red")
            dist.append(f" {s.negative_count} negative  ", style="white")
            dist.append("⚡", style="yellow")
            dist.append(f" {s.edge_count} edge", style="white")
            content_lines.append(dist)

        # Latest activity
        if s.latest_activity and not s.is_complete:
            content_lines.append(Text(""))
            activity = Text("  ")
            activity.append("Latest: ", style="cyan")
            # Truncate if too long
            activity_text = s.latest_activity[:60]
            if len(s.latest_activity) > 60:
                activity_text += "..."
            activity.append(activity_text, style="white")
            content_lines.append(activity)

        # Completion summary
        if s.is_complete:
            content_lines.append(Text(""))

            # Time summary
            elapsed_str = f"{int(s.elapsed_seconds)}s"
            if s.elapsed_seconds >= 60:
                elapsed_str = f"{int(s.elapsed_seconds) // 60}m {int(s.elapsed_seconds) % 60}s"
            content_lines.append(
                Text(f"  Generated {s.traces_count} traces in {elapsed_str}", style="white")
            )

            # Rules
            content_lines.append(Text(f"  ├─ 📜 {s.rules_count} rules extracted", style="dim"))

            # Scenarios with distribution
            dist_str = (
                f"({s.positive_count}+ {s.negative_count}- {s.edge_count}⚡ {s.irrelevant_count}○)"
            )
            content_lines.append(
                Text(f"  ├─ 📋 {s.scenarios_count} scenarios {dist_str}", style="dim")
            )

            # Traces
            content_lines.append(Text(f"  ├─ ✍️  {s.traces_count} traces synthesized", style="dim"))

            # Pass rate
            if s.pass_rate is not None:
                rate_style = (
                    "green" if s.pass_rate >= 80 else "yellow" if s.pass_rate >= 50 else "red"
                )
                line = Text("  └─ ⚖️  ")
                line.append(f"{s.pass_rate:.0f}%", style=rate_style)
                line.append(" passed verification", style="dim")
                content_lines.append(line)

        content_lines.append(Text(""))

        # Footer: metrics
        footer = Text("  ")
        footer.append(f"⏱ {s.elapsed_seconds:.0f}s", style="dim")
        footer.append("  •  ", style="dim")
        footer.append(f"💰 ${s.cost:.4f}", style="dim")
        if s.output_file:
            footer.append("  •  📁 ", style="dim")
            footer.append(s.output_file, style="cyan")
        elif not s.is_complete:
            footer.append("  •  ", style="dim")
            footer.append("Ctrl+C to cancel", style="dim")
        content_lines.append(footer)

        return Panel(
            Group(*content_lines),
            border_style="green" if s.is_complete else "cyan",
            padding=(0, 1),
        )

    def start(self, model: str = "") -> None:
        """Start the live display."""
        self._state = DisplayState(model=model)
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def update_phase(self, phase: str, message: str = "") -> None:
        """Update the current phase."""
        self._state.phase = phase
        self._state.phase_message = message
        self._frame_idx += 1
        self._refresh()

    def update_progress(self, current: int, total: int) -> None:
        """Update progress within the current phase."""
        self._state.progress_current = current
        self._state.progress_total = total
        self._frame_idx += 1
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

    def _refresh(self) -> None:
        """Refresh the live display."""
        if self._live and not self._hitl_mode:
            self._live.update(self._render())

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
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=8,
            transient=False,
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
    ) -> None:
        """Render HITL state (rules, scenarios, coverage) in compact format."""
        content_lines: list[Text] = []

        # Header
        header = Text()
        header.append("  ⚡ SYNKRO HITL", style="bold cyan")
        content_lines.append(header)
        content_lines.append(Text(""))

        # Two-column layout: Rules | Scenarios
        rules_by_category: dict[str, list] = {}
        for rule in logic_map.rules:
            cat = rule.category.value if hasattr(rule.category, "value") else str(rule.category)
            rules_by_category.setdefault(cat, []).append(rule)

        # Rules summary (left side concept)
        rules_line = Text(f"  📜 Rules ({len(logic_map.rules)})", style="bold white")
        content_lines.append(rules_line)

        # Show up to 4 categories
        categories = sorted(rules_by_category.items(), key=lambda x: -len(x[1]))
        for cat_name, rules in categories[:4]:
            content_lines.append(Text(f"  ├─ {cat_name} ({len(rules)})", style="dim"))
        if len(categories) > 4:
            content_lines.append(Text(f"  └─ +{len(categories) - 4} more categories", style="dim"))

        content_lines.append(Text(""))

        # Scenarios summary
        if scenarios:
            dist: dict[str, int] = {}
            for s in scenarios:
                t = (
                    s.scenario_type.value
                    if hasattr(s.scenario_type, "value")
                    else str(s.scenario_type)
                )
                dist[t] = dist.get(t, 0) + 1

            scenarios_line = Text(f"  📋 Scenarios ({len(scenarios)})", style="bold white")
            content_lines.append(scenarios_line)

            # Distribution
            dist_line = Text("  ")
            if dist.get("positive", 0):
                dist_line.append("✓", style="green")
                dist_line.append(f" {dist.get('positive', 0)} positive  ", style="dim")
            if dist.get("negative", 0):
                dist_line.append("✗", style="red")
                dist_line.append(f" {dist.get('negative', 0)} negative  ", style="dim")
            if dist.get("edge_case", 0):
                dist_line.append("⚡", style="yellow")
                dist_line.append(f" {dist.get('edge_case', 0)} edge_case  ", style="dim")
            if dist.get("irrelevant", 0):
                dist_line.append("○", style="dim")
                dist_line.append(f" {dist.get('irrelevant', 0)} irrelevant", style="dim")
            content_lines.append(dist_line)

        content_lines.append(Text(""))

        # Coverage and Turns row
        coverage_line = Text("  ")
        if coverage:
            cov_style = (
                "green"
                if coverage.overall_coverage_percent >= 80
                else "yellow"
                if coverage.overall_coverage_percent >= 50
                else "red"
            )
            coverage_line.append("📊 Coverage: ", style="dim")
            coverage_line.append(f"{coverage.overall_coverage_percent:.0f}%", style=cov_style)
            coverage_line.append(f"  ({len(coverage.gaps)} gaps)  ", style="dim")

        coverage_line.append("⚙️  Turns: ", style="dim")
        coverage_line.append(f"{current_turns}", style="cyan")
        content_lines.append(coverage_line)

        content_lines.append(Text(""))

        # Command hints
        hints = Text("  ")
        hints.append("done", style="cyan")
        hints.append(" · ", style="dim")
        hints.append("undo", style="cyan")
        hints.append(" · ", style="dim")
        hints.append("reset", style="cyan")
        hints.append(" · ", style="dim")
        hints.append("show rules", style="cyan")
        hints.append(" · ", style="dim")
        hints.append("show scenarios", style="cyan")
        hints.append(" · ", style="dim")
        hints.append("show gaps", style="cyan")
        content_lines.append(hints)

        panel = Panel(
            Group(*content_lines),
            border_style="cyan",
            padding=(0, 1),
        )

        self.console.print(panel)

    def render_paginated_list(
        self,
        title: str,
        items: list[tuple[str, str]],  # List of (id, description) tuples
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

        # Pagination footer
        footer = Text("  ")
        footer.append(f"Page {page}/{total_pages} ({per_page} per page)", style="dim")
        footer.append(" · ", style="dim")
        footer.append("n", style="cyan")
        footer.append(" next · ", style="dim")
        footer.append("p", style="cyan")
        footer.append(" prev · ", style="dim")
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

        # Rule details
        content_lines.append(Text(f"  [bold]ID:[/bold]         {rule.rule_id}"))

        cat = rule.category.value if hasattr(rule.category, "value") else str(rule.category)
        content_lines.append(Text(f"  [bold]Category:[/bold]   {cat}"))

        # Wrap text if long
        text_lines = [rule.text[i : i + 60] for i in range(0, len(rule.text), 60)]
        content_lines.append(Text(f"  [bold]Text:[/bold]       {text_lines[0]}"))
        for line in text_lines[1:]:
            content_lines.append(Text(f"               {line}"))

        if rule.condition:
            content_lines.append(Text(f"  [bold]Condition:[/bold]  {rule.condition}"))
        if rule.action:
            content_lines.append(Text(f"  [bold]Action:[/bold]     {rule.action}"))
        if rule.dependencies:
            content_lines.append(Text(f"  [bold]Depends on:[/bold] {', '.join(rule.dependencies)}"))

        if tested_by:
            content_lines.append(Text(""))
            content_lines.append(Text(f"  [bold]Tested by:[/bold]  {', '.join(tested_by[:6])}"))
            if len(tested_by) > 6:
                content_lines.append(Text(f"               +{len(tested_by) - 6} more"))

        content_lines.append(Text(""))

        # Footer
        footer = Text("  ")
        footer.append("q", style="cyan")
        footer.append(" back · ", style="dim")
        footer.append("edit", style="cyan")
        footer.append(" modify · ", style="dim")
        footer.append("delete", style="cyan")
        footer.append(" remove", style="dim")
        content_lines.append(footer)

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
        # Parse S1, S2, etc. to index
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

        # Scenario details
        content_lines.append(Text(f"  [bold]ID:[/bold]           S{idx + 1}"))

        stype = (
            scenario.scenario_type.value
            if hasattr(scenario.scenario_type, "value")
            else str(scenario.scenario_type)
        )
        type_display = stype.replace("_", " ").title()
        content_lines.append(Text(f"  [bold]Type:[/bold]         {type_display}"))

        # Wrap description if long
        desc_lines = [
            scenario.description[i : i + 55] for i in range(0, len(scenario.description), 55)
        ]
        content_lines.append(Text(f"  [bold]Description:[/bold]  {desc_lines[0]}"))
        for line in desc_lines[1:]:
            content_lines.append(Text(f"                  {line}"))

        if scenario.context:
            content_lines.append(Text(f"  [bold]Context:[/bold]      {scenario.context[:55]}"))

        if scenario.target_rule_ids:
            content_lines.append(
                Text(f"  [bold]Target Rules:[/bold] {', '.join(scenario.target_rule_ids)}")
            )

        if scenario.expected_outcome:
            exp_lines = [
                scenario.expected_outcome[i : i + 55]
                for i in range(0, len(scenario.expected_outcome), 55)
            ]
            content_lines.append(Text(f"  [bold]Expected:[/bold]     {exp_lines[0]}"))
            for line in exp_lines[1:]:
                content_lines.append(Text(f"                  {line}"))

        content_lines.append(Text(""))

        # Footer
        footer = Text("  ")
        footer.append("q", style="cyan")
        footer.append(" back · ", style="dim")
        footer.append("edit", style="cyan")
        footer.append(" modify · ", style="dim")
        footer.append("delete", style="cyan")
        footer.append(" remove", style="dim")
        content_lines.append(footer)

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
                return False  # Need subcommand

            target = parts[1]

            if target == "rules":
                # Show all rule categories
                items = [(r.rule_id, r.text) for r in logic_map.rules]
                page = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
                self.render_paginated_list("Rules", items, page)
                return True

            elif target == "scenarios" and scenarios:
                # Show all scenarios
                items = [(f"S{i + 1}", s.description) for i, s in enumerate(scenarios)]
                page = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
                self.render_paginated_list("Scenarios", items, page)
                return True

            elif target == "gaps" and coverage:
                # Show coverage gaps
                items = [(f"G{i + 1}", gap) for i, gap in enumerate(coverage.gaps)]
                if not items:
                    self.console.print("[green]No coverage gaps![/green]")
                else:
                    self.render_paginated_list("Coverage Gaps", items)
                return True

            elif target == "coverage" and coverage:
                # Show full coverage report
                from rich.table import Table

                table = Table(show_header=True, header_style="bold cyan", title="📊 Coverage")
                table.add_column("Sub-Category")
                table.add_column("Coverage", justify="right")
                table.add_column("Status")

                for cov in coverage.sub_category_coverage:
                    status_icon = {
                        "covered": "[green]✓[/green]",
                        "partial": "[yellow]~[/yellow]",
                        "uncovered": "[red]✗[/red]",
                    }.get(cov.coverage_status, "?")

                    table.add_row(
                        cov.sub_category_name,
                        f"{cov.coverage_percent:.0f}% ({cov.scenario_count})",
                        status_icon,
                    )

                table.add_row("", "", "", end_section=True)
                table.add_row(
                    f"[bold]Total ({coverage.covered_count}✓ {coverage.partial_count}~ {coverage.uncovered_count}✗)[/bold]",
                    f"[bold]{coverage.overall_coverage_percent:.0f}%[/bold]",
                    "",
                )
                self.console.print(table)
                return True

            elif target.upper().startswith("R"):
                # Show specific rule
                self.render_rule_detail(target.upper(), logic_map)
                return True

            elif target.upper().startswith("S") and scenarios:
                # Show specific scenario
                self.render_scenario_detail(target.upper(), scenarios)
                return True

        elif parts[0] == "find":
            if len(parts) < 2:
                return False

            # Search query
            query = " ".join(parts[1:]).strip("\"'")

            # Search rules
            matching_rules = [
                (r.rule_id, r.text)
                for r in logic_map.rules
                if query.lower() in r.text.lower() or query.lower() in r.rule_id.lower()
            ]

            if matching_rules:
                self.render_paginated_list(f"Rules matching '{query}'", matching_rules)
            else:
                self.console.print(f"[dim]No rules matching '{query}'[/dim]")

            # Search scenarios if available
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
