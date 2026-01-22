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
from rich.progress_bar import ProgressBar
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
    dataset_type: str = ""  # CONVERSATION, QA, INSTRUCTION, etc.
    traces_target: int = 0  # Target number of traces to generate
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

    # View mode tracking for unified HITL display
    view_mode: str = "main"  # "main", "rules_detail", "scenarios_detail", "logic_map_detail"
    hitl_active: bool = False
    hitl_turns: int = 0
    hitl_complexity: str = "medium"
    detail_page: int = 1
    detail_items_per_page: int = 12

    # Scenarios list for detail views
    scenarios_list: list = field(default_factory=list)
    coverage_report: "CoverageReport | None" = None

    # Original state for diff tracking (green=added, red=removed)
    original_rule_ids: set[str] = field(default_factory=set)
    original_scenario_ids: set[int] = field(default_factory=set)
    original_category_names: set[str] = field(default_factory=set)
    added_rule_ids: set[str] = field(default_factory=set)
    removed_rule_ids: set[str] = field(default_factory=set)
    added_scenario_indices: set[int] = field(default_factory=set)
    removed_scenario_indices: set[int] = field(default_factory=set)
    added_category_names: set[str] = field(default_factory=set)
    removed_category_names: set[str] = field(default_factory=set)

    # Previous coverage for diff display (before improvement)
    previous_coverage_percent: float | None = None
    previous_covered_count: int = 0
    previous_partial_count: int = 0
    previous_uncovered_count: int = 0
    # Maps sub-category ID -> (previous_percent, previous_scenario_count)
    previous_sub_category_coverage: dict[str, tuple[float, int]] = field(default_factory=dict)

    # Error message to display/prefill in input (cleared after use)
    error_message: str = ""


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
        from typing import Callable

        self.console = Console()
        self._live: Live | None = None
        self._state = DisplayState()
        self._hitl_mode = False
        self._start_time: float | None = None
        self._frame_idx = 0
        self._is_active = False  # Track if display is in active mode
        self._cost_source: Callable[[], float] | None = None  # Live cost polling

    def set_cost_source(self, cost_fn) -> None:
        """Set a function to poll for live cost updates."""
        self._cost_source = cost_fn

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

        # Update cost from live source (polled on each render)
        if self._cost_source and not s.is_complete:
            try:
                s.cost = self._cost_source()
            except Exception:
                pass  # Ignore errors from cost polling

        # Dispatch based on view mode
        if s.view_mode == "main":
            if s.hitl_active:
                return self._render_hitl_main()
            elif s.is_complete:
                return self._render_complete()
            else:
                return self._render_active()
        elif s.view_mode == "rules_detail":
            return self._render_rules_detail()
        elif s.view_mode == "scenarios_detail":
            return self._render_scenarios_detail()
        elif s.view_mode == "logic_map_detail":
            return self._render_logic_map_detail()
        elif s.view_mode == "coverage_detail":
            return self._render_coverage_detail()
        elif s.view_mode == "categories_detail":
            return self._render_categories_detail()

        # Fallback to active view
        return self._render_active()

    def _render_active(self) -> Panel:
        """Render the active (non-complete) view as a Panel."""
        content_parts: list = []

        # Active view - content only (status in title)
        content_parts.extend(self._render_active_view())

        # Build title with status info: SYNKRO | Phase | Elapsed | Cost
        title = self._render_title_with_status()

        return Panel(
            Group(*content_parts),
            title=title,
            subtitle=self._render_status_bar(),
            border_style="cyan",
            padding=(0, 1),
        )

    def _render_complete(self) -> Panel:
        """Render the completion view as a Panel."""
        content_parts: list = []

        # Completion view - simple summary
        content_parts.extend(self._render_complete_view())

        # Build title with status info
        title = self._render_title_with_status()

        return Panel(
            Group(*content_parts),
            title=title,
            subtitle=self._render_status_bar(),
            border_style="green",
            padding=(0, 1),
        )

    def _render_hitl_main(self) -> Panel:
        """Render HITL main view with two-column colorful layout."""
        s = self._state

        # Get content and match heights for Row 1: Rules | Scenarios
        rules_content = self._get_rules_content(show_diff=True)
        scenarios_content = self._get_scenarios_content(show_diff=True)
        rules_content, scenarios_content = self._match_content_heights(
            rules_content, scenarios_content
        )

        # Build panels with height-matched content
        rules_panel = self._build_rules_panel(show_diff=True, content=rules_content)
        scenarios_panel = self._build_scenarios_panel(show_diff=True, content=scenarios_content)
        # Row 2 panels don't need height matching - categories uses a Table
        coverage_panel = self._build_coverage_panel()
        categories_panel = self._build_categories_compact_panel()

        # Two-column rows using Table.grid for true equal widths
        row1 = Table.grid(expand=True)
        row1.add_column(ratio=1)
        row1.add_column(ratio=1)
        row1.add_row(rules_panel, scenarios_panel)

        row2 = Table.grid(expand=True)
        row2.add_column(ratio=1)
        row2.add_column(ratio=1)
        row2.add_row(coverage_panel, categories_panel)

        # Subtitle with model name
        subtitle = Text(f"Interactive review with {s.model}", style="dim") if s.model else Text("")

        # Commands footer
        cmd_line = Text()
        cmd_line.append("done", style="bold cyan")
        cmd_line.append(" │ ", style="dim")
        cmd_line.append("show rules", style="cyan")
        cmd_line.append(" │ ", style="dim")
        cmd_line.append("show scenarios", style="cyan")
        cmd_line.append(" │ ", style="dim")
        cmd_line.append("show map", style="cyan")
        cmd_line.append(" │ ", style="dim")
        cmd_line.append("show coverage", style="cyan")
        cmd_line.append(" │ ", style="dim")
        cmd_line.append("undo", style="cyan")
        cmd_line.append(" │ ", style="dim")
        cmd_line.append("help", style="cyan")
        cmd_line.append(" │ ", style="dim")
        cmd_line.append("exit", style="red")

        content = [subtitle, Text(""), row1, row2, Text(""), cmd_line]

        return Panel(
            Group(*content),
            title=self._render_hitl_title(),
            subtitle="[dim]Enter feedback or command[/dim]",
            border_style="cyan",
            padding=(0, 1),
        )

    def _render_hitl_title(self) -> Text:
        """Render title for HITL mode."""
        s = self._state
        title = Text()
        title.append("SYNKRO", style="bold cyan")
        title.append(" Review", style="bold yellow")

        if s.model:
            title.append("  │  ", style="dim")
            model_short = s.model.split("/")[-1]
            title.append(model_short, style="cyan")

        title.append("  │  ", style="dim")
        title.append(self._format_time(s.elapsed_seconds), style="white")

        title.append("  │  ", style="dim")
        title.append(f"${s.cost:.8f}", style="white")

        title.append("  │  ", style="dim")
        title.append(f"{s.hitl_turns} turns", style="cyan")

        return title

    # ─── Panel Builder Helpers (two-column layout) ───────────────────────────

    def _match_content_heights(self, left: list, right: list) -> tuple[list, list]:
        """Pad shorter content list to match the taller one."""
        # Use Text with a single newline to create an empty line that Rich will render
        while len(left) < len(right):
            left.append(Text(" "))
        while len(right) < len(left):
            right.append(Text(" "))
        return left, right

    def _get_rules_content(self, show_diff: bool = False) -> list:
        """Get rules content as list of renderables (without Panel wrapper)."""
        s = self._state

        CATEGORY_COLORS = {
            "constraint": "red",
            "exception": "yellow",
            "permission": "green",
            "procedure": "blue",
            "eligibility": "magenta",
            "requirement": "cyan",
        }

        categories: dict[str, list[str]] = {}
        for rule in s.logic_map.rules:
            cat = rule.category.value if hasattr(rule.category, "value") else str(rule.category)
            categories.setdefault(cat, []).append(rule.rule_id)

        lines: list = []
        for cat, ids in sorted(categories.items()):
            color = CATEGORY_COLORS.get(cat, "white")
            line = Text()
            line.append(f"{cat}", style=color)
            line.append(f"  {len(ids)}", style="bold white")
            lines.append(line)

        return lines

    def _get_rules_title(self, show_diff: bool = False) -> Text:
        """Get rules panel title."""
        s = self._state
        title = Text("Rules", style="bold magenta")
        title.append(f" ({s.rules_count})", style="magenta")
        if show_diff and s.added_rule_ids:
            title.append(f" +{len(s.added_rule_ids)}", style="bold green")
        if show_diff and s.removed_rule_ids:
            title.append(f" -{len(s.removed_rule_ids)}", style="bold red")
        return title

    def _build_rules_panel(self, show_diff: bool = False, content: list | None = None) -> Panel:
        """Build rules panel with category table."""
        if content is None:
            content = self._get_rules_content(show_diff)

        return Panel(
            Group(*content),
            title=self._get_rules_title(show_diff),
            title_align="left",
            border_style="magenta",
            padding=(0, 1),
            expand=True,
        )

    def _get_scenarios_content(self, show_diff: bool = False) -> list:
        """Get scenarios content as list of renderables (without Panel wrapper)."""
        s = self._state
        lines: list = []
        if s.positive_count:
            lines.append(
                Text.assemble(("● ", "bold green"), (f"{s.positive_count} positive", "green"))
            )
        if s.negative_count:
            lines.append(Text.assemble(("● ", "bold red"), (f"{s.negative_count} negative", "red")))
        if s.edge_count:
            lines.append(Text.assemble(("● ", "bold yellow"), (f"{s.edge_count} edge", "yellow")))
        if s.irrelevant_count:
            lines.append(Text.assemble(("● ", "dim"), (f"{s.irrelevant_count} irrelevant", "dim")))
        return lines

    def _get_scenarios_title(self, show_diff: bool = False) -> Text:
        """Get scenarios panel title."""
        s = self._state
        title = Text("Scenarios", style="bold blue")
        title.append(f" ({s.scenarios_count})", style="blue")
        if show_diff and s.added_scenario_indices:
            title.append(f" +{len(s.added_scenario_indices)}", style="bold green")
        if show_diff and s.removed_scenario_indices:
            title.append(f" -{len(s.removed_scenario_indices)}", style="bold red")
        return title

    def _build_scenarios_panel(self, show_diff: bool = False, content: list | None = None) -> Panel:
        """Build scenarios panel with colored bullets."""
        if content is None:
            content = self._get_scenarios_content(show_diff)

        return Panel(
            Group(*content),
            title=self._get_scenarios_title(show_diff),
            title_align="left",
            border_style="blue",
            padding=(0, 1),
            expand=True,
        )

    def _get_coverage_content(self) -> list:
        """Get coverage content as list of renderables (without Panel wrapper)."""
        s = self._state
        color = (
            "green" if s.coverage_percent >= 70 else "yellow" if s.coverage_percent >= 50 else "red"
        )

        bar = ProgressBar(
            total=100, completed=int(s.coverage_percent), width=None, complete_style=color
        )

        lines: list = [bar, Text("")]

        # Helper to format count with diff
        def fmt_count_diff(label: str, current: int, previous: int | None, style: str) -> Text:
            text = Text()
            text.append(f"{label} ", style="dim")
            text.append(str(current), style=style)
            if previous is not None and current != previous:
                diff = current - previous
                diff_style = "bold green" if diff > 0 else "bold red"
                diff_str = f"+{diff}" if diff > 0 else str(diff)
                text.append(f" ({diff_str})", style=diff_style)
            return text

        lines.append(
            fmt_count_diff(
                "covered",
                s.covered_count,
                s.previous_covered_count if s.previous_coverage_percent is not None else None,
                "green",
            )
        )
        lines.append(
            fmt_count_diff(
                "partial",
                s.partial_count,
                s.previous_partial_count if s.previous_coverage_percent is not None else None,
                "yellow",
            )
        )
        lines.append(
            fmt_count_diff(
                "uncovered",
                s.uncovered_count,
                s.previous_uncovered_count if s.previous_coverage_percent is not None else None,
                "red",
            )
        )

        return lines

    def _get_coverage_title(self) -> tuple[Text, str]:
        """Get coverage panel title and border color."""
        s = self._state
        color = (
            "green" if s.coverage_percent >= 70 else "yellow" if s.coverage_percent >= 50 else "red"
        )
        title = Text()
        title.append(f"{s.coverage_percent:.0f}%", style=f"bold {color}")

        # Show diff if we have previous coverage
        if s.previous_coverage_percent is not None:
            diff = s.coverage_percent - s.previous_coverage_percent
            if diff != 0:
                diff_style = "bold green" if diff > 0 else "bold red"
                diff_str = f"+{diff:.0f}%" if diff > 0 else f"{diff:.0f}%"
                title.append(f" ({diff_str})", style=diff_style)

        title.append(" Coverage", style=f"bold {color}")
        return title, color

    def _build_coverage_panel(self, content: list | None = None) -> Panel:
        """Build coverage panel with progress bar and stats."""
        if content is None:
            content = self._get_coverage_content()

        title, color = self._get_coverage_title()

        return Panel(
            Group(*content),
            title=title,
            title_align="left",
            border_style=color,
            padding=(0, 1),
            expand=True,
        )

    def _get_logic_map_compact_content(self) -> list:
        """Get compact logic map content showing rules grouped by category."""
        s = self._state
        lines: list = []

        if not s.logic_map:
            lines.append(Text("No rules available", style="dim"))
            return lines

        CATEGORY_COLORS = {
            "constraint": "red",
            "exception": "yellow",
            "permission": "green",
            "procedure": "blue",
            "eligibility": "magenta",
            "requirement": "cyan",
        }

        # Group rules by category
        by_cat: dict[str, list] = {}
        for rule in s.logic_map.rules:
            cat = rule.category.value if hasattr(rule.category, "value") else str(rule.category)
            by_cat.setdefault(cat, []).append(rule)

        # Show up to 6 rules total, distributed across categories
        shown = 0
        max_rules = 6
        for cat, rules in sorted(by_cat.items()):
            if shown >= max_rules:
                break
            color = CATEGORY_COLORS.get(cat, "white")
            for rule in rules[:2]:  # Max 2 per category
                if shown >= max_rules:
                    break
                line = Text()
                line.append(f"{rule.rule_id}", style=f"bold {color}")
                # Truncate rule text to fit
                text_preview = rule.text[:40] if len(rule.text) <= 40 else rule.text[:37] + "..."
                line.append(f" {text_preview}", style="dim")
                lines.append(line)
                shown += 1

        # Show remaining count if any
        remaining = len(s.logic_map.rules) - shown
        if remaining > 0:
            lines.append(Text(f"... +{remaining} more rules", style="dim"))

        return lines

    def _build_logic_map_compact_panel(self, content: list | None = None) -> Panel:
        """Build compact logic map panel for HITL mode."""
        s = self._state
        if content is None:
            content = self._get_logic_map_compact_content()

        title = Text("Logic Map", style="bold cyan")
        title.append(f" ({s.rules_count})", style="cyan")
        if s.added_rule_ids:
            title.append(f" +{len(s.added_rule_ids)}", style="bold green")
        if s.removed_rule_ids:
            title.append(f" -{len(s.removed_rule_ids)}", style="bold red")

        return Panel(
            Group(*content),
            title=title,
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
            expand=True,
        )

    def _get_categories_compact_content(self) -> Table:
        """Get compact categories content as 2-column table."""
        s = self._state

        # Create 2-column table without headers
        table = Table(show_header=False, box=None, padding=(0, 1), expand=True)
        table.add_column(ratio=1)
        table.add_column(ratio=1)

        if not s.coverage_report or not s.coverage_report.sub_category_coverage:
            table.add_row(Text("No categories", style="dim"), Text(""))
            return table

        # Build list of all items (status icon + name)
        items: list[Text] = []
        for sc in s.coverage_report.sub_category_coverage:
            line = Text()
            # Check if this category was added
            is_added = sc.sub_category_name in s.added_category_names
            # Status icon
            if sc.coverage_status == "covered":
                line.append("✓ ", style="green")
            elif sc.coverage_status == "partial":
                line.append("~ ", style="yellow")
            else:
                line.append("✗ ", style="red")
            # Sub-category name (truncated to fit column)
            name = (
                sc.sub_category_name[:20]
                if len(sc.sub_category_name) > 20
                else sc.sub_category_name
            )
            # Highlight added categories in green
            name_style = "bold green" if is_added else "dim"
            line.append(name, style=name_style)
            items.append(line)

        # Split into 2 columns - fill left column first, then right
        mid = (len(items) + 1) // 2
        left_items = items[:mid]
        right_items = items[mid:]

        # Add rows
        for i in range(max(len(left_items), len(right_items))):
            left = left_items[i] if i < len(left_items) else Text("")
            right = right_items[i] if i < len(right_items) else Text("")
            table.add_row(left, right)

        return table

    def _build_categories_compact_panel(self, content: Table | None = None) -> Panel:
        """Build compact categories panel for HITL mode (4th section)."""
        s = self._state
        if content is None:
            content = self._get_categories_compact_content()

        total = len(s.coverage_report.sub_category_coverage) if s.coverage_report else 0
        title = Text("Categories", style="bold magenta")
        title.append(f" ({total})", style="magenta")
        if s.added_category_names:
            title.append(f" +{len(s.added_category_names)}", style="bold green")
        if s.removed_category_names:
            title.append(f" -{len(s.removed_category_names)}", style="bold red")

        return Panel(
            content,
            title=title,
            title_align="left",
            border_style="magenta",
            padding=(0, 1),
            expand=True,
        )

    def _colorize_event(self, event: str) -> Text:
        """Colorize an event string based on its content."""
        text = Text()
        event_lower = event.lower()

        # Determine icon and color based on event type
        if "error" in event_lower or "failed" in event_lower:
            text.append("✗ ", style="bold red")
            text.append(event, style="red")
        elif "success" in event_lower or "complete" in event_lower or "done" in event_lower:
            text.append("✓ ", style="bold green")
            text.append(event, style="green")
        elif event_lower.startswith("generated"):
            text.append("+ ", style="bold cyan")
            text.append(event, style="cyan")
        elif "rules" in event_lower or event_lower.startswith("rule"):
            text.append("◆ ", style="bold magenta")
            text.append(event, style="magenta")
        elif "scenario" in event_lower:
            text.append("● ", style="bold blue")
            text.append(event, style="blue")
        elif "coverage" in event_lower:
            text.append("▸ ", style="bold yellow")
            text.append(event, style="yellow")
        elif "extract" in event_lower or "parsing" in event_lower:
            text.append("◇ ", style="bold white")
            text.append(event, style="white")
        else:
            text.append("• ", style="dim")
            text.append(event, style="dim white")

        return text

    def _get_events_content(self) -> list:
        """Get events content as list of renderables (without Panel wrapper)."""
        s = self._state
        recent = s.events[-4:] if s.events else []
        if recent:
            lines = [self._colorize_event(e) for e in recent]
        else:
            lines = [Text("• No events yet...", style="dim")]
        return lines

    def _build_events_panel(self, content: list | None = None) -> Panel:
        """Build events panel for active view."""
        if content is None:
            content = self._get_events_content()

        return Panel(
            Group(*content),
            title=Text.assemble(("Events", "bold white")),
            title_align="left",
            border_style="bright_black",
            padding=(0, 1),
            expand=True,
        )

    def _render_rules_detail(self) -> Panel:
        """Render full rules list with pagination using Rich Table."""
        from rich import box

        s = self._state
        content_parts: list = []
        page_info = ""
        total_pages = 1

        if not s.logic_map:
            content_parts.append(Text("No rules available", style="dim"))
        else:
            rules = s.logic_map.rules
            total_pages = max(
                1, (len(rules) + s.detail_items_per_page - 1) // s.detail_items_per_page
            )
            page = max(1, min(s.detail_page, total_pages))

            start = (page - 1) * s.detail_items_per_page
            end = start + s.detail_items_per_page
            page_rules = rules[start:end]

            # Show diff summary if there are changes
            if s.added_rule_ids or s.removed_rule_ids:
                diff_line = Text("Changes: ")
                if s.added_rule_ids:
                    diff_line.append(f"+{len(s.added_rule_ids)} added", style="bold green")
                if s.added_rule_ids and s.removed_rule_ids:
                    diff_line.append("  ", style="")
                if s.removed_rule_ids:
                    diff_line.append(f"-{len(s.removed_rule_ids)} removed", style="bold red")
                content_parts.append(diff_line)
                content_parts.append(Text(""))

            # Create table with full width
            table = Table(
                expand=True,
                box=box.SIMPLE,
                show_header=True,
                header_style="bold cyan",
                padding=(0, 1),
            )
            table.add_column("", width=1)  # Status indicator
            table.add_column("ID", style="cyan", width=6)
            table.add_column("Category", width=12)
            table.add_column("Rule", ratio=1, overflow="fold")

            for rule in page_rules:
                is_added = rule.rule_id in s.added_rule_ids
                status = "[green]+[/green]" if is_added else ""
                id_style = "bold green" if is_added else "cyan"
                text_style = "green" if is_added else "white"
                cat = rule.category.value if hasattr(rule.category, "value") else str(rule.category)

                # Truncate rule text for display
                text_preview = rule.text[:80] + "..." if len(rule.text) > 80 else rule.text

                table.add_row(
                    status,
                    f"[{id_style}]{rule.rule_id}[/{id_style}]",
                    cat,
                    f"[{text_style}]{text_preview}[/{text_style}]",
                )

            content_parts.append(table)
            page_info = f"Page {page}/{total_pages} ({len(rules)} total)"

        title = Text()
        title.append("Rules Detail", style="bold cyan")
        if s.added_rule_ids:
            title.append(f"  (+{len(s.added_rule_ids)})", style="bold green")
        if s.removed_rule_ids:
            title.append(f"  (-{len(s.removed_rule_ids)})", style="bold red")

        # Arrow key navigation hints
        subtitle = Text()
        subtitle.append(page_info if s.logic_map else "", style="dim")
        subtitle.append("  │  ", style="dim")
        subtitle.append("←/→", style="cyan")
        subtitle.append(" navigate  │  ", style="dim")
        subtitle.append("Enter", style="cyan")
        subtitle.append(" return", style="dim")

        return Panel(
            Group(*content_parts),
            title=title,
            subtitle=subtitle,
            border_style="cyan",
            padding=(0, 1),
        )

    def _render_scenarios_detail(self) -> Panel:
        """Render full scenarios list with pagination using Rich Table."""
        from rich import box

        s = self._state
        content_parts: list = []
        scenarios = s.scenarios_list
        page_info = ""
        total_pages = 1

        if not scenarios:
            content_parts.append(Text("No scenarios available", style="dim"))
        else:
            total_pages = max(
                1, (len(scenarios) + s.detail_items_per_page - 1) // s.detail_items_per_page
            )
            page = max(1, min(s.detail_page, total_pages))

            start = (page - 1) * s.detail_items_per_page
            end = start + s.detail_items_per_page
            page_scenarios = scenarios[start:end]

            # Show diff summary if there are changes
            if s.added_scenario_indices or s.removed_scenario_indices:
                diff_line = Text("Changes: ")
                if s.added_scenario_indices:
                    diff_line.append(f"+{len(s.added_scenario_indices)} added", style="bold green")
                if s.added_scenario_indices and s.removed_scenario_indices:
                    diff_line.append("  ", style="")
                if s.removed_scenario_indices:
                    diff_line.append(
                        f"-{len(s.removed_scenario_indices)} removed", style="bold red"
                    )
                content_parts.append(diff_line)
                content_parts.append(Text(""))

            # Create table with full width
            table = Table(
                expand=True,
                box=box.SIMPLE,
                show_header=True,
                header_style="bold cyan",
                padding=(0, 1),
            )
            table.add_column("", width=1)  # Status indicator
            table.add_column("ID", width=4)
            table.add_column("Type", width=10)
            table.add_column("Description", ratio=1, overflow="fold")
            table.add_column("Rules", width=18)

            type_styles = {
                "positive": "green",
                "negative": "red",
                "edge_case": "yellow",
                "irrelevant": "dim",
            }

            for idx, scenario in enumerate(page_scenarios, start=start):
                is_added = idx in s.added_scenario_indices
                status = "[green]+[/green]" if is_added else ""
                id_style = "bold green" if is_added else "cyan"
                text_style = "green" if is_added else "white"

                stype = (
                    scenario.scenario_type.value
                    if hasattr(scenario.scenario_type, "value")
                    else str(scenario.scenario_type)
                )
                type_style = type_styles.get(stype, "white")

                # Truncate description for display
                desc_preview = (
                    scenario.description[:60] + "..."
                    if len(scenario.description) > 60
                    else scenario.description
                )

                # Format rules
                rules_str = ""
                if scenario.target_rule_ids:
                    rules_str = ", ".join(scenario.target_rule_ids[:3])
                    if len(scenario.target_rule_ids) > 3:
                        rules_str += f" +{len(scenario.target_rule_ids) - 3}"

                table.add_row(
                    status,
                    f"[{id_style}]S{idx + 1}[/{id_style}]",
                    f"[{type_style}]{stype}[/{type_style}]",
                    f"[{text_style}]{desc_preview}[/{text_style}]",
                    f"[dim]{rules_str}[/dim]",
                )

            content_parts.append(table)
            page_info = f"Page {page}/{total_pages} ({len(scenarios)} total)"

        title = Text()
        title.append("Scenarios Detail", style="bold cyan")
        if s.added_scenario_indices:
            title.append(f"  (+{len(s.added_scenario_indices)})", style="bold green")
        if s.removed_scenario_indices:
            title.append(f"  (-{len(s.removed_scenario_indices)})", style="bold red")

        # Arrow key navigation hints
        subtitle = Text()
        subtitle.append(page_info, style="dim")
        subtitle.append("  │  ", style="dim")
        subtitle.append("←/→", style="cyan")
        subtitle.append(" navigate  │  ", style="dim")
        subtitle.append("Enter", style="cyan")
        subtitle.append(" return", style="dim")

        return Panel(
            Group(*content_parts),
            title=title,
            subtitle=subtitle,
            border_style="cyan",
            padding=(0, 1),
        )

    def _render_logic_map_detail(self) -> Panel:
        """Render logic map as tree structure by category with diff highlighting."""
        s = self._state
        content_parts: list = []

        if not s.logic_map:
            content_parts.append(Text("No logic map available", style="dim"))
        else:
            # Show diff summary if there are changes
            if s.added_rule_ids or s.removed_rule_ids:
                diff_line = Text("  Changes: ")
                if s.added_rule_ids:
                    diff_line.append(f"+{len(s.added_rule_ids)} added", style="bold green")
                if s.added_rule_ids and s.removed_rule_ids:
                    diff_line.append("  ", style="")
                if s.removed_rule_ids:
                    diff_line.append(f"-{len(s.removed_rule_ids)} removed", style="bold red")
                content_parts.append(diff_line)
                content_parts.append(Text(""))

            # Group rules by category
            by_cat: dict[str, list] = {}
            for rule in s.logic_map.rules:
                cat = rule.category.value if hasattr(rule.category, "value") else str(rule.category)
                by_cat.setdefault(cat, []).append(rule)

            for cat, rules in sorted(by_cat.items()):
                # Category header with count
                cat_line = Text()
                cat_line.append(f"┌─ {cat} ", style="bold cyan")
                cat_line.append(f"({len(rules)} rules)", style="dim")
                content_parts.append(cat_line)

                for i, rule in enumerate(rules):
                    is_last = i == len(rules) - 1
                    prefix = "└──" if is_last else "├──"

                    # Check if this rule was added
                    is_added = rule.rule_id in s.added_rule_ids
                    id_style = "bold green" if is_added else "cyan"
                    text_style = "green" if is_added else "white"
                    prefix_style = "green dim" if is_added else "dim"

                    rule_line = Text()
                    rule_line.append(f"│  {prefix} ", style=prefix_style)
                    rule_line.append(rule.rule_id, style=id_style)
                    text_preview = (
                        rule.text[:48] if len(rule.text) <= 48 else rule.text[:45] + "..."
                    )
                    rule_line.append(f"  {text_preview}", style=text_style)
                    content_parts.append(rule_line)

                content_parts.append(Text(""))

        title = Text()
        title.append("Logic Map", style="bold cyan")
        title.append(f"  ({s.rules_count} rules)", style="dim")
        if s.added_rule_ids:
            title.append(f"  (+{len(s.added_rule_ids)})", style="bold green")
        if s.removed_rule_ids:
            title.append(f"  (-{len(s.removed_rule_ids)})", style="bold red")

        subtitle = Text()
        subtitle.append("Enter", style="cyan")
        subtitle.append(" return", style="dim")

        return Panel(
            Group(*content_parts),
            title=title,
            subtitle=subtitle,
            border_style="cyan",
            padding=(0, 1),
        )

    def _render_coverage_detail(self) -> Panel:
        """Render coverage breakdown using Rich Table with diff tracking."""
        from rich import box

        s = self._state
        content_parts: list = []
        coverage = s.coverage_report

        if not coverage:
            content_parts.append(Text("No coverage data available", style="dim"))
        else:
            # Summary line with diff
            cov_style = (
                "green"
                if coverage.overall_coverage_percent >= 70
                else "yellow"
                if coverage.overall_coverage_percent >= 50
                else "red"
            )
            summary = Text()
            summary.append("Overall: ", style="dim")
            summary.append(f"{coverage.overall_coverage_percent:.0f}%", style=f"bold {cov_style}")

            # Show overall diff if available
            if s.previous_coverage_percent is not None:
                diff = coverage.overall_coverage_percent - s.previous_coverage_percent
                if diff != 0:
                    diff_style = "bold green" if diff > 0 else "bold red"
                    diff_str = f"+{diff:.0f}%" if diff > 0 else f"{diff:.0f}%"
                    summary.append(f" ({diff_str})", style=diff_style)

            summary.append(
                f"  ({coverage.covered_count} covered, {coverage.partial_count} partial, "
                f"{coverage.uncovered_count} uncovered)",
                style="dim",
            )
            content_parts.append(summary)
            content_parts.append(Text(""))

            # Create table with full width
            table = Table(
                expand=True,
                box=box.SIMPLE,
                show_header=True,
                header_style="bold cyan",
                padding=(0, 1),
            )
            table.add_column("Sub-Category", ratio=1)
            table.add_column("Scenarios", width=12, justify="right")
            table.add_column("Coverage", width=14, justify="right")
            table.add_column("Status", width=8, justify="center")

            for cov in coverage.sub_category_coverage:
                status_display = {
                    "covered": "[green]✓ covered[/green]",
                    "partial": "[yellow]~ partial[/yellow]",
                    "uncovered": "[red]✗ uncovered[/red]",
                }.get(cov.coverage_status, "?")

                pct_style = (
                    "green"
                    if cov.coverage_percent >= 70
                    else "yellow"
                    if cov.coverage_percent >= 30
                    else "red"
                )

                # Build scenarios cell with diff
                prev_data = s.previous_sub_category_coverage.get(cov.sub_category_id)
                scenarios_text = Text()
                scenarios_text.append(str(cov.scenario_count))
                if prev_data is not None:
                    _, prev_count = prev_data
                    count_diff = cov.scenario_count - prev_count
                    if count_diff > 0:
                        scenarios_text.append(f" (+{count_diff})", style="bold green")
                    elif count_diff < 0:
                        scenarios_text.append(f" ({count_diff})", style="bold red")

                # Build coverage cell with diff
                coverage_text = Text()
                coverage_text.append(f"{cov.coverage_percent:.0f}%", style=pct_style)
                if prev_data is not None:
                    prev_pct, _ = prev_data
                    pct_diff = cov.coverage_percent - prev_pct
                    if pct_diff > 0:
                        coverage_text.append(f" (+{pct_diff:.0f}%)", style="bold green")
                    elif pct_diff < 0:
                        coverage_text.append(f" ({pct_diff:.0f}%)", style="bold red")

                table.add_row(
                    cov.sub_category_name,
                    scenarios_text,
                    coverage_text,
                    status_display,
                )

            content_parts.append(table)

            # Show gaps if any
            if coverage.gaps:
                content_parts.append(Text(""))
                gaps_header = Text("Gaps: ", style="bold red")
                gaps_header.append(", ".join(coverage.gaps[:5]), style="red")
                if len(coverage.gaps) > 5:
                    gaps_header.append(f" +{len(coverage.gaps) - 5} more", style="dim red")
                content_parts.append(gaps_header)

        title = Text()
        title.append("Coverage Detail", style="bold cyan")

        # Arrow key navigation hints (if paginated in future)
        subtitle = Text()
        subtitle.append("Enter", style="cyan")
        subtitle.append(" return", style="dim")

        return Panel(
            Group(*content_parts),
            title=title,
            subtitle=subtitle,
            border_style="cyan",
            padding=(0, 1),
        )

    def _render_categories_detail(self) -> Panel:
        """Render categories/sub-categories with pagination using Rich Table."""
        from rich import box

        s = self._state
        content_parts: list = []
        coverage = s.coverage_report

        if not coverage or not coverage.sub_category_coverage:
            content_parts.append(Text("No categories available", style="dim"))
        else:
            # Show diff summary if there are changes
            if s.added_category_names or s.removed_category_names:
                diff_line = Text("  Changes: ")
                if s.added_category_names:
                    diff_line.append(f"+{len(s.added_category_names)} added", style="bold green")
                if s.added_category_names and s.removed_category_names:
                    diff_line.append("  ", style="")
                if s.removed_category_names:
                    diff_line.append(f"-{len(s.removed_category_names)} removed", style="bold red")
                content_parts.append(diff_line)
                content_parts.append(Text(""))

            sub_cats = coverage.sub_category_coverage

            # Calculate pagination
            total_pages = max(
                1, (len(sub_cats) + s.detail_items_per_page - 1) // s.detail_items_per_page
            )
            page = max(1, min(s.detail_page, total_pages))
            start = (page - 1) * s.detail_items_per_page
            end = start + s.detail_items_per_page
            page_items = sub_cats[start:end]

            # Create table
            table = Table(
                expand=True,
                box=box.SIMPLE,
                show_header=True,
                header_style="bold magenta",
                padding=(0, 1),
            )
            table.add_column("Category", ratio=1)
            table.add_column("Sub-Category", ratio=2)
            table.add_column("Scenarios", width=10, justify="right")
            table.add_column("Coverage", width=10, justify="right")
            table.add_column("Status", width=8, justify="center")

            for sc in page_items:
                status_display = {
                    "covered": "[green]✓[/green]",
                    "partial": "[yellow]~[/yellow]",
                    "uncovered": "[red]✗[/red]",
                }.get(sc.coverage_status, "?")

                pct_style = (
                    "green"
                    if sc.coverage_percent >= 70
                    else "yellow"
                    if sc.coverage_percent >= 30
                    else "red"
                )

                # Check if this category was added
                is_added = sc.sub_category_name in s.added_category_names
                cat_style = "bold green" if is_added else ""
                subcat_style = "bold green" if is_added else ""

                table.add_row(
                    Text(sc.parent_category or "General", style=cat_style),
                    Text(sc.sub_category_name, style=subcat_style),
                    str(sc.scenario_count),
                    Text(f"{sc.coverage_percent:.0f}%", style=pct_style),
                    status_display,
                )

            content_parts.append(table)

            # Pagination info
            content_parts.append(Text(""))
            page_info = Text()
            page_info.append(
                f"Page {page}/{total_pages} ({len(sub_cats)} sub-categories)", style="dim"
            )
            content_parts.append(page_info)

        title = Text()
        title.append("Categories Detail", style="bold magenta")
        if s.added_category_names:
            title.append(f"  (+{len(s.added_category_names)})", style="bold green")
        if s.removed_category_names:
            title.append(f"  (-{len(s.removed_category_names)})", style="bold red")

        # Navigation hints
        subtitle = Text()
        subtitle.append("←/→", style="magenta")
        subtitle.append(" navigate  ", style="dim")
        subtitle.append("Enter", style="magenta")
        subtitle.append(" return", style="dim")

        return Panel(
            Group(*content_parts),
            title=title,
            subtitle=subtitle,
            border_style="magenta",
            padding=(0, 1),
        )

    def _render_title_with_status(self) -> Text:
        """Render title line with SYNKRO, model, type, traces, and status info."""
        s = self._state
        title = Text()
        title.append("SYNKRO", style="bold cyan")

        # Model name (shortened if needed)
        if s.model:
            title.append("  │  ", style="dim")
            model_short = s.model.split("/")[-1]  # Remove provider prefix if present
            title.append(model_short, style="cyan")

        # Dataset type
        if s.dataset_type:
            title.append("  │  ", style="dim")
            title.append(s.dataset_type, style="magenta")

        # Traces target
        if s.traces_target > 0:
            title.append("  │  ", style="dim")
            title.append(f"{s.traces_target} traces", style="green")

        return title

    def _render_active_view(self) -> list:
        """Render the active (non-complete) view - content only, status is in title."""
        content_parts: list = []

        # Two-column content area (Rules + Scenarios + Coverage + Events)
        content_parts.append(self._render_content_column())

        return content_parts

    def _render_content_column(self) -> Group:
        """Render two-column colorful layout for active view with progressive display."""
        s = self._state
        parts: list = []

        has_rules = s.logic_map and s.rules_count > 0
        has_scenarios = s.scenarios_count > 0
        has_coverage = s.coverage_percent is not None
        has_events = bool(s.events)

        # Row 1: Rules | Scenarios (only if either exists)
        if has_rules or has_scenarios:
            row1 = Table.grid(expand=True)
            if has_rules and has_scenarios:
                # Both exist - match heights
                rules_content = self._get_rules_content(show_diff=False)
                scenarios_content = self._get_scenarios_content(show_diff=False)
                rules_content, scenarios_content = self._match_content_heights(
                    rules_content, scenarios_content
                )
                row1.add_column(ratio=1)
                row1.add_column(ratio=1)
                row1.add_row(
                    self._build_rules_panel(show_diff=False, content=rules_content),
                    self._build_scenarios_panel(show_diff=False, content=scenarios_content),
                )
            elif has_rules:
                row1.add_column(ratio=1)
                row1.add_row(self._build_rules_panel(show_diff=False))
            else:  # has_scenarios only
                row1.add_column(ratio=1)
                row1.add_row(self._build_scenarios_panel(show_diff=False))
            parts.append(row1)

        # Row 2: Coverage | Events (only if either exists)
        if has_coverage or has_events:
            row2 = Table.grid(expand=True)
            if has_coverage and has_events:
                # Both exist - match heights
                coverage_content = self._get_coverage_content()
                events_content = self._get_events_content()
                coverage_content, events_content = self._match_content_heights(
                    coverage_content, events_content
                )
                row2.add_column(ratio=1)
                row2.add_column(ratio=1)
                row2.add_row(
                    self._build_coverage_panel(content=coverage_content),
                    self._build_events_panel(content=events_content),
                )
            elif has_coverage:
                row2.add_column(ratio=1)
                row2.add_row(self._build_coverage_panel())
            else:  # has_events only
                row2.add_column(ratio=1)
                row2.add_row(self._build_events_panel())
            parts.append(row2)

        # Fallback if nothing exists yet
        if not parts:
            parts.append(Text("Initializing...", style="dim"))

        return Group(*parts)

    def _render_status_bar(self) -> Text:
        """Render bottom status bar with spinner, phase, time, and cost."""
        s = self._state

        if s.is_complete:
            bar = Text()
            bar.append("✓ ", style="green")
            bar.append("Complete", style="bold green")
            bar.append(f"  │  {s.traces_count} traces", style="dim")
            bar.append(f"  │  {self._format_time(s.elapsed_seconds)}", style="dim")
            bar.append(f"  │  ${s.cost:.8f}", style="dim")
            return bar

        spinner = self.SPINNER_FRAMES[self._frame_idx % len(self.SPINNER_FRAMES)]
        bar = Text()
        bar.append(f"{spinner} ", style="cyan")
        bar.append(s.phase, style="bold")
        if s.progress_total > 0:
            bar.append(f"  │  {s.progress_current}/{s.progress_total}", style="dim")
        bar.append(f"  │  {self._format_time(s.elapsed_seconds)}", style="dim")
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
            lines.append(Text(f"└─ Cost: ${s.cost:.8f}", style="dim"))

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

    def resume_live(self, force: bool = False) -> None:
        """Resume the live display after input.

        Args:
            force: If True, resume even during HITL mode (for coverage improvement)
        """
        if not self._live and (not self._hitl_mode or force):
            self._live = Live(
                self,  # Pass self - Rich calls __rich__() on each refresh
                console=self.console,
                refresh_per_second=10,
                transient=True,
            )
            self._live.start()

    def start(self, model: str = "", dataset_type: str = "", traces_target: int = 0) -> None:
        """Start the live display with auto-animating spinner."""
        self._state = DisplayState(
            model=model, dataset_type=dataset_type, traces_target=traces_target
        )
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

    def set_error(self, message: str) -> None:
        """Set an error message to display/prefill in the input field."""
        self._state.error_message = message

    def clear_error(self) -> None:
        """Clear any pending error message."""
        self._state.error_message = ""

    def add_event(self, event: str) -> None:
        """Add an event to the scrolling log (bounded to prevent memory growth)."""
        self._state.events.append(event)
        # Keep last 1000 events to prevent unbounded growth in long sessions
        # (UI only shows last 4, but full log useful for debugging)
        if len(self._state.events) > 1000:
            self._state.events = self._state.events[-1000:]
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
        # Reset diff tracking for new HITL session
        self._reset_diff_tracking()

    def _reset_diff_tracking(self) -> None:
        """Reset diff tracking state for a fresh HITL session."""
        self._state.original_rule_ids = set()
        self._state.original_scenario_ids = set()
        self._state.added_rule_ids = set()
        self._state.removed_rule_ids = set()
        self._state.added_scenario_indices = set()
        self._state.removed_scenario_indices = set()

    def exit_hitl_mode(self) -> None:
        """Resume live display after HITL."""
        self._hitl_mode = False
        self._state.hitl_active = False
        self._state.view_mode = "main"
        self._frame_idx = 0
        self.console.clear()  # Clear HITL panel before starting Live
        self._live = Live(
            self,  # Rich calls __rich__() on each refresh for animation
            console=self.console,
            refresh_per_second=10,
            transient=True,
        )
        self._live.start()

    # =========================================================================
    # View Mode Control Methods
    # =========================================================================

    def enter_detail_view(self, mode: str, page: int = 1) -> None:
        """Switch to a detail view mode."""
        self._state.view_mode = mode
        self._state.detail_page = page

    def exit_detail_view(self) -> None:
        """Return to main view."""
        self._state.view_mode = "main"
        self._state.detail_page = 1

    def _update_current_data(
        self,
        logic_map: "LogicMap",
        scenarios: list | None,
        coverage: "CoverageReport | None",
    ) -> None:
        """Update current data and recompute diffs. Called before showing detail views."""
        # Update logic map and rules
        self._state.logic_map = logic_map
        self._state.rules_count = len(logic_map.rules)
        self._state.rule_ids = [r.rule_id for r in logic_map.rules]

        # Recompute rule diffs
        current_rule_ids = {r.rule_id for r in logic_map.rules}
        if self._state.original_rule_ids:
            self._state.added_rule_ids = current_rule_ids - self._state.original_rule_ids
            self._state.removed_rule_ids = self._state.original_rule_ids - current_rule_ids

        # Update scenarios
        self._state.scenarios_list = scenarios or []
        if scenarios:
            self._state.scenarios_count = len(scenarios)
            dist: dict[str, int] = {}
            for s in scenarios:
                t = (
                    s.scenario_type.value
                    if hasattr(s.scenario_type, "value")
                    else str(s.scenario_type)
                )
                dist[t] = dist.get(t, 0) + 1
            self._state.positive_count = dist.get("positive", 0)
            self._state.negative_count = dist.get("negative", 0)
            self._state.edge_count = dist.get("edge_case", 0)
            self._state.irrelevant_count = dist.get("irrelevant", 0)

            # Recompute scenario diffs
            current_count = len(scenarios)
            if self._state.original_scenario_ids:
                original_count = len(self._state.original_scenario_ids)
                if current_count > original_count:
                    self._state.added_scenario_indices = set(range(original_count, current_count))
                else:
                    self._state.added_scenario_indices = set()
                if current_count < original_count:
                    self._state.removed_scenario_indices = set(range(current_count, original_count))
                else:
                    self._state.removed_scenario_indices = set()

        # Update coverage
        self._state.coverage_report = coverage
        if coverage:
            self._state.coverage_percent = coverage.overall_coverage_percent
            self._state.covered_count = coverage.covered_count
            self._state.partial_count = coverage.partial_count
            self._state.uncovered_count = coverage.uncovered_count

            # Recompute category diffs
            current_category_names = {sc.sub_category_name for sc in coverage.sub_category_coverage}
            if self._state.original_category_names:
                self._state.added_category_names = (
                    current_category_names - self._state.original_category_names
                )
                self._state.removed_category_names = (
                    self._state.original_category_names - current_category_names
                )

    def snapshot_coverage(self) -> None:
        """Snapshot current coverage as previous for diff display.

        Call this BEFORE starting coverage improvement to track changes.
        """
        s = self._state
        s.previous_coverage_percent = s.coverage_percent
        s.previous_covered_count = s.covered_count
        s.previous_partial_count = s.partial_count
        s.previous_uncovered_count = s.uncovered_count

        # Snapshot per-sub-category coverage
        if s.coverage_report:
            s.previous_sub_category_coverage = {
                cov.sub_category_id: (cov.coverage_percent, cov.scenario_count)
                for cov in s.coverage_report.sub_category_coverage
            }

    def update_coverage(self, coverage: "CoverageReport") -> None:
        """Update coverage state for live refresh after scenario generation."""
        self._state.coverage_report = coverage
        self._state.coverage_percent = coverage.overall_coverage_percent
        self._state.covered_count = coverage.covered_count
        self._state.partial_count = coverage.partial_count
        self._state.uncovered_count = coverage.uncovered_count

    def set_hitl_state(
        self,
        logic_map: "LogicMap",
        scenarios: list | None,
        coverage: "CoverageReport | None",
        turns: int,
        complexity: str = "medium",
    ) -> None:
        """Set all HITL-related state for unified rendering with diff tracking."""
        self._state.hitl_active = True
        self._state.logic_map = logic_map
        self._state.rules_count = len(logic_map.rules)
        self._state.rule_ids = [r.rule_id for r in logic_map.rules]
        self._state.hitl_turns = turns
        self._state.hitl_complexity = complexity

        # Track original rules for diff highlighting (only set once per session)
        current_rule_ids = {r.rule_id for r in logic_map.rules}
        if not self._state.original_rule_ids:
            # First time - set original state
            self._state.original_rule_ids = current_rule_ids.copy()
            self._state.added_rule_ids = set()
            self._state.removed_rule_ids = set()
        else:
            # Compute diff: added = in current but not in original
            self._state.added_rule_ids = current_rule_ids - self._state.original_rule_ids
            # Removed = in original but not in current
            self._state.removed_rule_ids = self._state.original_rule_ids - current_rule_ids

        # Store scenarios list for detail view
        self._state.scenarios_list = scenarios or []
        if scenarios:
            self._state.scenarios_count = len(scenarios)
            dist: dict[str, int] = {}
            for s in scenarios:
                t = (
                    s.scenario_type.value
                    if hasattr(s.scenario_type, "value")
                    else str(s.scenario_type)
                )
                dist[t] = dist.get(t, 0) + 1
            self._state.positive_count = dist.get("positive", 0)
            self._state.negative_count = dist.get("negative", 0)
            self._state.edge_count = dist.get("edge_case", 0)
            self._state.irrelevant_count = dist.get("irrelevant", 0)

            # Track original scenarios for diff highlighting
            current_scenario_count = len(scenarios)
            if not self._state.original_scenario_ids:
                # First time - set original scenario indices
                self._state.original_scenario_ids = set(range(current_scenario_count))
                self._state.added_scenario_indices = set()
                self._state.removed_scenario_indices = set()
            else:
                # For scenarios, track by index - new ones are indices >= original count
                original_count = len(self._state.original_scenario_ids)
                if current_scenario_count > original_count:
                    self._state.added_scenario_indices = set(
                        range(original_count, current_scenario_count)
                    )
                else:
                    self._state.added_scenario_indices = set()
                # Removed scenarios: original indices no longer present
                if current_scenario_count < original_count:
                    self._state.removed_scenario_indices = set(
                        range(current_scenario_count, original_count)
                    )
                else:
                    self._state.removed_scenario_indices = set()
        else:
            self._state.scenarios_count = 0
            self._state.positive_count = 0
            self._state.negative_count = 0
            self._state.edge_count = 0
            self._state.irrelevant_count = 0

        # Store coverage
        self._state.coverage_report = coverage
        if coverage:
            self._state.coverage_percent = coverage.overall_coverage_percent
            self._state.covered_count = coverage.covered_count
            self._state.partial_count = coverage.partial_count
            self._state.uncovered_count = coverage.uncovered_count

            # Track original categories for diff highlighting
            current_category_names = {sc.sub_category_name for sc in coverage.sub_category_coverage}
            if not self._state.original_category_names:
                # First time - set original state
                self._state.original_category_names = current_category_names.copy()
                self._state.added_category_names = set()
                self._state.removed_category_names = set()
            else:
                # Compute diff
                self._state.added_category_names = (
                    current_category_names - self._state.original_category_names
                )
                self._state.removed_category_names = (
                    self._state.original_category_names - current_category_names
                )
        else:
            self._state.coverage_percent = None

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
        content_lines.append(Text("  --- Scenarios ---", style="bold white"))
        if scenarios:
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
        else:
            content_lines.append(Text("    [dim]No scenarios yet[/dim]", style="dim"))
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

    def _get_total_pages(self) -> int:
        """Get total pages for current detail view."""
        s = self._state
        if s.view_mode == "rules_detail" and s.logic_map:
            return max(
                1, (len(s.logic_map.rules) + s.detail_items_per_page - 1) // s.detail_items_per_page
            )
        elif s.view_mode == "scenarios_detail" and s.scenarios_list:
            return max(
                1, (len(s.scenarios_list) + s.detail_items_per_page - 1) // s.detail_items_per_page
            )
        elif s.view_mode == "categories_detail" and s.coverage_report:
            sub_cats = s.coverage_report.sub_category_coverage or []
            return max(1, (len(sub_cats) + s.detail_items_per_page - 1) // s.detail_items_per_page)
        return 1

    def _get_key_or_line(self, prompt: str = "", prefill: str = "") -> tuple[str, str]:
        """
        Get user input with arrow key support for pagination.

        Args:
            prompt: The prompt to display before input
            prefill: Optional text to pre-populate in the input (e.g., error message)

        Returns:
            Tuple of (input_type, value) where:
            - ("arrow", "left"|"right") for arrow keys
            - ("enter", "") for Enter key alone
            - ("text", "user input") for text followed by Enter
        """
        import sys

        try:
            import readchar
        except ImportError:
            # Fallback: no arrow key support, use standard input
            self.console.print(f"[cyan]{prompt}[/cyan]", end="")
            try:
                return ("text", input().strip())
            except (KeyboardInterrupt, EOFError):
                return ("text", "done")

        # Initialize buffer with prefill if provided
        buffer: list[str] = list(prefill) if prefill else []
        self.console.print(f"[cyan]{prompt}[/cyan]", end="", highlight=False)

        # Display prefill text in red (error) so user can see and delete it
        if prefill:
            sys.stdout.write(f"\033[91m{prefill}\033[0m")  # Red text

        sys.stdout.flush()

        while True:
            try:
                key = readchar.readkey()
            except (KeyboardInterrupt, EOFError):
                print()
                return ("text", "done")

            if key == readchar.key.LEFT:
                return ("arrow", "left")
            elif key == readchar.key.RIGHT:
                return ("arrow", "right")
            elif key in (readchar.key.ENTER, "\r", "\n"):
                print()  # Newline after input
                return ("text", "".join(buffer).strip()) if buffer else ("enter", "")
            elif key == readchar.key.BACKSPACE or key == "\x7f":
                if buffer:
                    buffer.pop()
                    # Visual backspace
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            elif key == readchar.key.CTRL_C:
                print()
                return ("text", "done")
            elif len(key) == 1 and key.isprintable():
                buffer.append(key)
                sys.stdout.write(key)
                sys.stdout.flush()
            # Ignore other special keys

    def hitl_get_input(
        self,
        logic_map: "LogicMap",
        scenarios: list["GoldenScenario"] | None,
        coverage: "CoverageReport | None",
        current_turns: int,
        prompt: str = "synkro> ",
        complexity_level: str = "medium",
    ) -> str:
        """
        Clear screen, render HITL state using unified panel, and get user input.

        This is the main entry point for HITL interaction - it combines
        rendering and input into a single clean flow to prevent panel stacking.
        Handles detail view navigation internally with arrow key support.
        """
        # Set HITL state for unified rendering
        self.set_hitl_state(logic_map, scenarios, coverage, current_turns, complexity_level)

        # Clear console thoroughly - use ANSI escape codes for reliable clearing
        # \033[2J clears entire screen, \033[H moves cursor to home position
        print("\033[2J\033[H", end="", flush=True)
        self.console.print(self._render())

        # Choose prompt based on view mode
        if self._state.view_mode != "main":
            input_prompt = ""  # No prompt in detail view - use arrow keys
        else:
            input_prompt = prompt

        # Get error message to prefill (only in main view) and clear it
        prefill = ""
        if self._state.view_mode == "main" and self._state.error_message:
            prefill = self._state.error_message
            self._state.error_message = ""  # Clear after use

        # Get input with arrow key support (and error prefill)
        input_type, value = self._get_key_or_line(input_prompt, prefill=prefill)

        # Handle arrow keys for pagination in detail views
        if input_type == "arrow" and self._state.view_mode != "main":
            total_pages = self._get_total_pages()
            if value == "left" and self._state.detail_page > 1:
                self._state.detail_page -= 1
            elif value == "right" and self._state.detail_page < total_pages:
                self._state.detail_page += 1
            # Re-render with new page
            return self.hitl_get_input(
                logic_map, scenarios, coverage, current_turns, prompt, complexity_level
            )

        # Handle Enter key in detail view: return to main
        if input_type == "enter" and self._state.view_mode != "main":
            self.exit_detail_view()
            return self.hitl_get_input(
                logic_map, scenarios, coverage, current_turns, prompt, complexity_level
            )

        # Handle arrow keys in main view: ignore and re-prompt
        if input_type == "arrow" and self._state.view_mode == "main":
            return self.hitl_get_input(
                logic_map, scenarios, coverage, current_turns, prompt, complexity_level
            )

        user_input = value

        # Handle text input in detail view
        if self._state.view_mode != "main":
            if user_input.lower() in ("q", "quit", "exit", "back"):
                self.exit_detail_view()
                return self.hitl_get_input(
                    logic_map, scenarios, coverage, current_turns, prompt, complexity_level
                )

            # Legacy page command support (still works)
            if user_input.lower().startswith("page "):
                try:
                    page = int(user_input.split()[1])
                    self._state.detail_page = page
                    return self.hitl_get_input(
                        logic_map, scenarios, coverage, current_turns, prompt, complexity_level
                    )
                except (IndexError, ValueError):
                    pass

            # Any other input in detail view: exit detail view and return as command
            self.exit_detail_view()

        return user_input

    def handle_show_command(
        self,
        command: str,
        logic_map: "LogicMap",
        scenarios: list["GoldenScenario"] | None,
        coverage: "CoverageReport | None",
    ) -> bool:
        """Parse and handle show/find/filter commands. Returns True if handled.

        For list views (rules, scenarios, map), switches to detail view mode.
        For single items (R001, S3), uses modal popup.

        IMPORTANT: Always updates state with passed data first to ensure
        detail views show the CURRENT state with proper diff highlighting.
        """
        parts = command.lower().split()
        if not parts:
            return False

        if parts[0] == "show":
            if len(parts) == 1:
                return False

            target = parts[1]

            # Full list views - UPDATE STATE FIRST, then switch to detail view mode
            if target == "rules":
                # Update state with current data before showing detail view
                self._update_current_data(logic_map, scenarios, coverage)
                page = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
                self.enter_detail_view("rules_detail", page)
                return True

            elif target == "scenarios":
                # Update state with current data before showing detail view
                self._update_current_data(logic_map, scenarios, coverage)
                page = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
                self.enter_detail_view("scenarios_detail", page)
                return True

            elif target in ("map", "logicmap", "logic_map", "logic-map"):
                # Update state with current data before showing detail view
                self._update_current_data(logic_map, scenarios, coverage)
                self.enter_detail_view("logic_map_detail")
                return True

            elif target == "gaps" and coverage:
                items = [(f"G{i + 1}", gap) for i, gap in enumerate(coverage.gaps)]
                if not items:
                    self._hitl_print(Text("[green]No coverage gaps![/green]"))
                else:
                    self._hitl_print_list("Coverage Gaps", items)
                return True

            elif target == "coverage" and coverage:
                # Update state with current data before showing detail view
                self._update_current_data(logic_map, scenarios, coverage)
                self.enter_detail_view("coverage_detail")
                return True

            elif target in ("categories", "cats", "taxonomy"):
                # Update state with current data before showing detail view
                self._update_current_data(logic_map, scenarios, coverage)
                page = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
                self.enter_detail_view("categories_detail", page)
                return True

            # Single item views - use modal popup (Enter to close)
            elif target.upper().startswith("R"):
                self._hitl_print_rule(target.upper(), logic_map)
                return True

            elif target.upper().startswith("S") and scenarios:
                self._hitl_print_scenario(target.upper(), scenarios)
                return True

        elif parts[0] == "find":
            if len(parts) < 2:
                return False

            query = " ".join(parts[1:]).strip("\"'")
            results: list = []

            matching_rules = [
                (r.rule_id, r.text)
                for r in logic_map.rules
                if query.lower() in r.text.lower() or query.lower() in r.rule_id.lower()
            ]

            if matching_rules:
                results.append((f"Rules matching '{query}'", matching_rules))

            if scenarios:
                matching_scenarios = [
                    (f"S{i + 1}", s.description)
                    for i, s in enumerate(scenarios)
                    if query.lower() in s.description.lower()
                ]
                if matching_scenarios:
                    results.append((f"Scenarios matching '{query}'", matching_scenarios))

            if results:
                self._hitl_print_search_results(results)
            else:
                self._hitl_print(Text(f"[dim]No matches for '{query}'[/dim]"))

            return True

        return False

    # =========================================================================
    # HITL Print Helpers - Always clear screen first to prevent stacking
    # =========================================================================

    def _hitl_print(self, content) -> None:
        """Print content in HITL mode - clears screen first to prevent stacking."""
        # Use ANSI escape codes for reliable clearing across terminals
        print("\033[2J\033[H", end="", flush=True)
        self.console.print(content)
        self.console.print("\n[dim]Press Enter to continue...[/dim]")
        try:
            self.console.input()
        except (KeyboardInterrupt, EOFError):
            pass

    def _hitl_print_list(self, title: str, items: list[tuple[str, str]], page: int = 1) -> None:
        """Print paginated list in HITL mode."""
        per_page = 15
        total_pages = max(1, (len(items) + per_page - 1) // per_page)
        page = max(1, min(page, total_pages))

        start = (page - 1) * per_page
        end = start + per_page
        page_items = items[start:end]

        content_lines: list = [Text("")]
        for item_id, description in page_items:
            line = Text("  ")
            line.append(item_id, style="cyan")
            line.append(f"  {description[:60]}", style="white")
            if len(description) > 60:
                line.append("...", style="dim")
            content_lines.append(line)

        content_lines.append(Text(""))
        content_lines.append(Text(f"  Page {page}/{total_pages} ({len(items)} total)", style="dim"))

        panel = Panel(
            Group(*content_lines),
            title=f"[bold]{title}[/bold]",
            border_style="cyan",
        )
        self._hitl_print(panel)

    def _hitl_print_coverage(self, coverage: "CoverageReport") -> None:
        """Print coverage table in HITL mode."""
        table = Table(show_header=True, header_style="bold cyan", title="Coverage")
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
            "[bold]Total[/bold]",
            f"[bold]{coverage.overall_coverage_percent:.0f}%[/bold]",
            f"({coverage.covered_count}✓ {coverage.partial_count}~ {coverage.uncovered_count}✗)",
        )
        self._hitl_print(table)

    def _hitl_print_rule(self, rule_id: str, logic_map: "LogicMap") -> None:
        """Print rule detail in HITL mode."""
        rule = logic_map.get_rule(rule_id)
        if not rule:
            self._hitl_print(Text(f"[red]Rule {rule_id} not found[/red]"))
            return

        content_lines: list = [Text("")]
        content_lines.append(Text(f"  ID:       {rule.rule_id}", style="cyan"))
        cat = rule.category.value if hasattr(rule.category, "value") else str(rule.category)
        content_lines.append(Text(f"  Category: {cat}", style="white"))
        content_lines.append(Text(f"  Text:     {rule.text}", style="white"))
        if rule.condition:
            content_lines.append(Text(f"  Condition: {rule.condition}", style="dim"))
        if rule.action:
            content_lines.append(Text(f"  Action:   {rule.action}", style="dim"))
        content_lines.append(Text(""))

        panel = Panel(
            Group(*content_lines),
            title=f"[bold]Rule {rule_id}[/bold]",
            border_style="cyan",
        )
        self._hitl_print(panel)

    def _hitl_print_scenario(self, scenario_id: str, scenarios: list["GoldenScenario"]) -> None:
        """Print scenario detail in HITL mode."""
        try:
            idx = int(scenario_id.upper().replace("S", "")) - 1
            if idx < 0 or idx >= len(scenarios):
                self._hitl_print(Text(f"[red]Scenario {scenario_id} not found[/red]"))
                return
        except ValueError:
            self._hitl_print(Text(f"[red]Invalid scenario ID: {scenario_id}[/red]"))
            return

        scenario = scenarios[idx]
        content_lines: list = [Text("")]
        content_lines.append(Text(f"  ID:          S{idx + 1}", style="cyan"))
        stype = (
            scenario.scenario_type.value
            if hasattr(scenario.scenario_type, "value")
            else str(scenario.scenario_type)
        )
        content_lines.append(Text(f"  Type:        {stype}", style="white"))
        content_lines.append(Text(f"  Description: {scenario.description}", style="white"))
        if scenario.context:
            content_lines.append(Text(f"  Context:     {scenario.context}", style="dim"))
        if scenario.target_rule_ids:
            content_lines.append(
                Text(f"  Rules:       {', '.join(scenario.target_rule_ids)}", style="dim")
            )
        content_lines.append(Text(""))

        panel = Panel(
            Group(*content_lines),
            title=f"[bold]Scenario S{idx + 1}[/bold]",
            border_style="green",
        )
        self._hitl_print(panel)

    def _hitl_print_search_results(self, results: list[tuple[str, list]]) -> None:
        """Print search results in HITL mode."""
        content_lines: list = [Text("")]
        for title, items in results:
            content_lines.append(Text(f"  {title}:", style="bold cyan"))
            for item_id, desc in items[:5]:
                line = Text("    ")
                line.append(item_id, style="cyan")
                line.append(f"  {desc[:50]}", style="white")
                if len(desc) > 50:
                    line.append("...", style="dim")
                content_lines.append(line)
            if len(items) > 5:
                content_lines.append(Text(f"    ... +{len(items) - 5} more", style="dim"))
            content_lines.append(Text(""))

        panel = Panel(
            Group(*content_lines),
            title="[bold]Search Results[/bold]",
            border_style="cyan",
        )
        self._hitl_print(panel)


__all__ = ["LiveProgressDisplay", "DisplayState"]
