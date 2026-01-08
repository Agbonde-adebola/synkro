"""Rich UI components for Human-in-the-Loop interaction."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synkro.types.logic_map import LogicMap, GoldenScenario
    from synkro.types.core import Plan


class LogicMapDisplay:
    """Rich-based display for Logic Maps."""

    def __init__(self) -> None:
        from rich.console import Console

        self.console = Console()

    def display_full(self, logic_map: "LogicMap") -> None:
        """Display the complete Logic Map with all details."""
        from rich.panel import Panel
        from rich.tree import Tree

        # Build tree view of rules by category
        tree = Tree("[bold cyan]Logic Map[/bold cyan]")

        # Group rules by category
        categories: dict[str, list] = {}
        for rule in logic_map.rules:
            cat = rule.category.value if hasattr(rule.category, "value") else str(rule.category)
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(rule)

        # Add each category as a branch
        for category, rules in sorted(categories.items()):
            branch = tree.add(f"[bold]{category}[/bold] ({len(rules)} rules)")
            for rule in rules:
                rule_text = f"[cyan]{rule.rule_id}[/cyan]: {rule.text[:60]}..."
                if rule.dependencies:
                    rule_text += f" [dim]→ {', '.join(rule.dependencies)}[/dim]"
                branch.add(rule_text)

        # Show root rules
        root_info = f"[dim]Root rules: {', '.join(logic_map.root_rules)}[/dim]"

        self.console.print()
        self.console.print(
            Panel(
                tree,
                title=f"[bold]📜 Logic Map ({len(logic_map.rules)} rules)[/bold]",
                subtitle=root_info,
                border_style="cyan",
            )
        )

    def display_diff(self, before: "LogicMap", after: "LogicMap") -> None:
        """Display all rules with changes highlighted in different colors."""
        from rich.panel import Panel
        from rich.tree import Tree

        before_ids = {r.rule_id for r in before.rules}
        after_ids = {r.rule_id for r in after.rules}

        added = after_ids - before_ids
        removed = before_ids - after_ids
        common = before_ids & after_ids

        # Check for modifications in common rules
        modified: set[str] = set()
        before_map = {r.rule_id: r for r in before.rules}
        after_map = {r.rule_id: r for r in after.rules}

        for rule_id in common:
            if before_map[rule_id].text != after_map[rule_id].text:
                modified.add(rule_id)
            elif before_map[rule_id].dependencies != after_map[rule_id].dependencies:
                modified.add(rule_id)

        if not added and not removed and not modified:
            self.console.print("[dim]No changes detected[/dim]")
            return

        # Build tree view of rules by category (like display_full but with colors)
        tree = Tree("[bold cyan]Logic Map[/bold cyan]")

        # Group rules by category
        categories: dict[str, list] = {}
        for rule in after.rules:
            cat = rule.category.value if hasattr(rule.category, "value") else str(rule.category)
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(rule)

        # Add each category as a branch
        for category, rules in sorted(categories.items()):
            branch = tree.add(f"[bold]{category}[/bold] ({len(rules)} rules)")
            for rule in rules:
                # Determine style based on change type
                if rule.rule_id in added:
                    prefix = "[green]+ "
                    style_close = "[/green]"
                    id_style = "[green]"
                elif rule.rule_id in modified:
                    prefix = "[yellow]~ "
                    style_close = "[/yellow]"
                    id_style = "[yellow]"
                else:
                    prefix = ""
                    style_close = ""
                    id_style = "[cyan]"

                rule_text = f"{prefix}{id_style}{rule.rule_id}[/]: {rule.text[:60]}...{style_close}"
                if rule.dependencies:
                    rule_text += f" [dim]→ {', '.join(rule.dependencies)}[/dim]"
                branch.add(rule_text)

        # Add removed rules section at the bottom
        if removed:
            removed_branch = tree.add("[red][bold]REMOVED[/bold][/red]")
            for rule_id in sorted(removed):
                rule = before_map[rule_id]
                removed_branch.add(f"[red][strike]- {rule_id}: {rule.text[:60]}...[/strike][/red]")

        # Build legend
        legend = "[dim]Legend: [green]+ Added[/green] | [yellow]~ Modified[/yellow] | [red][strike]- Removed[/strike][/red][/dim]"

        self.console.print()
        self.console.print(
            Panel(
                tree,
                title=f"[bold]📜 Logic Map ({len(after.rules)} rules)[/bold]",
                subtitle=legend,
                border_style="cyan",
            )
        )

    def display_rule(self, rule_id: str, logic_map: "LogicMap") -> None:
        """Display details of a specific rule."""
        from rich.panel import Panel

        rule = logic_map.get_rule(rule_id)
        if not rule:
            self.console.print(f"[red]Rule {rule_id} not found[/red]")
            return

        content = f"""[bold]ID:[/bold] {rule.rule_id}
[bold]Text:[/bold] {rule.text}
[bold]Category:[/bold] {rule.category}
[bold]Condition:[/bold] {rule.condition or 'N/A'}
[bold]Action:[/bold] {rule.action or 'N/A'}
[bold]Dependencies:[/bold] {', '.join(rule.dependencies) if rule.dependencies else 'None (root rule)'}"""

        self.console.print(Panel(content, title=f"Rule {rule_id}", border_style="cyan"))

    def show_error(self, message: str) -> None:
        """Display an error message."""
        self.console.print(f"[red]Error:[/red] {message}")

    def show_success(self, message: str) -> None:
        """Display a success message."""
        self.console.print(f"[green]✓[/green] {message}")

    def spinner(self, message: str = "Processing..."):
        """Context manager that shows a loading spinner.

        Usage:
            with display.spinner("Applying changes..."):
                await some_llm_call()
        """
        from rich.status import Status
        return Status(f"[cyan]{message}[/cyan]", spinner="dots", console=self.console)

    def display_session_state(
        self,
        plan: "Plan",
        logic_map: "LogicMap",
        current_turns: int,
    ) -> None:
        """Display both conversation settings and logic map together."""
        from rich.panel import Panel
        from rich.table import Table

        # Conversation settings panel
        turns_table = Table(show_header=False, box=None)
        turns_table.add_row(
            "[dim]Complexity:[/dim]",
            f"[cyan]{plan.complexity_level.title()}[/cyan]",
        )
        turns_table.add_row(
            "[dim]Turns:[/dim]",
            f"[green]{current_turns}[/green]",
        )

        self.console.print()
        self.console.print(
            Panel(
                turns_table,
                title="[cyan]Conversation Settings[/cyan]",
                border_style="cyan",
            )
        )

        # Then display logic map (existing method)
        self.display_full(logic_map)

    def display_scenarios(
        self,
        scenarios: list["GoldenScenario"],
        distribution: dict[str, int] | None = None,
    ) -> None:
        """Display all scenarios with S1, S2... IDs grouped by type."""
        from rich.panel import Panel
        from rich.tree import Tree

        # Scenario tree view grouped by type
        tree = Tree("[bold cyan]Scenarios[/bold cyan]")

        # Group by type
        by_type: dict[str, list[tuple[int, "GoldenScenario"]]] = {}
        for i, scenario in enumerate(scenarios, start=1):
            type_key = scenario.scenario_type.value
            if type_key not in by_type:
                by_type[type_key] = []
            by_type[type_key].append((i, scenario))

        # Add each type as a branch
        for type_name in ["positive", "negative", "edge_case", "irrelevant"]:
            if type_name in by_type:
                type_display = type_name.replace("_", " ").title()
                branch = tree.add(f"[bold]{type_display}[/bold] ({len(by_type[type_name])})")
                for idx, scenario in by_type[type_name]:
                    desc = scenario.description[:50] + "..." if len(scenario.description) > 50 else scenario.description
                    rules = ", ".join(scenario.target_rule_ids[:3]) if scenario.target_rule_ids else "None"
                    if len(scenario.target_rule_ids) > 3:
                        rules += "..."
                    branch.add(f"[cyan]S{idx}[/cyan]: {desc} [dim]→ {rules}[/dim]")

        self.console.print()
        self.console.print(
            Panel(
                tree,
                title=f"[bold]📋 Scenarios ({len(scenarios)} total)[/bold]",
                border_style="green",
            )
        )

    def display_scenario(
        self,
        scenario_id: str,
        scenarios: list["GoldenScenario"],
    ) -> None:
        """Display details of a specific scenario."""
        from rich.panel import Panel

        # Parse S1, S2, etc. to index
        try:
            idx = int(scenario_id.upper().replace("S", "")) - 1
            if idx < 0 or idx >= len(scenarios):
                self.console.print(f"[red]Scenario {scenario_id} not found (valid: S1-S{len(scenarios)})[/red]")
                return
        except ValueError:
            self.console.print(f"[red]Invalid scenario ID: {scenario_id}[/red]")
            return

        scenario = scenarios[idx]
        content = f"""[bold]ID:[/bold] S{idx + 1}
[bold]Type:[/bold] {scenario.scenario_type.value.replace('_', ' ').title()}
[bold]Description:[/bold] {scenario.description}
[bold]Context:[/bold] {scenario.context or 'N/A'}
[bold]Target Rules:[/bold] {', '.join(scenario.target_rule_ids) if scenario.target_rule_ids else 'None'}
[bold]Expected Outcome:[/bold] {scenario.expected_outcome}"""

        self.console.print(Panel(content, title=f"Scenario S{idx + 1}", border_style="green"))

    def display_scenario_diff(
        self,
        before: list["GoldenScenario"],
        after: list["GoldenScenario"],
    ) -> None:
        """Display all scenarios with changes highlighted in different colors."""
        from rich.panel import Panel
        from rich.tree import Tree

        # Create simple ID comparison
        before_descs = {s.description for s in before}
        after_descs = {s.description for s in after}

        added_descs = after_descs - before_descs
        removed_descs = before_descs - after_descs

        if not added_descs and not removed_descs:
            self.console.print("[dim]No changes detected[/dim]")
            return

        # Build tree view grouped by type
        tree = Tree("[bold cyan]Scenarios[/bold cyan]")

        # Group by type
        by_type: dict[str, list[tuple[int, "GoldenScenario", bool]]] = {}
        for i, scenario in enumerate(after, start=1):
            type_key = scenario.scenario_type.value
            if type_key not in by_type:
                by_type[type_key] = []
            is_added = scenario.description in added_descs
            by_type[type_key].append((i, scenario, is_added))

        # Add each type as a branch
        for type_name in ["positive", "negative", "edge_case", "irrelevant"]:
            if type_name in by_type:
                type_display = type_name.replace("_", " ").title()
                branch = tree.add(f"[bold]{type_display}[/bold] ({len(by_type[type_name])})")
                for idx, scenario, is_added in by_type[type_name]:
                    desc = scenario.description[:50] + "..." if len(scenario.description) > 50 else scenario.description

                    if is_added:
                        prefix = "[green]+ "
                        style_close = "[/green]"
                        id_style = "[green]"
                    else:
                        prefix = ""
                        style_close = ""
                        id_style = "[cyan]"

                    rules = ", ".join(scenario.target_rule_ids[:3]) if scenario.target_rule_ids else "None"
                    branch.add(f"{prefix}{id_style}S{idx}[/]: {desc}{style_close} [dim]→ {rules}[/dim]")

        # Add removed scenarios at bottom
        if removed_descs:
            removed_branch = tree.add("[red][bold]REMOVED[/bold][/red]")
            for scenario in before:
                if scenario.description in removed_descs:
                    desc = scenario.description[:50] + "..." if len(scenario.description) > 50 else scenario.description
                    removed_branch.add(f"[red][strike]- {desc}[/strike][/red]")

        # Build legend
        legend = "[dim]Legend: [green]+ Added[/green] | [red][strike]- Removed[/strike][/red][/dim]"

        self.console.print()
        self.console.print(
            Panel(
                tree,
                title=f"[bold]📋 Scenarios ({len(after)} total)[/bold]",
                subtitle=legend,
                border_style="green",
            )
        )

    def display_full_session_state(
        self,
        plan: "Plan",
        logic_map: "LogicMap",
        current_turns: int,
        scenarios: list["GoldenScenario"] | None,
        distribution: dict[str, int] | None,
    ) -> None:
        """Display logic map, scenarios, and session details (at bottom)."""
        from rich.panel import Panel

        # Display logic map first
        self.display_full(logic_map)

        # Display scenarios if available
        if scenarios:
            self.display_scenarios(scenarios)

        # Session Details panel at bottom (instructions + settings)
        session_details = f"""[bold]Commands:[/bold] [cyan]done[/cyan] | [cyan]undo[/cyan] | [cyan]reset[/cyan] | [cyan]show R001[/cyan] | [cyan]show S3[/cyan] | [cyan]help[/cyan]

[bold]Feedback:[/bold] [yellow]"shorter"[/yellow] [yellow]"5 turns"[/yellow] [yellow]"remove R005"[/yellow] [yellow]"add scenario for..."[/yellow] [yellow]"delete S3"[/yellow]

[dim]Complexity:[/dim] [cyan]{plan.complexity_level.title()}[/cyan]    [dim]Turns:[/dim] [green]{current_turns}[/green]"""

        self.console.print()
        self.console.print(
            Panel(
                session_details,
                title="[bold cyan]Session Details[/bold cyan]",
                border_style="cyan",
            )
        )


class InteractivePrompt:
    """Handles user input for HITL sessions."""

    def __init__(self) -> None:
        from rich.console import Console

        self.console = Console()

    def show_instructions(self) -> None:
        """Display instructions for the HITL session."""
        from rich.panel import Panel

        instructions = """[bold]Commands:[/bold]
  • Type feedback to modify the Logic Map (e.g., "add a rule for...", "remove R005")
  • [cyan]done[/cyan] - Continue with current Logic Map
  • [cyan]undo[/cyan] - Revert last change
  • [cyan]reset[/cyan] - Restore original Logic Map
  • [cyan]show R001[/cyan] - Show details of a specific rule
  • [cyan]help[/cyan] - Show this message"""

        self.console.print()
        self.console.print(
            Panel(
                instructions,
                title="[bold cyan]Interactive Logic Map Editor[/bold cyan]",
                border_style="cyan",
            )
        )

    def show_unified_instructions(self) -> None:
        """Display instructions for the unified HITL session (turns + rules + scenarios)."""
        from rich.panel import Panel

        instructions = """[bold]Commands:[/bold]
  • [cyan]done[/cyan] - Continue with current settings
  • [cyan]undo[/cyan] - Revert last change
  • [cyan]reset[/cyan] - Restore original state
  • [cyan]show R001[/cyan] - Show details of a specific rule
  • [cyan]show S3[/cyan] - Show details of a specific scenario
  • [cyan]help[/cyan] - Show this message

[bold]Feedback examples:[/bold]
  [dim]Turns:[/dim]
  • [yellow]"shorter conversations"[/yellow] - Adjust conversation turns
  • [yellow]"I want 5 turns"[/yellow] - Set specific turn count
  [dim]Rules:[/dim]
  • [yellow]"remove R005"[/yellow] - Delete a rule
  • [yellow]"add rule for late submissions"[/yellow] - Add a new rule
  [dim]Scenarios:[/dim]
  • [yellow]"add scenario for expenses at $50 limit"[/yellow] - Add edge case
  • [yellow]"delete S3"[/yellow] - Remove a scenario
  • [yellow]"more negative scenarios"[/yellow] - Adjust distribution"""

        self.console.print()
        self.console.print(
            Panel(
                instructions,
                title="[bold cyan]Interactive Session[/bold cyan]",
                border_style="cyan",
            )
        )

    def get_feedback(self) -> str:
        """Prompt user for feedback on the Logic Map."""
        from rich.prompt import Prompt

        self.console.print()
        return Prompt.ask("[cyan]Enter feedback[/cyan]")

    def confirm_continue(self) -> bool:
        """Ask user if they want to continue with current Logic Map."""
        from rich.prompt import Confirm

        return Confirm.ask("Continue with this Logic Map?", default=True)
