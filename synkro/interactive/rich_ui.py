"""Rich UI components for Human-in-the-Loop interaction."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from synkro.types.logic_map import LogicMap
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
        """Display instructions for the unified HITL session (turns + rules)."""
        from rich.panel import Panel

        instructions = """[bold]Commands:[/bold]
  • [cyan]done[/cyan] - Continue with current settings
  • [cyan]undo[/cyan] - Revert last change
  • [cyan]reset[/cyan] - Restore original state
  • [cyan]show R001[/cyan] - Show details of a specific rule
  • [cyan]help[/cyan] - Show this message

[bold]Feedback examples:[/bold]
  • [yellow]"shorter conversations"[/yellow] - Adjust conversation turns
  • [yellow]"I want 5 turns"[/yellow] - Set specific turn count
  • [yellow]"remove R005"[/yellow] - Delete a rule
  • [yellow]"add rule for late submissions"[/yellow] - Add a new rule"""

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
