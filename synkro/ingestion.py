"""Policy ingestion for pre-computing Logic Maps.

This module provides functions to ingest policies once and reuse
the extracted Logic Map for faster generation.

Usage:
    # Ingest a policy (one-time, with optional review)
    config = synkro.ingest(policy_text, output="./policy_config.json", review=True)

    # Use the pre-computed config for fast generation
    pipeline = create_pipeline(logic_map="./policy_config.json")
    dataset = pipeline.generate(traces=20)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from synkro.core.policy import Policy
from synkro.generation.logic_extractor import LogicExtractor
from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.types.core import Category
from synkro.types.logic_map import LogicMap, RuleCategory


class ComplexityInfo(BaseModel):
    """Derived complexity information from Logic Map."""

    level: Literal["simple", "conditional", "complex"] = Field(
        description="Complexity level derived from rule count and dependencies"
    )
    recommended_turns: int = Field(ge=1, le=6, description="Recommended conversation turns")
    rule_count: int = Field(description="Total number of rules")
    max_depth: int = Field(description="Maximum dependency chain depth")


class PolicyConfig(BaseModel):
    """
    Complete policy configuration for generation.

    Contains the Logic Map plus derived complexity and categories,
    enabling fast generation without re-analyzing the policy.
    """

    policy_hash: str = Field(description="SHA256 hash of original policy text")
    policy_text: str = Field(description="Original policy text for reference")
    complexity: ComplexityInfo = Field(description="Derived complexity info")
    logic_map: LogicMap = Field(description="Extracted Logic Map (rules as DAG)")
    categories: list[Category] = Field(description="Derived categories from rule categories")

    def save(self, path: str | Path) -> "PolicyConfig":
        """
        Save the policy configuration to a JSON file.

        Args:
            path: File path to save to

        Returns:
            Self for method chaining
        """
        path = Path(path)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)
        return self

    @classmethod
    def load(cls, path: str | Path) -> "PolicyConfig":
        """
        Load a policy configuration from a JSON file.

        Args:
            path: File path to load from

        Returns:
            PolicyConfig instance
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)

    def matches_policy(self, policy_text: str) -> bool:
        """Check if this config matches the given policy text."""
        return self.policy_hash == _hash_policy(policy_text)


def _hash_policy(policy_text: str) -> str:
    """Create a hash of the policy text."""
    return hashlib.sha256(policy_text.encode()).hexdigest()[:16]


def _derive_complexity(logic_map: LogicMap) -> ComplexityInfo:
    """
    Derive complexity information from a Logic Map.

    Uses rule count and dependency depth to determine complexity,
    eliminating the need for an LLM call.
    """
    rule_count = len(logic_map.rules)

    # Calculate max dependency depth
    def get_depth(rule_id: str, visited: set) -> int:
        if rule_id in visited:
            return 0
        visited.add(rule_id)
        rule = logic_map.get_rule(rule_id)
        if not rule or not rule.dependencies:
            return 1
        return 1 + max(get_depth(dep, visited.copy()) for dep in rule.dependencies)

    max_depth = 0
    for rule in logic_map.rules:
        depth = get_depth(rule.rule_id, set())
        max_depth = max(max_depth, depth)

    # Determine complexity level based on rules and depth
    if rule_count <= 3 and max_depth <= 1:
        level = "simple"
        recommended_turns = 1
    elif rule_count <= 8 and max_depth <= 2:
        level = "conditional"
        recommended_turns = 3
    else:
        level = "complex"
        recommended_turns = min(5, max_depth + 2)

    return ComplexityInfo(
        level=level,
        recommended_turns=recommended_turns,
        rule_count=rule_count,
        max_depth=max_depth,
    )


def _derive_categories(logic_map: LogicMap, target_traces: int = 20) -> list[Category]:
    """
    Derive scenario categories from Logic Map rule categories.

    Groups rules by their category and creates generation categories
    with appropriate trace distribution.
    """
    # Group rules by category
    category_rules: dict[str, list] = {}
    for rule in logic_map.rules:
        cat = rule.category.value if isinstance(rule.category, RuleCategory) else str(rule.category)
        if cat not in category_rules:
            category_rules[cat] = []
        category_rules[cat].append(rule)

    # Create categories with trace counts proportional to rule counts
    total_rules = len(logic_map.rules)
    categories = []

    # Map rule categories to descriptive names
    category_names = {
        "constraint": "Constraints & Requirements",
        "permission": "Permissions & Allowances",
        "procedure": "Procedures & Processes",
        "exception": "Exceptions & Special Cases",
    }

    category_descriptions = {
        "constraint": "Scenarios testing mandatory requirements and restrictions",
        "permission": "Scenarios testing what is allowed and permitted",
        "procedure": "Scenarios testing step-by-step processes",
        "exception": "Scenarios testing special cases and overrides",
    }

    remaining_traces = target_traces
    cat_items = list(category_rules.items())

    for i, (cat, rules) in enumerate(cat_items):
        # Calculate proportional trace count
        if i == len(cat_items) - 1:
            # Last category gets remaining to ensure exact total
            count = remaining_traces
        else:
            proportion = len(rules) / total_rules
            count = max(1, round(target_traces * proportion))
            remaining_traces -= count

        categories.append(
            Category(
                name=category_names.get(cat, cat.title()),
                description=category_descriptions.get(cat, f"Scenarios testing {cat} rules"),
                count=count,
            )
        )

    return categories


async def _extract_logic_map_with_progress(
    policy_text: str,
    model: Model = OpenAI.GPT_4O_MINI,
    base_url: str | None = None,
    on_progress: callable | None = None,
) -> tuple[LogicMap, int]:
    """Extract Logic Map from policy text with progress tracking.

    Strategy:
    - Small docs (< 8K chars): Single LLM call is fastest
    - Medium/large docs (>= 8K chars): Parallel chunked extraction
      (avoids output token limits when many rules are extracted)

    Uses fast models by default since rule extraction is a simple task.

    Returns:
        Tuple of (LogicMap, chunk_count)
    """
    llm = LLM(model=model, base_url=base_url, temperature=0.3)

    # Use chunked extraction for documents that could produce many rules
    # ~8K chars typically produces 30-40 rules, which fits in output limits
    if len(policy_text) >= 8000:
        from synkro.generation.logic_extractor_fast import FastLogicExtractor

        extractor = FastLogicExtractor(llm=llm, on_progress=on_progress)
        chunk_count = extractor.get_chunk_count(policy_text)
        logic_map = await extractor.extract(policy_text)
        return logic_map, chunk_count
    else:
        # Single call - faster for small docs
        extractor = LogicExtractor(llm=llm)
        logic_map = await extractor.extract(policy_text)
        return logic_map, 1


# File extensions that indicate a file path
_FILE_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".docx", ".doc", ".rst", ".html"}


def _looks_like_file_path(s: str) -> bool:
    """Check if a string looks like a file path."""
    # Check for path separators or file extensions
    if "/" in s or "\\" in s:
        # Has path separators - check if it could be a file
        path = Path(s)
        if path.suffix.lower() in _FILE_EXTENSIONS:
            return True
        # Check if it exists
        if path.exists() and path.is_file():
            return True
    # Check for file extension at the end
    for ext in _FILE_EXTENSIONS:
        if s.lower().endswith(ext):
            return True
    return False


def _read_policy_file(path: Path) -> str:
    """Read policy content from a file using Policy.from_file()."""
    if not path.exists():
        raise FileNotFoundError(f"Policy file not found: {path}")

    # Use Policy.from_file() which already handles PDF, DOCX, TXT, MD
    policy = Policy.from_file(path)
    return policy.text


def _display_config(config: PolicyConfig) -> None:
    """Display the policy configuration for review."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree

    console = Console()

    # Complexity info
    complexity_text = (
        f"[bold]Level:[/bold] {config.complexity.level}\n"
        f"[bold]Recommended Turns:[/bold] {config.complexity.recommended_turns}\n"
        f"[bold]Rule Count:[/bold] {config.complexity.rule_count}\n"
        f"[bold]Max Dependency Depth:[/bold] {config.complexity.max_depth}"
    )
    console.print(
        Panel(
            complexity_text,
            title="[bold cyan]📊 Complexity Analysis[/bold cyan]",
            border_style="cyan",
        )
    )

    # Logic Map
    tree = Tree("[bold cyan]Logic Map[/bold cyan]")

    # Group rules by category
    categories: dict[str, list] = {}
    for rule in config.logic_map.rules:
        cat = rule.category.value if hasattr(rule.category, "value") else str(rule.category)
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(rule)

    for category, rules in sorted(categories.items()):
        branch = tree.add(f"[bold]{category}[/bold] ({len(rules)} rules)")
        for rule in rules:
            rule_text = f"[cyan]{rule.rule_id}[/cyan]: {rule.text[:60]}..."
            if rule.dependencies:
                rule_text += f" [dim]→ {', '.join(rule.dependencies)}[/dim]"
            branch.add(rule_text)

    console.print(
        Panel(
            tree,
            title=f"[bold]📜 Extracted Rules ({len(config.logic_map.rules)} total)[/bold]",
            subtitle=f"[dim]Root rules: {', '.join(config.logic_map.root_rules)}[/dim]",
            border_style="cyan",
        )
    )

    # Categories table
    table = Table(title="📁 Derived Categories")
    table.add_column("Category", style="cyan")
    table.add_column("Description")
    table.add_column("Traces", justify="right")

    for cat in config.categories:
        table.add_row(cat.name, cat.description, str(cat.count))

    console.print(table)


def _interactive_review(config: PolicyConfig, policy_text: str) -> PolicyConfig:
    """Run interactive review session for the policy configuration."""
    from rich.console import Console
    from rich.prompt import Prompt

    console = Console()

    # Display current config
    _display_config(config)

    console.print("\n[bold]Review Options:[/bold]")
    console.print("  [cyan]done[/cyan]     - Accept and save")
    console.print("  [cyan]turns N[/cyan]  - Set recommended turns to N")
    console.print("  [cyan]show RXX[/cyan] - Show details of rule RXX")
    console.print("  [cyan]refresh[/cyan] - Re-display the configuration")

    while True:
        feedback = Prompt.ask("\n[bold]Your input[/bold]").strip().lower()

        if feedback == "done" or feedback == "":
            break

        if feedback == "refresh":
            _display_config(config)
            continue

        if feedback.startswith("turns "):
            try:
                turns = int(feedback.split()[1])
                if 1 <= turns <= 6:
                    config.complexity.recommended_turns = turns
                    console.print(f"[green]✓ Set recommended turns to {turns}[/green]")
                else:
                    console.print("[red]Turns must be between 1 and 6[/red]")
            except (ValueError, IndexError):
                console.print("[red]Usage: turns N (where N is 1-6)[/red]")
            continue

        if feedback.startswith("show "):
            rule_id = feedback.split()[1].upper()
            rule = config.logic_map.get_rule(rule_id)
            if rule:
                console.print(f"\n[bold]{rule.rule_id}[/bold]")
                console.print(f"  [bold]Text:[/bold] {rule.text}")
                console.print(f"  [bold]Category:[/bold] {rule.category}")
                console.print(f"  [bold]Condition:[/bold] {rule.condition}")
                console.print(f"  [bold]Action:[/bold] {rule.action}")
                console.print(
                    f"  [bold]Dependencies:[/bold] {', '.join(rule.dependencies) or 'None'}"
                )
            else:
                console.print(f"[red]Rule {rule_id} not found[/red]")
            continue

        console.print(
            "[dim]Unknown command. Type 'done' to finish or 'refresh' to see options.[/dim]"
        )

    return config


def ingest(
    policy: str | Path | Policy,
    output: str | Path | None = None,
    model: Model = OpenAI.GPT_4O_MINI,
    review: bool = False,
    base_url: str | None = None,
    target_traces: int = 20,
) -> PolicyConfig:
    """
    Ingest a policy and extract its Logic Map for reuse.

    This is the recommended first step for any policy. It:
    1. Extracts the Logic Map (rules as DAG) using parallel chunked extraction
    2. Derives complexity from rule structure (no LLM call)
    3. Derives categories from rule categories (no LLM call)
    4. Optionally allows interactive review
    5. Saves to a JSON file for reuse

    Uses a fast model by default (gpt-4o-mini, gemini-2.0-flash, etc.)
    since rule extraction is a simple task that doesn't need large models.

    Args:
        policy: Policy text, file path, or Policy object.
            If a Path or string ending in common file extensions
            (.txt, .md, .pdf, .docx), reads content from file.
        output: File path to save config (optional)
        model: Model to use for extraction (default: fast model)
        review: If True, show interactive review session
        base_url: Optional API base URL for local LLM providers
        target_traces: Target traces for category distribution (default: 20)

    Returns:
        PolicyConfig with Logic Map, complexity, and categories

    Examples:
        >>> # Ingest from file
        >>> config = synkro.ingest("./policy.md", output="./policy_config.json")

        >>> # Ingest from string
        >>> config = synkro.ingest(policy_text, output="./policy.json")

        >>> # With interactive review
        >>> config = synkro.ingest("./policy.txt", output="./policy.json", review=True)

        >>> # Just extract without saving
        >>> config = synkro.ingest(policy_text)
        >>> print(config.logic_map.rules)
    """
    import time

    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    console = Console()
    timings: dict[str, float] = {}
    source_name = None

    # Normalize policy - handle file paths, Policy objects, and strings
    t0 = time.time()
    if isinstance(policy, Policy):
        policy_text = policy.text
        source_name = "policy"
    elif isinstance(policy, Path):
        # Path object - read file
        policy_text = _read_policy_file(policy)
        source_name = policy.name
    elif isinstance(policy, str) and _looks_like_file_path(policy):
        # String that looks like a file path - read file
        policy_text = _read_policy_file(Path(policy))
        source_name = Path(policy).name
    else:
        # Assume it's policy text
        policy_text = policy
        source_name = "text input"
    timings["read_file"] = time.time() - t0

    # Calculate document stats
    word_count = len(policy_text.split())
    char_count = len(policy_text)

    # Display document info
    console.print(f"\n[bold cyan]📄 Document:[/bold cyan] {source_name}")
    console.print(f"[dim]   {char_count:,} characters • {word_count:,} words[/dim]")

    # Extract Logic Map with progress tracking
    t0 = time.time()
    use_parallel = char_count >= 8000

    if use_parallel:
        # Use progress bar for parallel extraction
        rules_found = [0]  # Use list to allow mutation in callback

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Extracting rules...[/bold cyan]"),
            BarColumn(bar_width=30),
            TaskProgressColumn(),
            TextColumn("•"),
            TextColumn("[green]{task.fields[rules]} rules[/green]"),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("extract", total=100, rules=0)

            def on_progress(completed: int, total: int, rules: int):
                rules_found[0] = rules
                pct = int((completed / total) * 100) if total > 0 else 0
                progress.update(task, completed=pct, rules=rules)

            logic_map, chunk_count = asyncio.run(
                _extract_logic_map_with_progress(policy_text, model, base_url, on_progress)
            )
    else:
        # Single call - just show spinner
        from rich.status import Status

        with Status("[bold cyan]Extracting rules...[/bold cyan]", console=console):
            logic_map, chunk_count = asyncio.run(
                _extract_logic_map_with_progress(policy_text, model, base_url)
            )

    timings["extract_rules"] = time.time() - t0

    # Show extraction results
    console.print(
        f"[green]✓[/green] Extracted [bold]{len(logic_map.rules)}[/bold] rules in [bold]{timings['extract_rules']:.1f}s[/bold]"
    )
    if use_parallel:
        console.print(f"[dim]   {chunk_count} chunks processed in parallel[/dim]")

    # Derive complexity and categories
    complexity = _derive_complexity(logic_map)
    categories = _derive_categories(logic_map, target_traces)

    # Create config
    config = PolicyConfig(
        policy_hash=_hash_policy(policy_text),
        policy_text=policy_text,
        complexity=complexity,
        logic_map=logic_map,
        categories=categories,
    )

    # Interactive review if requested
    if review:
        config = _interactive_review(config, policy_text)
    else:
        # Just display summary
        _display_config(config)

    # Save if output path provided
    if output:
        config.save(output)
        console.print(f"\n[green]✓ Saved to {output}[/green]")

    return config


def load_config(path: str | Path) -> PolicyConfig:
    """
    Load a previously saved policy configuration.

    Args:
        path: File path to load from

    Returns:
        PolicyConfig instance
    """
    return PolicyConfig.load(path)


__all__ = [
    "ingest",
    "load_config",
    "PolicyConfig",
    "ComplexityInfo",
]
