"""Synkro CLI - Generate training data from the command line."""

import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

# Load .env file from current directory or parent directories
load_dotenv()

app = typer.Typer(
    name="synkro",
    help="Generate training datasets from documents.",
    no_args_is_help=True,
)


def _get_default_model() -> str:
    """Auto-detect available API key and return appropriate default model."""
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return "gemini/gemini-2.5-flash"  # gemini/ prefix for Google AI API
    elif os.getenv("ANTHROPIC_API_KEY"):
        return "claude-3-5-sonnet-20241022"
    elif os.getenv("OPENAI_API_KEY"):
        return "gpt-4o-mini"
    elif os.getenv("CEREBRAS_API_KEY"):
        return "cerebras/gpt-oss-120b"
    else:
        # Default to OpenAI, will fail with clear error if no key
        return "gpt-4o-mini"


def _get_ingestion_model() -> str:
    """Auto-detect available API key and return fast model for rule extraction.

    Rule extraction is a simple task (read text, identify rules, categorize).
    Small/fast models are more than capable and much faster.
    """
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        return "gemini/gemini-2.0-flash"  # Fast and capable for extraction
    elif os.getenv("ANTHROPIC_API_KEY"):
        return "claude-3-5-haiku-20241022"  # Fast Anthropic model
    elif os.getenv("OPENAI_API_KEY"):
        return "gpt-4o-mini"  # Fast OpenAI model
    elif os.getenv("CEREBRAS_API_KEY"):
        return "cerebras/gpt-oss-120b"  # Fast Cerebras model
    else:
        return "gpt-4o-mini"


@app.command()
def generate(
    source: Optional[str] = typer.Argument(
        None,
        help="Policy text, file path (.pdf, .docx, .txt, .md), folder path, or URL. Optional if --config is provided.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Pre-ingested config file (from 'synkro ingest'). Skips extraction for faster generation.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (auto-generated if not specified)",
    ),
    traces: int = typer.Option(
        20,
        "--traces",
        "-n",
        help="Number of traces to generate",
    ),
    format: str = typer.Option(
        "messages",
        "--format",
        "-f",
        help="Output format: messages, qa, langsmith, langfuse, tool_call, chatml",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model for generation (auto-detects from API key if not specified)",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        "-p",
        help="LLM provider for local models (ollama, vllm)",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "--endpoint",
        "-e",
        help="API endpoint URL (e.g., http://localhost:11434)",
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        "-i/-I",
        help="Enable interactive Logic Map editing before generation (enabled by default)",
    ),
    fast: bool = typer.Option(
        False,
        "--fast",
        help="Skip coverage tracking for faster generation",
    ),
    pretty: bool = typer.Option(
        False,
        "--pretty",
        help="Pretty-print JSON output (multi-line). Easier to read, but larger files.",
    ),
):
    """
    Generate training data from a policy document.

    Examples:

        synkro generate policy.pdf

        synkro generate policy.pdf --config policy_config.json  # Fast mode with pre-ingested rules

        synkro generate --config policy_config.json -n 50  # Source optional with config

        synkro generate "All expenses over $50 need approval" --traces 50

        synkro generate handbook.docx -o training.jsonl -n 100

        synkro generate policy.pdf --interactive  # Review and edit Logic Map

        synkro generate policy.pdf --fast  # Skip coverage tracking
    """
    from rich.console import Console

    import synkro
    from synkro import Policy, create_pipeline

    console = Console()

    # Validate inputs
    if source is None and config is None:
        console.print("[red]Error: Either source or --config must be provided[/red]")
        raise typer.Exit(1)

    # Handle local LLM provider configuration
    base_url = endpoint
    effective_model = model or _get_default_model()

    if provider:
        # Format model string for LiteLLM if provider specified
        if "/" not in effective_model:
            effective_model = f"{provider}/{effective_model}"

        # Use default endpoint if not specified
        if not endpoint:
            defaults = {
                "ollama": "http://localhost:11434",
                "vllm": "http://localhost:8000",
            }
            base_url = defaults.get(provider)

    # Use pipeline for more control when config is provided
    if config:
        if not config.exists():
            console.print(f"[red]Config file not found: {config}[/red]")
            raise typer.Exit(1)

        console.print(f"[cyan]Using pre-ingested config: {config}[/cyan]")

        pipeline = create_pipeline(
            model=effective_model,
            logic_map=config,
            skip_coverage=fast,
            enable_hitl=interactive,
            base_url=base_url,
        )

        # Policy is optional when using config
        if source:
            source_path = Path(source)
            if source_path.exists():
                policy = Policy.from_file(source_path)
            elif source.startswith(("http://", "https://")):
                policy = Policy.from_url(source)
            else:
                policy = Policy(text=source)
            dataset = pipeline.generate(policy, traces=traces)
        else:
            dataset = pipeline.generate(traces=traces)
    else:
        # Standard path without config
        source_path = Path(source)

        if source_path.exists():
            policy = Policy.from_file(source_path)
        elif source.startswith(("http://", "https://")):
            policy = Policy.from_url(source)
        else:
            policy = Policy(text=source)

        dataset = synkro.generate(
            policy,
            traces=traces,
            generation_model=effective_model,
            enable_hitl=interactive,
            base_url=base_url,
        )

    # Save
    if output:
        dataset.save(output, format=format, pretty_print=pretty)
    else:
        dataset.save(format=format, pretty_print=pretty)


@app.command()
def ingest(
    source: str = typer.Argument(
        ...,
        help="Policy file path (.pdf, .docx, .txt, .md) or raw text",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output config file path (default: <source_name>_config.json)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model for rule extraction (default: fast model like gemini-2.0-flash)",
    ),
    review: bool = typer.Option(
        False,
        "--review",
        "-r",
        help="Interactive review of extracted rules",
    ),
    endpoint: Optional[str] = typer.Option(
        None,
        "--endpoint",
        "-e",
        help="API endpoint URL for local models",
    ),
):
    """
    Ingest a policy and extract rules for fast generation.

    This extracts the Logic Map (rules as DAG) once and saves it.
    Use the saved config with 'generate --config' for faster runs.

    Uses a fast model (gemini-2.0-flash, gpt-4o-mini, or claude-3-5-haiku)
    by default since rule extraction is a simple task.

    Examples:

        synkro ingest policy.pdf

        synkro ingest policy.md -o my_config.json

        synkro ingest policy.txt --review  # Interactive review

        synkro ingest policy.pdf --model gpt-4o  # Override with stronger model
    """
    from rich.console import Console

    import synkro

    console = Console()

    # Determine output path
    if output is None:
        source_path = Path(source)
        if source_path.exists():
            output = Path(f"{source_path.stem}_config.json")
        else:
            output = Path("policy_config.json")

    # Always use fast model for ingestion (rule extraction is simple)
    # User can override with --model if needed
    effective_model = model or _get_ingestion_model()

    # Ingest
    synkro.ingest(
        source,
        output=output,
        model=effective_model,
        review=review,
        base_url=endpoint,
    )

    console.print(f"\n[dim]Use with: synkro generate --config {output}[/dim]\n")


@app.command()
def remove(
    config: Path = typer.Argument(
        ...,
        help="Config file to remove (.json)",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Remove without confirmation",
    ),
):
    """
    Remove an ingested policy config file.

    Examples:

        synkro remove policy_config.json

        synkro remove policy_config.json --force
    """
    from rich.console import Console
    from rich.prompt import Confirm

    console = Console()

    if not config.exists():
        console.print(f"[red]Config file not found: {config}[/red]")
        raise typer.Exit(1)

    if not config.suffix == ".json":
        console.print(f"[red]Expected .json file, got: {config}[/red]")
        raise typer.Exit(1)

    # Confirm unless --force
    if not force:
        if not Confirm.ask(f"Remove {config}?"):
            console.print("[dim]Cancelled[/dim]")
            raise typer.Exit(0)

    config.unlink()
    console.print(f"[green]✓ Removed {config}[/green]")


@app.command()
def configs():
    """
    List all ingested policy configs in the current directory.
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()

    # Find all *_config.json files
    config_files = list(Path(".").glob("*_config.json")) + list(Path(".").glob("*config*.json"))
    config_files = sorted(set(config_files))

    if not config_files:
        console.print("[dim]No config files found in current directory[/dim]")
        console.print("[dim]Run 'synkro ingest <policy>' to create one[/dim]")
        return

    table = Table(title="Ingested Policy Configs")
    table.add_column("File", style="cyan")
    table.add_column("Rules", justify="right")
    table.add_column("Complexity")
    table.add_column("Turns", justify="right")

    for path in config_files:
        try:
            from synkro.ingestion import PolicyConfig

            config = PolicyConfig.load(path)
            table.add_row(
                str(path),
                str(len(config.logic_map.rules)),
                config.complexity.level,
                str(config.complexity.recommended_turns),
            )
        except Exception:
            table.add_row(str(path), "?", "?", "?")

    console.print(table)


@app.command()
def demo():
    """
    Run a quick demo with a built-in example policy.
    """
    from rich.console import Console

    import synkro
    from synkro.examples import EXPENSE_POLICY

    console = Console()
    console.print("\n[cyan]Running demo with built-in expense policy...[/cyan]\n")

    dataset = synkro.generate(EXPENSE_POLICY, traces=5)
    dataset.save("demo_output.jsonl")

    console.print("\n[green]Demo complete![/green]")
    console.print("[dim]Check demo_output.jsonl for the generated training data.[/dim]\n")


@app.command()
def version():
    """Show version information."""
    from rich.console import Console

    import synkro

    console = Console()
    console.print(f"[cyan]synkro[/cyan] v{synkro.__version__}")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
