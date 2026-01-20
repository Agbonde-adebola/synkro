# CLAUDE.md - Synkro Development Guide

## Project Overview

Synkro is a Python framework for turning unstructured policies into training data for LLMs. It extracts rules from policy documents, generates scenarios, and synthesizes conversation traces.

## Tech Stack

- **Python**: 3.10+ required
- **Build**: Hatchling
- **Linting**: Ruff
- **Testing**: pytest, pytest-asyncio
- **UI**: Rich (console UI with Live displays)
- **LLM**: LiteLLM for multi-provider support

## Key Directories

```
synkro/
├── core/           # Dataset, checkpoint management
├── interactive/    # Live display, HITL session, rich UI
├── pipeline/       # Main runner, generation pipeline
├── types/          # Pydantic models (logic_map, coverage, etc.)
└── cli.py          # CLI entry point
```

## Before Publishing

**ALWAYS run pre-commit before publishing to PyPI or pushing to GitHub:**

```bash
# Run pre-commit checks (ruff lint + format)
pre-commit run --all-files

# If it fails, it will auto-fix issues. Run again to verify:
pre-commit run --all-files
```

## Publishing Workflow

1. **Bump version** in `pyproject.toml`
2. **Run pre-commit**: `pre-commit run --all-files`
3. **Build**: `python -m build`
4. **Upload to PyPI**: `python -m twine upload dist/synkro-X.Y.Z*`
5. **Commit and push**: `git add -A && git commit -m "..." && git push origin main`

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_imports.py -v

# Note: tests/test_streaming_api.py requires API keys (integration tests)
```

## Live Display Architecture

The `LiveProgressDisplay` class in `synkro/interactive/live_display.py` handles the terminal UI:

- Uses Rich's `Live` component with `transient=True` for in-place updates
- Pass **callable** to `Live()` (not the result) for auto-refresh animation
- Use `refresh()` not `update()` to trigger re-render without replacing the callable
- When Live is active, `spinner()` should return no-op to prevent stacking

### Key Methods

- `start()` - Start live display
- `stop()` - Stop and print final panel
- `_render()` - Callable that returns the Panel (called by Rich on each refresh)
- `_refresh()` - Trigger immediate refresh via `self._live.refresh()`
- `hitl_get_input()` - For HITL: clears screen, renders state, gets input

### Common Pitfalls

1. **Panels stacking**: Don't call `console.print()` while Live is running
2. **Spinner not animating**: Pass `self._render` (callable), not `self._render()` (result)
3. **State not updating**: Use `refresh()` not `update()` in `_refresh()`

## HITL (Human-in-the-Loop) Mode

The HITL session allows users to interactively edit rules/scenarios:

- `enter_hitl_mode()` - Pauses Live display
- `hitl_get_input()` - Unified render + input (clears screen first)
- `exit_hitl_mode()` - Resumes Live display

## Reporting

`RichReporter` in `synkro/reporting.py` connects the pipeline to the UI:

- `spinner()` returns no-op when Live is active (prevents stacking)
- Callbacks update the display state and add events
- Don't call `console.print()` during Live - update state instead

## Common Commands

```bash
# Local install for testing
pip install -e .

# Run example
python examples/quickstart.py

# Lint only
ruff check synkro/

# Format only
ruff format synkro/
```

## Version History Convention

Bump patch version (0.4.X) for bug fixes and minor features. The version is in `pyproject.toml`.
