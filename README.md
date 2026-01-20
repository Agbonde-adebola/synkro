# Synkro

[![PyPI version](https://img.shields.io/pypi/v/synkro.svg)](https://pypi.org/project/synkro/)
[![Downloads](https://static.pepy.tech/badge/synkro)](https://pepy.tech/project/synkro)
[![Downloads/Week](https://static.pepy.tech/badge/synkro/week)](https://pepy.tech/project/synkro)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-synkro.sh-purple.svg)](https://synkro.sh/docs)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Framework for turning unstructured policies, handbooks, and documentation into high-quality conversation, tool calling or evaluation data for LLMs.

## Features

- **Quality Evaluation** - Each response is graded and automatically refined if it fails
- **Multiple Formats** - Conversation (multi-turn), Instruction (single-turn), Evaluation (Q&A), and Tool Calling
- **Eval Platform Support** - Export to LangSmith, Langfuse, or generic Q&A format
- **Tool Call Training** - Generate OpenAI function calling format for teaching models to use custom tools
- **Coverage Tracking** - Track scenario diversity like code coverage, identify gaps, and improve coverage with natural language commands
- **Top LLM Providers** - OpenAI, Anthropic, Google, and local models (Ollama, vLLM)
- **File Support** - PDF, DOCX, TXT, Markdown, URLs
- **CLI Included** - Generate datasets from the command line
- **Cost Tracking** - See total cost and LLM call breakdown after each generation

## Installation

```bash
pip install synkro
```

## Quick Start

```python
from synkro import create_pipeline, DatasetType
from synkro.models import Google
from synkro.examples import EXPENSE_POLICY

pipeline = create_pipeline(
    model=Google.GEMINI_25_FLASH,          # Fast generation
    grading_model=Google.GEMINI_25_PRO,    # Quality grading
    dataset_type=DatasetType.CONVERSATION,
)

dataset = pipeline.generate(EXPENSE_POLICY, traces=50)
dataset.save("training.jsonl")
```

### From Files

```python
from synkro.pipelines import create_pipeline
from synkro.core.policy import Policy

policy = Policy.from_file("handbook.pdf")  # PDF, DOCX, TXT, MD
pipeline = create_pipeline()
dataset = pipeline.generate(policy, traces=100)
dataset.save()
```

### From URLs

```python
from synkro.core.policy import Policy

policy = Policy.from_url("https://example.com/terms")
dataset = pipeline.generate(policy)
```

## Dataset Types

| Type | Turns | Output Formats | Best For |
|------|-------|----------------|----------|
| **CONVERSATION** | Multi | messages, chatml | Fine-tuning chat models |
| **INSTRUCTION** | 1 | messages, chatml | Instruction-following models |
| **EVALUATION** | 1 | qa, langsmith, langfuse | LLM evaluation & benchmarks |
| **TOOL_CALL** | Multi | tool_call, chatml | Teaching tool use |

### Conversation (Default)

```python
from synkro.types import DatasetType

pipeline = create_pipeline(dataset_type=DatasetType.CONVERSATION)
dataset = pipeline.generate(policy)
```

Output (multi-turn):
```json
{"messages": [
  {"role": "user", "content": "What's the approval process for $350?"},
  {"role": "assistant", "content": "For a $350 expense, you need manager approval..."},
  {"role": "user", "content": "What if my manager is unavailable?"},
  {"role": "assistant", "content": "You can request approval from..."}
]}
```

### Instruction

```python
pipeline = create_pipeline(dataset_type=DatasetType.INSTRUCTION)
dataset = pipeline.generate(policy)
```

Output (single-turn):
```json
{"messages": [
  {"role": "user", "content": "What's the approval process for $350?"},
  {"role": "assistant", "content": "For a $350 expense, you need manager approval. Submit the expense report with receipt..."}
]}
```

### Evaluation

Generate Q&A datasets for LLM evaluation with ground truth:

```python
pipeline = create_pipeline(dataset_type=DatasetType.EVALUATION)
dataset = pipeline.generate(policy, traces=50)

# Save in different formats
dataset.save("eval.jsonl", format="qa")         # Generic Q&A
dataset.save("eval.jsonl", format="langsmith")  # LangSmith format
dataset.save("eval.jsonl", format="langfuse")   # Langfuse format
```

Output (`format="qa"`):
```json
{
  "question": "Can I submit a $200 expense without a receipt?",
  "answer": "All expenses require receipts per policy...",
  "expected_outcome": "Deny - missing receipt violates R003",
  "ground_truth_rules": ["R003", "R005"],
  "difficulty": "negative",
  "category": "Receipt Requirements"
}
```

Output (`format="langsmith"`):
```json
{
  "inputs": {"question": "...", "context": "..."},
  "outputs": {"answer": "..."},
  "metadata": {"expected_outcome": "...", "ground_truth_rules": [...]}
}
```

Output (`format="langfuse"`):
```json
{
  "input": {"question": "...", "context": "..."},
  "expectedOutput": {"answer": "...", "expected_outcome": "..."},
  "metadata": {"ground_truth_rules": [...], "difficulty": "..."}
}
```

### Tool Calling

Generate training data for teaching models when and how to use your custom tools:

```python
from synkro import create_pipeline, ToolDefinition, DatasetType

# Define your tools
web_search = ToolDefinition(
    name="web_search",
    description="Search the web for current information",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"}
        },
        "required": ["query"]
    },
    mock_responses=["NYC: 72°F, sunny", "BTC: $67,234"]
)

# Create pipeline with tools
pipeline = create_pipeline(
    dataset_type=DatasetType.TOOL_CALL,
    tools=[web_search],
)

# Generate from tool usage guidelines
dataset = pipeline.generate("""
Use web_search for real-time data like weather, prices.
Answer general questions directly without tools.
""", traces=20)

dataset.save("tool_training.jsonl", format="tool_call")  # OpenAI format
dataset.save("tool_training.jsonl", format="chatml")     # ChatML with XML tags
```

**Output Formats:**

OpenAI function calling (`format="tool_call"`):
```json
{"messages": [
  {"role": "user", "content": "What's the weather in NYC?"},
  {"role": "assistant", "content": null, "tool_calls": [
    {"id": "call_abc", "type": "function", "function": {"name": "web_search", "arguments": "{\"query\": \"weather NYC\"}"}}
  ]},
  {"role": "tool", "tool_call_id": "call_abc", "content": "NYC: 72°F, sunny"},
  {"role": "assistant", "content": "The weather in NYC is 72°F and sunny."}
]}
```

ChatML with XML tags (`format="chatml"`):
```json
{"messages": [
  {"role": "user", "content": "What's the weather in NYC?"},
  {"role": "assistant", "content": "<tool_call>\n{\"name\": \"web_search\", \"arguments\": {\"query\": \"weather NYC\"}}\n</tool_call>"},
  {"role": "tool", "content": "<tool_response>\nNYC: 72°F, sunny\n</tool_response>"},
  {"role": "assistant", "content": "The weather in NYC is 72°F and sunny."}
]}
```

## Evaluation & Grading

Every response is graded on policy compliance, citations, and reasoning. Failed responses are automatically refined (up to N iterations).

```python
from synkro.pipelines import create_pipeline
from synkro.models.openai import OpenAI

pipeline = create_pipeline(
    model=OpenAI.GPT_4O_MINI,       # Fast generation
    grading_model=OpenAI.GPT_4O,    # Quality grading
    max_iterations=3,               # Refinement attempts
)

dataset = pipeline.generate(policy, traces=100)

# Check quality
print(f"Pass rate: {dataset.passing_rate:.1%}")

# Filter to only passing traces
high_quality = dataset.filter(passed=True)
high_quality.save("training.jsonl")
```

## Eval API

Generate test scenarios and grade your own model's responses against policy compliance.

```python
import synkro

# Generate scenarios with ground truth (no synthetic responses)
result = synkro.generate_scenarios(
    policy="Expenses over $50 require manager approval...",
    count=100,
)

# Each scenario has ground truth labels
for scenario in result.scenarios:
    print(scenario.user_message)       # "Can I expense a $200 dinner?"
    print(scenario.expected_outcome)   # "Requires manager approval per R001"
    print(scenario.target_rule_ids)    # ["R001", "R003"]
    print(scenario.scenario_type)      # "positive" | "negative" | "edge_case"

# Grade YOUR model's responses
for scenario in result.scenarios:
    response = my_model(scenario.user_message)  # Your model
    grade = synkro.grade(response, scenario, policy)

    if not grade.passed:
        print(f"Failed: {grade.feedback}")
```

### When to Use

| Use Case | API |
|----------|-----|
| Generate training data | `synkro.generate()` |
| Generate eval scenarios | `synkro.generate_scenarios()` |
| Grade external model | `synkro.grade()` |

### Scenario Types

Scenarios are generated with balanced coverage:

| Type | % | Description |
|------|---|-------------|
| `positive` | 35% | Happy path - user meets all criteria |
| `negative` | 30% | Violations - user fails one criterion |
| `edge_case` | 25% | Boundary conditions at exact limits |
| `irrelevant` | 10% | Outside policy scope |

### EvalScenario Fields

```python
scenario.user_message      # The test input
scenario.expected_outcome  # Ground truth behavior
scenario.target_rule_ids   # Rules being tested
scenario.scenario_type     # positive/negative/edge_case/irrelevant
scenario.category          # Policy category
scenario.context           # Additional context
```

### Temperature

Use `temperature` to control output diversity:

```python
# High temp for diverse scenario coverage
result = synkro.generate_scenarios(policy, temperature=0.8)

# Low temp for deterministic training data
dataset = synkro.generate(policy, temperature=0.2)
```

## Coverage Tracking

Track how well your generated scenarios cover different aspects of your policy, similar to code coverage for tests.

```python
import synkro

# Generate with logic map access
result = synkro.generate(policy, traces=50, return_logic_map=True)

# View coverage report
synkro.coverage_report(result)
```

Output:
```
Coverage Report
========================================
Overall: 68.8%
Sub-categories: 2 covered, 1 partial, 1 uncovered
Total scenarios: 20

Gaps (2):
  - Receipt requirements [HIGH] (0% coverage, 0 scenarios)
  - Travel booking rules [MEDIUM] (partial: 40% coverage)

Suggestions:
  1. Add 3+ scenarios for 'Receipt requirements' testing R008, R009
  2. Add edge_case scenarios for 'Travel booking rules'
```

### Coverage Report Formats

```python
# Print to console (default)
synkro.coverage_report(result)

# Get as dictionary for programmatic use
report = synkro.coverage_report(result, format="dict")
print(f"Coverage: {report['overall_coverage_percent']}%")
print(f"Gaps: {len(report['gaps'])}")

# Get as JSON string
json_str = synkro.coverage_report(result, format="json")

# Get raw CoverageReport object
report = synkro.coverage_report(result, format="report")
for gap in report.gaps:
    print(f"Gap: {gap}")
```

### Interactive Coverage Commands

In interactive mode, use natural language to view and improve coverage:

| Command | Action |
|---------|--------|
| `"show coverage"` | Display coverage summary |
| `"show coverage gaps"` | Show uncovered sub-categories |
| `"show heatmap"` | Visual coverage by category |
| `"increase coverage for refunds by 20%"` | Add scenarios for a sub-category |
| `"get amount thresholds to 80%"` | Target specific coverage percentage |
| `"add more negative scenarios for time eligibility"` | Add specific scenario types |

### Coverage Metrics

Each sub-category is tracked with:

| Metric | Description |
|--------|-------------|
| `coverage_percent` | % of expected coverage achieved |
| `coverage_status` | `covered` (80%+), `partial` (30-80%), `uncovered` (<30%) |
| `scenario_count` | Number of scenarios testing this sub-category |
| `type_distribution` | Breakdown by positive/negative/edge_case |

## Cost & Performance

Approximate costs using Gemini 2.5 Flash (multi-turn conversations):

| Traces | LLM Calls | Time | Cost |
|--------|-----------|------|------|
| 100 | ~335 | ~13 min | ~$3 |
| 500 | ~1,675 | ~1 hour | ~$14 |
| 1000 | ~3,350 | ~2 hours | ~$28 |

*Based on ~3.3 LLM calls per trace (generation + grading) with max_iterations=3. Actual costs vary by policy complexity and turn count.*

## Local LLMs

Run with Ollama, vLLM, or any OpenAI-compatible endpoint:

```python
from synkro import create_pipeline
from synkro.models import Local

# Ollama
pipeline = create_pipeline(model=Local.OLLAMA("llama3.2"))

# vLLM
pipeline = create_pipeline(model=Local.VLLM("mistral-7b"))

# Custom endpoint
pipeline = create_pipeline(model=Local.CUSTOM("my-model", endpoint="http://localhost:8080"))
```

**CLI:**
```bash
synkro generate policy.pdf --provider ollama --model llama3.2
synkro generate policy.pdf --provider vllm --endpoint http://localhost:8000
```

## CLI

```bash
# From file
synkro generate policy.pdf --traces 50

# From text
synkro generate "All expenses over $50 need approval" -n 20

# From URL
synkro generate https://example.com/policy -o training.jsonl

# Skip interactive mode
synkro generate policy.pdf --no-interactive

# Quick demo with built-in policy
synkro demo
```

**Options:**
- `--traces, -n` - Number of traces (default: 20)
- `--output, -o` - Output file path
- `--model, -m` - Model for generation
- `--format, -f` - Output format: `messages`, `qa`, `langsmith`, `langfuse`, `tool_call`, `chatml`
- `--provider, -p` - LLM provider for local models (`ollama`, `vllm`)
- `--endpoint, -e` - Custom API endpoint URL
- `--interactive/-i, --no-interactive/-I` - Review/edit extracted rules before generation (default: on)

## Interactive Mode

By default, synkro extracts policy rules into a Logic Map and lets you review/edit them before generation. The interactive session also shows the recommended conversation turns based on policy complexity:

```
╭─────────────────────────── Conversation Settings ────────────────────────────╮
│  Complexity:  Conditional                                                    │
│  Turns:       3                                                              │
╰──────────────────────────────────────────────────────────────────────────────╯

╭────────────────────────── 📜 Logic Map (3 rules) ────────────────────────────╮
│ ├── R001: Expenses over $50 require manager approval                         │
│ ├── R002: Client meals limited to $75/person                                 │
│ └── R003: Receipts required for all expenses                                 │
╰──────────────────────────────────────────────────────────────────────────────╯

Enter feedback: shorter conversations
✓ Set to 2 turns (User requested shorter/simpler conversations)

Enter feedback: add a rule for travel expenses
✓ Added R004: Travel expenses over $500 require VP approval

Enter feedback: done
✅ Session complete - 1 rule change(s), 2 turns
```

You can adjust both **conversation turns** and **rules** using natural language:

| Input | Action |
|-------|--------|
| `"shorter conversations"` | Reduce turns (1-2) |
| `"I want 5 turns"` | Set specific turn count |
| `"more thorough"` | Increase turns (5-6) |
| `"remove R002"` | Delete a rule |
| `"add a rule for..."` | Add new rule |

Commands: `done`, `undo`, `reset`, `show R001`, `help`

## Advanced Features

### Checkpointing

Resume interrupted generations:

```python
pipeline = create_pipeline(checkpoint_dir="./checkpoints")
dataset = pipeline.generate(policy, traces=100)  # Resumes from checkpoint
```

### Dataset Operations

```python
# Filter by quality
high_quality = dataset.filter(passed=True)

# Remove duplicates
unique = dataset.dedupe(threshold=0.85)

# Check pass rate
print(f"Pass rate: {dataset.passing_rate:.1%}")
```

### Folder Loading

Generate from multiple documents at once:

```python
from synkro.core.policy import Policy

policy = Policy.from_file("policies/")  # Loads all PDF, DOCX, TXT, MD files
dataset = pipeline.generate(policy, traces=100)
```

### Thinking Mode

Generate training data with explicit reasoning in `<think>` tags, compatible with Qwen3 and DeepSeek-R1:

```python
pipeline = create_pipeline(thinking=True)
dataset = pipeline.generate(policy, traces=50)
```

Output:
```json
{"messages": [
  {"role": "user", "content": "Can I expense a $350 team dinner?"},
  {"role": "assistant", "content": "<think>\nLet me check the expense policy...\n- Rule: Expenses over $50 require manager approval\n- $350 exceeds the $50 threshold\n- Manager approval is required\n</think>\n\nFor a $350 team dinner, you'll need manager approval since it exceeds the $50 threshold. Please submit your expense report with the receipt and request approval from your manager."}
]}
```

Works with all dataset types (`CONVERSATION`, `INSTRUCTION`, `TOOL_CALL`).

## Logic Map Inspection

Access the extracted rules programmatically:

```python
result = pipeline.generate(policy, traces=50, return_logic_map=True)

# Inspect extracted rules
for rule in result.logic_map.rules:
    print(f"{rule.rule_id}: {rule.text}")

# Get the dataset
dataset = result.dataset
```

## Development

```bash
# install dev dependencies
pip install -e ".[dev]"

# install git hooks
pre-commit install

# (optional) run on entire repo once
pre-commit run --all-files
```
