# Synkro

Turn policies, handbooks, and documentation into high-quality training data for fine-tuning LLMs.

## Features

- **Quality Evaluation** - Each response is graded and automatically refined if it fails
- **Multiple Formats** - Conversation (multi-turn), Instruction (single-turn), and Tool Calling
- **Tool Call Training** - Generate OpenAI function calling format for teaching models to use custom tools
- **Top LLM Providers** - OpenAI, Anthropic, and Google
- **File Support** - PDF, DOCX, TXT, Markdown, URLs
- **CLI Included** - Generate datasets from the command line

## Installation

```bash
pip install synkro
```

## Quick Start

```python
from synkro.pipelines import create_pipeline
from synkro.models.google import Google
from synkro.types import DatasetType

pipeline = create_pipeline(
    model=Google.GEMINI_25_FLASH,          # Fast generation
    grading_model=Google.GEMINI_25_PRO,    # Quality grading
    dataset_type=DatasetType.CONVERSATION,
)

dataset = pipeline.generate(
    "All expenses over $50 require manager approval.",
    traces=50,
)
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
| **CONVERSATION** | Multi | messages | Fine-tuning chat models |
| **INSTRUCTION** | 1 | messages | Instruction-following models |
| **TOOL_CALL** | Multi | OpenAI function calling, ChatML | Teaching tool use |

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
- `--format, -f` - Output format: `sft` (default) or `qa`
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