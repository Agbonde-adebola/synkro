# Synkro

Turn policies, handbooks, and documentation into high-quality training data for fine-tuning LLMs.

## Features

- **Quality Evaluation** - Each response is graded and automatically refined if it fails
- **Multiple Formats** - SFT (chat) and QA (question-answer)
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
    dataset_type=DatasetType.SFT,
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

| Format | Output | Best For |
|--------|--------|----------|
| **SFT** | Chat messages | Fine-tuning chat models |
| **QA** | Question-answer pairs | RAG systems, knowledge bases |

### SFT (Default)

```python
from synkro.types import DatasetType

pipeline = create_pipeline(dataset_type=DatasetType.SFT)
dataset = pipeline.generate(policy)
```

Output:
```json
{"messages": [
  {"role": "system", "content": "You are a policy expert..."},
  {"role": "user", "content": "What's the approval process for $350?"},
  {"role": "assistant", "content": "For a $350 expense, you need..."}
]}
```

### QA

```python
pipeline = create_pipeline(dataset_type=DatasetType.QA)
```

Output:
```json
{"question": "What's the approval process?", "answer": "You need...", "context": "..."}
```

## Evaluation & Grading

Every response is graded on policy compliance, citations, and reasoning. Failed responses are automatically refined (up to 3 iterations).

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

## CLI

```bash
# From file
synkro generate policy.pdf --traces 50 --format sft

# From text
synkro generate "All expenses over $50 need approval" -n 20

# From URL
synkro generate https://example.com/policy -o training.jsonl
```

**Options:**
- `--traces, -n` - Number of traces (default: 20)
- `--format, -f` - Output format: sft or qa (default: sft)
- `--output, -o` - Output file path
- `--model, -m` - Model for generation