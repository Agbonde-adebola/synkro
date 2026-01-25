# Synkro

![](https://static.scarf.sh/a.png?x-pxid=f08f2a53-e0cf-4291-83f4-b518f620bf69)
[![PyPI version](https://img.shields.io/pypi/v/synkro.svg?cacheSeconds=3600)](https://pypi.org/project/synkro/)
[![Downloads](https://static.pepy.tech/badge/synkro)](https://pepy.tech/project/synkro)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-synkro.sh-purple.svg)](https://synkro.sh/docs)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

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
    model=Google.GEMINI_25_FLASH,
    grading_model=Google.GEMINI_25_PRO,
    dataset_type=DatasetType.CONVERSATION,
)

dataset = pipeline.generate(EXPENSE_POLICY, traces=50)
dataset.save("training.jsonl")
```

Or use the CLI:

```bash
synkro generate policy.pdf --traces 50

# Quick demo with built-in policy
synkro demo
```

## Features

- **Multiple dataset types** - Conversation, Instruction, Evaluation, Tool Calling
- **Auto grading & refinement** - Responses graded and refined until passing
- **Coverage tracking** - Track scenario diversity, identify gaps
- **Eval platform export** - LangSmith, Langfuse, Q&A formats
- **Any LLM** - OpenAI, Anthropic, Google, Ollama, vLLM
- **Any document** - PDF, DOCX, TXT, Markdown, URLs

## Documentation

Full documentation at **[synkro.sh/docs](https://synkro.sh/docs)**

- [Quickstart](https://synkro.sh/docs/quickstart)
- [Dataset Types](https://synkro.sh/docs/datasets/conversation)
- [Coverage Tracking](https://synkro.sh/docs/concepts/coverage)
- [Tool Calling](https://synkro.sh/docs/guides/tool-calling)
- [API Reference](https://synkro.sh/docs/api-reference/overview)
