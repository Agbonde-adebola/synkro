"""Cerebras models.

Docs: https://docs.litellm.ai/docs/providers/cerebras
"""

from enum import Enum


class Cerebras(str, Enum):
    """Cerebras models via LiteLLM."""

    LLAMA_33_70B = "cerebras/llama-3.3-70b"
    GPT_OSS_120B = "cerebras/gpt-oss-120b"
    QWEN_3_32B = "cerebras/qwen-3-32b"
