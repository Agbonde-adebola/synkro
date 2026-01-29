"""Cerebras models.

Docs: https://docs.litellm.ai/docs/providers/cerebras

Note: Only gpt-oss-120b supports structured outputs, which is required for Synkro.
See: https://inference-docs.cerebras.ai/capabilities/structured-outputs
"""

from enum import Enum


class Cerebras(str, Enum):
    """Cerebras models via LiteLLM.

    Only models supporting structured outputs are included.
    """

    GPT_OSS_120B = "cerebras/gpt-oss-120b"
