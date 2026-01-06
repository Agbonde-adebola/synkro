"""Local LLM providers (Ollama, vLLM, etc.)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LocalModel:
    """Represents a local model configuration.

    Attributes:
        provider: The provider name (ollama, vllm, openai)
        model: The model name
        endpoint: The API endpoint URL
    """

    provider: str
    model: str
    endpoint: str

    def __str__(self) -> str:
        return f"{self.provider}/{self.model}"


class Local:
    """Factory for local LLM configurations.

    Provides a clean API for configuring local LLM providers like Ollama and vLLM.
    Returns LocalModel instances that the LLM client can use to configure the
    connection automatically.

    Examples:
        >>> from synkro import LLM, Local

        # Ollama (default localhost:11434)
        >>> llm = LLM(model=Local.OLLAMA("llama3.1"))

        # vLLM (default localhost:8000)
        >>> llm = LLM(model=Local.VLLM("mistral"))

        # Custom endpoint
        >>> llm = LLM(model=Local.OLLAMA("llama3.1", endpoint="http://server:11434"))

        # Any OpenAI-compatible server
        >>> llm = LLM(model=Local.CUSTOM("my-model", endpoint="http://localhost:8080/v1"))
    """

    DEFAULT_ENDPOINTS = {
        "ollama": "http://localhost:11434",
        "vllm": "http://localhost:8000",
        "openai": "http://localhost:8000/v1",
    }

    @classmethod
    def OLLAMA(cls, model: str, endpoint: str | None = None) -> LocalModel:
        """Create Ollama model config.

        Args:
            model: Model name (e.g., "llama3.1", "mistral", "codellama")
            endpoint: Optional custom endpoint (default: http://localhost:11434)

        Returns:
            LocalModel configured for Ollama
        """
        return LocalModel(
            provider="ollama",
            model=model,
            endpoint=endpoint or cls.DEFAULT_ENDPOINTS["ollama"],
        )

    @classmethod
    def VLLM(cls, model: str, endpoint: str | None = None) -> LocalModel:
        """Create vLLM model config.

        Args:
            model: Model name (e.g., "mistral-7b", "llama-2-13b")
            endpoint: Optional custom endpoint (default: http://localhost:8000)

        Returns:
            LocalModel configured for vLLM
        """
        return LocalModel(
            provider="vllm",
            model=model,
            endpoint=endpoint or cls.DEFAULT_ENDPOINTS["vllm"],
        )

    @classmethod
    def CUSTOM(cls, model: str, endpoint: str) -> LocalModel:
        """Create custom local model config for any OpenAI-compatible server.

        Args:
            model: Model name
            endpoint: API endpoint URL (required)

        Returns:
            LocalModel configured for OpenAI-compatible API
        """
        return LocalModel(
            provider="openai",
            model=model,
            endpoint=endpoint,
        )
