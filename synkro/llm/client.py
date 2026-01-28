"""Type-safe LLM wrapper using LiteLLM."""

import asyncio
import warnings
from typing import TYPE_CHECKING, Type, TypeVar, overload

from pydantic import BaseModel

from synkro.models import LocalModel, Model, OpenAI, get_model_string

# Lazy LiteLLM import to avoid import-time side effects (important for CLIs/tests)
_LITELLM_CONFIGURED = False


def _get_litellm():
    global _LITELLM_CONFIGURED

    import litellm
    from litellm import acompletion, completion_cost, supports_response_schema

    if not _LITELLM_CONFIGURED:
        litellm.suppress_debug_info = True
        litellm.enable_json_schema_validation = True
        litellm.drop_params = True  # Drop unsupported params (e.g., temperature for gpt-5)
        _LITELLM_CONFIGURED = True

    return litellm, acompletion, completion_cost, supports_response_schema


# Retry configuration for rate limits
MAX_RETRIES = 5
INITIAL_BACKOFF = 1.0  # seconds
MAX_BACKOFF = 60.0  # seconds


async def _retry_with_backoff(coro_func, *args, **kwargs):
    """Execute async function with exponential backoff on rate limit errors."""
    backoff = INITIAL_BACKOFF
    last_exception = None

    for attempt in range(MAX_RETRIES):
        try:
            return await coro_func(*args, **kwargs)
        except Exception as e:
            error_str = str(e).lower()
            # Check for rate limit errors (429, resource exhausted, rate limit)
            is_rate_limit = (
                "429" in error_str
                or "rate" in error_str
                or "resource exhausted" in error_str
                or "ratelimit" in error_str
            )
            if not is_rate_limit:
                raise  # Not a rate limit error, don't retry

            last_exception = e
            if attempt < MAX_RETRIES - 1:
                # Exponential backoff with jitter
                import random

                jitter = random.uniform(0, backoff * 0.1)
                sleep_time = min(backoff + jitter, MAX_BACKOFF)
                await asyncio.sleep(sleep_time)
                backoff *= 2  # Exponential increase

    # All retries exhausted
    raise last_exception


if TYPE_CHECKING:
    from synkro.types.metrics import Metrics, TrackedLLM

# Suppress Pydantic serialization warnings from litellm response types
warnings.filterwarnings(
    "ignore",
    message=".*Pydantic serializer warnings.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Expected `Message`.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*Expected `StreamingChoices`.*",
    category=UserWarning,
)

T = TypeVar("T", bound=BaseModel)


class LLM:
    """
    Type-safe LLM wrapper using LiteLLM for universal provider support.

    Supports structured outputs via native JSON mode for reliable responses.

    Supported providers: OpenAI, Anthropic, Google (Gemini), Local (Ollama, vLLM)

    Examples:
        >>> from synkro import LLM, OpenAI, Anthropic, Google, Local

        # Use OpenAI
        >>> llm = LLM(model=OpenAI.GPT_4O_MINI)
        >>> response = await llm.generate("Hello!")

        # Use Anthropic
        >>> llm = LLM(model=Anthropic.CLAUDE_35_SONNET)

        # Use Google Gemini
        >>> llm = LLM(model=Google.GEMINI_25_FLASH)

        # Use local Ollama
        >>> llm = LLM(model=Local.OLLAMA("llama3.1"))

        # Use local vLLM
        >>> llm = LLM(model=Local.VLLM("mistral"))

        # Structured output
        >>> class Output(BaseModel):
        ...     answer: str
        ...     confidence: float
        >>> result = await llm.generate_structured("What is 2+2?", Output)
        >>> result.answer
        '4'
    """

    def __init__(
        self,
        model: Model = OpenAI.GPT_4O_MINI,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ):
        """
        Initialize the LLM client.

        Args:
            model: Model to use (enum, LocalModel, or string)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate (default: None = model's max)
            api_key: Optional API key override
            base_url: Optional API base URL (auto-set when using Local models)
        """
        # Handle LocalModel - extract endpoint automatically
        if isinstance(model, LocalModel):
            self.model = f"{model.provider}/{model.model}"
            self._base_url = model.endpoint
        else:
            self.model = get_model_string(model)
            self._base_url = base_url

        self.temperature = temperature
        self.max_tokens = max_tokens
        self._api_key = api_key

        # Cost and usage tracking
        self._total_cost = 0.0
        self._call_count = 0

    async def generate(self, prompt: str, system: str | None = None) -> str:
        """
        Generate a text response.

        Args:
            prompt: The user prompt
            system: Optional system prompt

        Returns:
            Generated text response
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "api_key": self._api_key,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self._base_url:
            kwargs["api_base"] = self._base_url

        _, acompletion, _, _ = _get_litellm()
        response = await _retry_with_backoff(acompletion, **kwargs)
        self._track_cost(response)
        return response.choices[0].message.content

    async def generate_batch(self, prompts: list[str], system: str | None = None) -> list[str]:
        """
        Generate responses for multiple prompts in parallel.

        Args:
            prompts: List of user prompts
            system: Optional system prompt for all

        Returns:
            List of generated responses
        """
        import asyncio

        tasks = [self.generate(p, system) for p in prompts]
        return await asyncio.gather(*tasks)

    @overload
    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system: str | None = None,
        temperature: float | None = None,
    ) -> T: ...

    @overload
    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[list[T]],
        system: str | None = None,
        temperature: float | None = None,
    ) -> list[T]: ...

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[T] | Type[list[T]],
        system: str | None = None,
        temperature: float | None = None,
    ) -> T | list[T]:
        """
        Generate a structured response matching a Pydantic model.

        Uses LiteLLM's native JSON mode with response_format for
        reliable structured outputs.

        Args:
            prompt: The user prompt
            response_model: Pydantic model class for the response
            system: Optional system prompt

        Returns:
            Parsed response matching the model

        Example:
            >>> class Analysis(BaseModel):
            ...     sentiment: str
            ...     score: float
            >>> result = await llm.generate_structured(
            ...     "Analyze: I love this product!",
            ...     Analysis
            ... )
            >>> result.sentiment
            'positive'
        """
        # Check if model supports structured outputs
        _, _, _, supports_response_schema = _get_litellm()
        if not supports_response_schema(model=self.model, custom_llm_provider=None):
            raise ValueError(
                f"Model '{self.model}' does not support structured outputs (response_format). "
                f"Use a model that supports JSON schema like GPT-4o, Gemini 1.5+, or Claude 3.5+."
            )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        # Use LiteLLM's native response_format with Pydantic model
        kwargs = {
            "model": self.model,
            "messages": messages,
            "response_format": response_model,
            "temperature": temperature if temperature is not None else self.temperature,
            "api_key": self._api_key,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self._base_url:
            kwargs["api_base"] = self._base_url

        _, acompletion, _, _ = _get_litellm()
        response = await _retry_with_backoff(acompletion, **kwargs)
        self._track_cost(response)
        return response_model.model_validate_json(response.choices[0].message.content)

    async def generate_chat(
        self, messages: list[dict], response_model: Type[T] | None = None
    ) -> str | T:
        """
        Generate a response for a full conversation.

        Args:
            messages: List of message dicts with 'role' and 'content'
            response_model: Optional Pydantic model for structured output

        Returns:
            Generated response (string or structured)
        """
        if response_model:
            # Check if model supports structured outputs
            _, _, _, supports_response_schema = _get_litellm()
            if not supports_response_schema(model=self.model, custom_llm_provider=None):
                raise ValueError(
                    f"Model '{self.model}' does not support structured outputs (response_format). "
                    f"Use a model that supports JSON schema like GPT-4o, Gemini 1.5+, or Claude 3.5+."
                )

            # Use LiteLLM's native response_format with Pydantic model
            kwargs = {
                "model": self.model,
                "messages": messages,
                "response_format": response_model,
                "temperature": self.temperature,
                "api_key": self._api_key,
            }
            if self.max_tokens is not None:
                kwargs["max_tokens"] = self.max_tokens
            if self._base_url:
                kwargs["api_base"] = self._base_url

            _, acompletion, _, _ = _get_litellm()
            response = await _retry_with_backoff(acompletion, **kwargs)
            self._track_cost(response)
            return response_model.model_validate_json(response.choices[0].message.content)

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "api_key": self._api_key,
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if self._base_url:
            kwargs["api_base"] = self._base_url

        _, acompletion, _, _ = _get_litellm()
        response = await _retry_with_backoff(acompletion, **kwargs)
        self._track_cost(response)
        return response.choices[0].message.content

    def _track_cost(self, response) -> None:
        """Track cost and call count from a response."""
        self._call_count += 1
        try:
            _, _, completion_cost, _ = _get_litellm()
            cost = completion_cost(completion_response=response)
            self._total_cost += cost
        except Exception:
            # Some models may not have pricing info
            pass

    @property
    def total_cost(self) -> float:
        """Get total cost of all LLM calls made by this client."""
        return self._total_cost

    @property
    def call_count(self) -> int:
        """Get total number of LLM calls made by this client."""
        return self._call_count

    def reset_tracking(self) -> None:
        """Reset cost and call tracking."""
        self._total_cost = 0.0
        self._call_count = 0

    def reset_call_count(self) -> None:
        """Reset only the call count, preserving cost tracking."""
        self._call_count = 0

    def with_metrics(self, metrics: "Metrics", phase: str) -> "TrackedLLM":
        """Create a tracked wrapper that records to an external Metrics object.

        Use this to track costs across multiple LLM clients into a unified
        Metrics instance.

        Args:
            metrics: External Metrics object to track to
            phase: Phase name for tracking (e.g., "extraction", "scenarios")

        Returns:
            TrackedLLM wrapper that proxies calls and records metrics

        Examples:
            >>> metrics = Metrics()
            >>> tracked = llm.with_metrics(metrics, phase="extraction")
            >>> result = await tracked.generate("Hello")
            >>> print(metrics.get_phase("extraction").cost)
        """
        from synkro.types.metrics import TrackedLLM

        return TrackedLLM(self, metrics, phase)
