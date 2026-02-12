from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional

from openai import OpenAI, AsyncOpenAI
from google import genai


Provider = Literal["openai", "google"]


@dataclass(frozen=True)
class ModelInfo:
    """Simple registry entry for supported models."""

    model: str
    provider: Provider
    aliases: List[str] = field(default_factory=list)
    description: Optional[str] = None


# Keep friendly keys stable for DVC params and CLI flags.
# The canonical model names reflect current API model identifiers.
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "gpt-5-mini": ModelInfo(
        model="gpt-5-mini",
        provider="openai",
        aliases=["gpt5-mini", "gpt5 mini", "gpt-5 mini"],
        description="OpenAI lightweight model with native JSON mode.",
    ),
    "gpt-5.2": ModelInfo(
        model="gpt-5.2",
        provider="openai",
        aliases=["gpt5.2", "gpt5 2"],
        description="Flagship OpenAI reasoning & generation model.",
    ),
    "gpt-4.1-mini": ModelInfo(
        model="gpt-4.1-mini",
        provider="openai",
        aliases=["gpt4.1-mini", "gpt-4.1mini"],
        description="Cost‑efficient OpenAI baseline compatible with structured outputs.",
    ),
    "gemini-3-flash-preview": ModelInfo(
        model="gemini-3-flash-preview",
        provider="google",
        aliases=["gemini-3-flash", "gemini3-flash"],
        description="Latest Gemini Flash 3.0 preview model.",
    ),
    "gemini-2.5-flash-lite": ModelInfo(
        model="gemini-2.5-flash-lite",
        provider="google",
        aliases=["gemini-2.5-flash-lite", "gemini2.5-flash-lite", "gemini-2.5-lite"],
        description="Gemini 2.5 Flash lite variant.",
    ),
    "gemini-2.5-flash": ModelInfo(
        model="gemini-2.5-flash",
        provider="google",
        aliases=["gemini-2.5-flash-exp"],
        description="Gemini 2.5 Flash with JSON schema support.",
    ),
    "gemma-3-27b-it": ModelInfo(
        model="gemma-3-27b-it",
        provider="google",
        aliases=["gemma-3-27b", "gemma-3-27b-instruct"],
        description="Gemma 3 instruction‑tuned model.",
    ),
}


def resolve_model(name: str) -> ModelInfo:
    key = name.lower().strip()
    for canonical, info in MODEL_REGISTRY.items():
        if key == canonical or key in info.aliases:
            return info
    raise ValueError(
        f"Unknown model '{name}'. Supported options: {', '.join(MODEL_REGISTRY.keys())}"
    )


def build_client(
    model_info: ModelInfo,
    api_key_env: Optional[str] = None,
    base_url_env: Optional[str] = None,
    async_mode: bool = False,
):
    """Return the correct SDK client for the given model/provider."""
    if model_info.provider == "openai":
        env_var = api_key_env or "OPENAI_API_KEY"
        api_key = os.getenv(env_var)

        base_url = os.getenv(base_url_env) if base_url_env else None
        # Convenience for Azure OpenAI: if caller uses the Azure key env and
        # doesn't pass base_url_env explicitly, pick up the standard endpoint env.
        if not base_url and env_var == "AZURE_PROJECT_API_KEY":
            base_url = os.getenv("AZURE_OPENAI_ENDPOINT")

        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        return AsyncOpenAI(**kwargs) if async_mode else OpenAI(**kwargs)

    env_var = api_key_env or "GEMINI_API_KEY"
    api_key = os.getenv(env_var)

    # google-genai client covers Gemini + Gemma
    return genai.Client(api_key=api_key)

def supported_model_names() -> Iterable[str]:
    return MODEL_REGISTRY.keys()
