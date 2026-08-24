"""LLM completion wrapper with thoughts/output split (Gemini + OpenAI).

Uses non-streaming completions (reliable) then splits the response on a
``>>>`` marker. The caller emits text word-by-word to the UI for the
streaming visual effect — indistinguishable from real streaming, but without
tag-splitting fragility.

Provider is chosen by model name: ``gpt-*`` → OpenAI, anything else → Gemini.
Set ``DEMO_MODEL`` to switch (e.g. ``DEMO_MODEL=gpt-5-mini``).
"""

from __future__ import annotations

import os
import time
from typing import Callable, Optional

try:
    from google import genai
    from google.genai import types as gtypes
except ImportError:
    genai = None
    gtypes = None

try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None

DEFAULT_MODEL = os.environ.get("DEMO_MODEL", "gemini-3.7-flash")
_SPLIT_MARKER = ">>>"

_client = None
_openai_client = None

# Usage listener: the demo server registers a callback here. Each LLM
# invocation reports its model, the token counts from the provider usage
# fields, and its latency. The UI builds its call, token, and cost
# counters from these reports.
_usage_emitter: Optional[Callable[[dict], None]] = None


def set_usage_emitter(fn: Optional[Callable[[dict], None]]) -> None:
    global _usage_emitter
    _usage_emitter = fn


def clear_usage_emitter() -> None:
    global _usage_emitter
    _usage_emitter = None


def _report_usage(model: str, input_tokens: int, output_tokens: int,
                  latency_ms: float, error: bool = False) -> None:
    if _usage_emitter is None:
        return
    try:
        _usage_emitter({
            "model": model,
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "latency_ms": round(latency_ms, 1),
            "error": error,
        })
    except Exception:
        pass


def _get_client():
    global _client
    if _client is None:
        if genai is None:
            raise RuntimeError("google-genai not installed")
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        _client = genai.Client(api_key=api_key)
    return _client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        if _OpenAI is None:
            raise RuntimeError("openai package not installed")
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY not set")
        _openai_client = _OpenAI()
    return _openai_client


def _complete_openai(*, system: str, user: str, model: str,
                     temperature: float, max_output_tokens: int,
                     json_mode: bool = False) -> str:
    client = _get_openai_client()
    kwargs: dict = dict(
        model=model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        # Reasoning models spend reasoning tokens inside this budget, the
        # same as Gemini thinking tokens. Give the visible output more
        # headroom.
        max_completion_tokens=max(1024, max_output_tokens * 2),
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if model.startswith("gpt-5"):
        # The gpt-5 family rejects a custom temperature. Keep the reasoning
        # effort minimal so the budget goes to the visible output.
        kwargs["reasoning_effort"] = "minimal"
    else:
        kwargs["temperature"] = temperature
    started = time.perf_counter()
    try:
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception:
            # Older/newer models disagree on optional params — retry bare.
            kwargs.pop("reasoning_effort", None)
            kwargs.pop("temperature", None)
            resp = client.chat.completions.create(**kwargs)
        usage = getattr(resp, "usage", None)
        _report_usage(
            model,
            getattr(usage, "prompt_tokens", 0) if usage else 0,
            getattr(usage, "completion_tokens", 0) if usage else 0,
            (time.perf_counter() - started) * 1000,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        _report_usage(model, 0, 0, (time.perf_counter() - started) * 1000,
                      error=True)
        return f"[LLM error: {e}]"


def complete(
    *,
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_output_tokens: int = 2048,
    thinking: bool = False,
    json_mode: bool = False,
) -> str:
    """Non-streaming completion. Returns full text.

    ``json_mode`` constrains the response to valid JSON at the provider
    level (Gemini response_mime_type / OpenAI response_format).
    """
    if model.startswith("gpt-"):
        return _complete_openai(system=system, user=user, model=model,
                                temperature=temperature,
                                max_output_tokens=max_output_tokens,
                                json_mode=json_mode)
    client = _get_client()
    config_kwargs: dict = dict(
        system_instruction=system,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
    )
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"
    # Gemini models spend "thinking" tokens inside max_output_tokens. With
    # thinking on, a small budget can return a truncated fragment or
    # nothing. The demo needs visible output, so thinking is off by
    # default where the model permits it (the flash family; pro models
    # reject a zero budget). ``thinking=True`` turns it on for Gemini 3
    # models at the low level, and raises the token cap so thought tokens
    # do not truncate the visible output.
    if thinking and model.startswith("gemini-3"):
        config_kwargs["thinking_config"] = gtypes.ThinkingConfig(
            thinking_level="low")
        config_kwargs["max_output_tokens"] = max(4096, max_output_tokens * 2)
    elif "flash" in model or "lite" in model:
        config_kwargs["thinking_config"] = gtypes.ThinkingConfig(thinking_budget=0)
    config = gtypes.GenerateContentConfig(**config_kwargs)
    started = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=model, contents=user, config=config,
        )
        um = getattr(resp, "usage_metadata", None)
        out_tokens = 0
        if um is not None:
            out_tokens = ((getattr(um, "candidates_token_count", 0) or 0)
                          + (getattr(um, "thoughts_token_count", 0) or 0))
        _report_usage(
            model,
            (getattr(um, "prompt_token_count", 0) or 0) if um else 0,
            out_tokens,
            (time.perf_counter() - started) * 1000,
        )
        return (resp.text or "").strip()
    except Exception as e:
        _report_usage(model, 0, 0, (time.perf_counter() - started) * 1000,
                      error=True)
        return f"[LLM error: {e}]"


def complete_and_split(
    *,
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_output_tokens: int = 2048,
    thinking: bool = False,
) -> tuple[str, str]:
    """Complete with a reasoning/action split.

    The system prompt is augmented to ask the model to reason first, then emit
    ``>>>`` and give its action. Returns ``(thoughts, output)``.

    Fallback if no marker: first ~60% → thoughts, rest → output.
    """
    augmented_system = (
        f"{system}\n\n"
        f"First, briefly reason about the task (2-4 sentences).\n"
        f"Then on a new line write '{_SPLIT_MARKER}' and provide your "
        f"final answer/action after the marker.\n"
        f"Keep your total response under {max_output_tokens // 4} words."
    )

    text = complete(
        system=augmented_system,
        user=user,
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking=thinking,
    )

    idx = text.find(_SPLIT_MARKER)
    if idx >= 0:
        thoughts = text[:idx].strip()
        output = text[idx + len(_SPLIT_MARKER):].strip()
    elif "\n\n" in text:
        # Fallback: split at first double-newline
        parts = text.split("\n\n", 1)
        thoughts, output = parts[0].strip(), parts[1].strip()
    else:
        # Fallback: 60/40 split
        split_at = int(len(text) * 0.6)
        thoughts = text[:split_at].rsplit(" ", 1)[0].strip()
        output = text[split_at:].strip()

    return thoughts, output


def split_into_tokens(text: str, chunk_size: int = 3) -> list[str]:
    """Split text into small chunks for streaming visual effect.

    Groups ~3 words per chunk so the UI animates smoothly.
    """
    words = text.split(" ")
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if i + chunk_size < len(words):
            chunk += " "
        chunks.append(chunk)
    return chunks if chunks else [text]
