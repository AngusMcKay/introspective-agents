"""LLM access.

`GatewayLLM` talks to the LiteLLM gateway using the OpenAI-compatible API. Because the
gateway exposes one API for every provider, swapping Claude for GPT or a Bedrock model
is a config change in gateway/litellm_config.yaml, not a code change.

`FakeLLM` returns scripted responses so the whole graph can be tested offline and in CI
without API keys or cost.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from introspective_agent.agent.tracing import current_tracer
from introspective_agent.config import settings

T = TypeVar("T", bound=BaseModel)


class LLM(Protocol):
    async def complete_json(self, *, task: str, system: str, user: str, schema: type[T], model: str | None = None) -> T: ...


def _extract_json(text: str) -> str:
    """Models sometimes wrap JSON in prose or ``` fences despite instructions."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text


def _schema_instructions(schema: type[BaseModel]) -> str:
    return (
        "\n\nRespond with a single JSON object and nothing else. It must validate against this JSON Schema:\n"
        + json.dumps(schema.model_json_schema())
    )


class GatewayLLM:
    def __init__(self, base_url: str | None = None, api_key: str | None = None, max_retries: int = 1):
        from openai import AsyncOpenAI  # imported lazily so tests don't need the package configured

        self.client = AsyncOpenAI(base_url=base_url or settings.gateway_url, api_key=api_key or settings.gateway_key)
        self.max_retries = max_retries

    async def complete_json(self, *, task: str, system: str, user: str, schema: type[T], model: str | None = None) -> T:
        model = model or settings.agent_model
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system + _schema_instructions(schema)},
            {"role": "user", "content": user},
        ]
        tracer = current_tracer()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            t0 = time.perf_counter()
            resp = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
                # LiteLLM reads this for per-request logging/cost attribution in the gateway.
                extra_body={"metadata": {"task": task}},
            )
            text = resp.choices[0].message.content or ""
            usage = resp.usage
            tracer.event(
                "llm_call",
                task=task,
                model=model,
                attempt=attempt,
                latency_ms=round((time.perf_counter() - t0) * 1000),
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
            )
            try:
                return schema.model_validate_json(_extract_json(text))
            except ValidationError as e:
                last_error = e
                tracer.event("llm_schema_error", task=task, attempt=attempt, error=str(e)[:500])
                messages += [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"That did not validate against the schema:\n{e}\nReturn corrected JSON only."},
                ]
        raise RuntimeError(f"LLM output for task '{task}' failed validation after retries") from last_error


Responder = Callable[[str, str], dict[str, Any] | BaseModel]


class FakeLLM:
    """Scripted LLM for tests. Map task name -> dict, model instance, or fn(system, user) -> dict."""

    def __init__(self, responses: dict[str, dict[str, Any] | BaseModel | Responder]):
        self.responses = responses
        self.calls: list[tuple[str, str, str]] = []  # (task, system, user)

    async def complete_json(self, *, task: str, system: str, user: str, schema: type[T], model: str | None = None) -> T:
        self.calls.append((task, system, user))
        current_tracer().event("llm_call", task=task, model="fake")
        if task not in self.responses:
            raise KeyError(f"FakeLLM has no response for task {task!r}")
        r = self.responses[task]
        if callable(r) and not isinstance(r, BaseModel):
            r = r(system, user)
        if isinstance(r, BaseModel):
            r = r.model_dump(mode="json")
        return schema.model_validate(r)
