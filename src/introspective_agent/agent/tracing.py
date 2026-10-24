"""Minimal tracing: one JSONL file per run, one line per span or event.

This shows *what* observability needs to capture (per-node latency, tool calls, token
usage, guardrail decisions, label changes) without a dependency. The natural upgrade is
OpenTelemetry spans exported to Langfuse, Arize Phoenix or similar. The `span` and
`event` calls map one-to-one onto OTel concepts, so the swap is mechanical.
"""

from __future__ import annotations

import contextvars
import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from introspective_agent.config import settings


class Tracer:
    def __init__(self, trace_id: str | None = None, out_dir: Path | None = None, write: bool = True):
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.records: list[dict[str, Any]] = []
        self.path = (out_dir or settings.trace_dir) / f"{self.trace_id}.jsonl" if write else None

    def _emit(self, record: dict[str, Any]) -> None:
        record = {"trace_id": self.trace_id, "ts": round(time.time(), 3), **record}
        self.records.append(record)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(record, default=str) + "\n")

    def event(self, name: str, **attrs: Any) -> None:
        self._emit({"type": "event", "name": name, **attrs})

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[dict[str, Any]]:
        extra: dict[str, Any] = {}
        t0 = time.perf_counter()
        status = "ok"
        try:
            yield extra
        except Exception as e:
            status = f"error: {type(e).__name__}: {e}"
            raise
        finally:
            self._emit({
                "type": "span", "name": name, "status": status,
                "duration_ms": round((time.perf_counter() - t0) * 1000, 1), **attrs, **extra,
            })

    def summary(self) -> dict[str, Any]:
        llm = [r for r in self.records if r.get("name") == "llm_call"]
        return {
            "trace_id": self.trace_id,
            "llm_calls": len(llm),
            "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in llm),
            "completion_tokens": sum(r.get("completion_tokens") or 0 for r in llm),
            "tool_calls": sum(1 for r in self.records if r.get("name") == "tool_call"),
            "duration_ms": sum(r["duration_ms"] for r in self.records if r.get("type") == "span" and r["name"].startswith("node:")),
        }


_NULL = Tracer(trace_id="untraced", write=False)
_current: contextvars.ContextVar[Tracer] = contextvars.ContextVar("tracer", default=_NULL)


def current_tracer() -> Tracer:
    return _current.get()


@contextmanager
def use_tracer(tracer: Tracer) -> Iterator[Tracer]:
    token = _current.set(tracer)
    try:
        yield tracer
    finally:
        _current.reset(token)
