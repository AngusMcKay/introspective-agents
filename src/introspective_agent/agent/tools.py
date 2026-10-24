"""MCP client wrapper: how the agent reaches its tools.

`target` can be a URL (Streamable HTTP, e.g. the MCP server container) or an
`MCPServer` instance (in-process, used by tests and the eval runner so they need no
running server).

Each call opens a fresh MCP session. Simple and robust; for throughput you'd hold one
session open for the lifetime of a request or a worker.
"""

from __future__ import annotations

import json
from typing import Any

from mcp import Client

from introspective_agent.agent.tracing import current_tracer


class MCPTools:
    def __init__(self, target: Any):
        self.target = target
        self._tool_cache: list[dict[str, Any]] | None = None

    async def list_tools(self) -> list[dict[str, Any]]:
        if self._tool_cache is None:
            async with Client(self.target) as client:
                result = await client.list_tools()
            self._tool_cache = [
                {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
                for t in result.tools
            ]
        return self._tool_cache

    async def call(self, name: str, arguments: dict[str, Any]) -> tuple[Any, bool]:
        """Return (content, is_error)."""
        tracer = current_tracer()
        with tracer.span("tool_call_span", tool=name):
            async with Client(self.target) as client:
                result = await client.call_tool(name, arguments)
        tracer.event("tool_call", tool=name, arguments=arguments, is_error=result.is_error)
        if result.is_error:
            text = " ".join(getattr(c, "text", "") for c in result.content)
            return {"error": text}, True
        if result.structured_content is not None:
            return result.structured_content, False
        text = " ".join(getattr(c, "text", "") for c in result.content)
        try:
            return json.loads(text), False
        except json.JSONDecodeError:
            return {"text": text}, False
