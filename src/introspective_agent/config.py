"""Settings, read from environment variables (or a .env file you `source`).

Deliberately plain: no settings framework, so it's obvious where each value comes from.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DOCS = Path(__file__).resolve().parent / "mcp_server" / "sample_docs"


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    # --- Model gateway (LiteLLM proxy, OpenAI-compatible) -------------------------
    # The agent never talks to a model provider directly. It talks to the gateway,
    # which handles provider keys, routing, fallbacks, budgets and logging.
    gateway_url: str = field(default_factory=lambda: _env("GATEWAY_URL", "http://localhost:4000"))
    gateway_key: str = field(default_factory=lambda: _env("GATEWAY_KEY", "sk-local-dev"))
    # Model *aliases* defined in gateway/litellm_config.yaml, not provider model names.
    agent_model: str = field(default_factory=lambda: _env("AGENT_MODEL", "agent-model"))
    judge_model: str = field(default_factory=lambda: _env("JUDGE_MODEL", "judge-model"))

    # --- MCP server ---------------------------------------------------------------
    mcp_url: str = field(default_factory=lambda: _env("MCP_URL", "http://localhost:8001/mcp"))

    # --- Knowledge base: a folder of .md / .txt files (manuals, notes) --------------
    docs_dir: Path = field(default_factory=lambda: Path(_env("DOCS_DIR", str(SAMPLE_DOCS))))

    # --- Agent behaviour ---------------------------------------------------------
    max_draft_attempts: int = field(default_factory=lambda: int(_env("MAX_DRAFT_ATTEMPTS", "2")))
    max_tool_calls: int = field(default_factory=lambda: int(_env("MAX_TOOL_CALLS", "6")))
    max_question_chars: int = field(default_factory=lambda: int(_env("MAX_QUESTION_CHARS", "1000")))

    # --- Observability ---------------------------------------------------------------
    trace_dir: Path = field(default_factory=lambda: Path(_env("TRACE_DIR", str(PROJECT_ROOT / "traces"))))


settings = Settings()
