"""Ask the agent a question from the terminal.

    python -m introspective_agent.cli "My boiler shows F22. What do I do?"
    python -m introspective_agent.cli "..." --mode self_label          # self-assessment only, for comparison
    python -m introspective_agent.cli "..." --docs ~/manuals           # use your own documents
    python -m introspective_agent.cli "..." --mcp http://localhost:8001/mcp   # use a running MCP server

By default the MCP server runs in-process, so you only need the gateway (or set
GATEWAY_URL to any OpenAI-compatible endpoint).
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from introspective_agent.agent.graph import answer
from introspective_agent.agent.llm import GatewayLLM
from introspective_agent.agent.render import render
from introspective_agent.agent.tools import MCPTools
from introspective_agent.agent.tracing import Tracer
from introspective_agent.mcp_server.server import build_server


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question")
    p.add_argument("--mode", choices=["grounded", "self_label"], default="grounded")
    p.add_argument("--docs", type=Path, default=None, help="folder of .md/.txt documents (default: DOCS_DIR)")
    p.add_argument("--mcp", default=None, help="MCP server URL; default runs it in-process")
    p.add_argument("--json", action="store_true", help="print the full structured answer")
    args = p.parse_args()

    tools = MCPTools(args.mcp or build_server(args.docs))
    tracer = Tracer()
    final = asyncio.run(answer(args.question, llm=GatewayLLM(), tools=tools, mode=args.mode, tracer=tracer))
    print(final.model_dump_json(indent=2) if args.json else render(final))
    print(f"\n[trace {tracer.trace_id}] {json.dumps(tracer.summary())}  -> {tracer.path}")


if __name__ == "__main__":
    main()
