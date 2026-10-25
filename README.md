# Introspective agents

A small, complete AI engineering project. It answers practical questions (DIY, household, mechanical) from internal knowledge and/or a knowledge base of manuals and notes, and labels every claim it makes by how well it is supported:

| Label | Meaning |
|---|---|
| **verified** | Stated in a retrieved source, with a citation |
| **derived** | Calculated or reasoned from retrieved sources, with the steps shown |
| **inherent_knowledge** | A strong, well-established fact from the model's trained knowledge; no source retrieved |
| **derived_inherent** | Reasoned from trained knowledge, or recalled with less confidence; not a strong fact |
| **estimated** | Judgement, assumption, rule of thumb or intuition |
| **unsupported** | Nothing backs it up; may be wrong or invented |

The first two can be checked against a source. The middle two can't, which is where invented torque figures, capacities and fault codes tend to hide.

It runs in two modes so you can test a real question: **is a model's own assessment of what it made up reliable?**

- `self_label`: the model labels its claims, then questions itself and revises the labels.
- `grounded`: as above, plus checks matched to each label. Evidence-based claims must cite real evidence, and every number must appear in it. A judge model reviews evidence claims against their sources, and a separate, sceptical judge reviews claims made from memory.

`evals/run_evals.py` runs both modes on the same questions and compares them.

## Architecture

```
            ┌──────────────── docker compose ─────────────────┐
 question   │  api (FastAPI)                                  │
 ─────────► │   └─ LangGraph agent ──► gateway (LiteLLM) ──► model provider
            │          │                                      │
            │          └── MCP client ──► mcp (MCP server) ──► documents + unit conversion
            └─────────────────────────────────────────────────┘
                       traces/*.jsonl  ◄── every node, tool call, LLM call, label change
```

The graph (`src/epistemic_agent/agent/graph.py`):

```
guard_input → plan → gather_evidence → draft → [check_citations ↺ draft] → self_critique → [verify] → finalize
     └──────────────────────── blocked ──────────────────────────────────────────────────────────┘
```

Brackets are grounded mode only. How `verify` treats each label:

| Label claimed | Check | If it fails |
|---|---|---|
| verified | cites real evidence; every number appears in it; evidence judge says "entailed" | → derived (if it only follows) or unsupported |
| derived | cites real evidence; evidence judge says it follows | → unsupported |
| inherent_knowledge | knowledge judge says "well_established" | → derived_inherent (plausible) or unsupported (doubtful) |
| derived_inherent | knowledge judge doesn't say "doubtful" | → unsupported |
| estimated, unsupported | none; they claim nothing | |

## How it maps to the role

| Concept | Where |
|---|---|
| Agent framework (LangGraph): state, nodes, conditional edges, retry loops | `agent/graph.py`, `agent/routing.py`, `agent/state.py` |
| MCP server: tools, resources, input validation, error handling, stdio + HTTP | `mcp_server/server.py` |
| MCP client and tool use; tool allowlisting | `agent/tools.py`, `agent/nodes.py::gather_evidence` |
| Model gateway: aliases, routing, fallback, central keys | `gateway/litellm_config.yaml`, `agent/llm.py` |
| Structured output with schema validation and retry | `agent/schemas.py`, `agent/llm.py` |
| Retrieval | `search_documents` / `read_document` tools (keyword; swap for embeddings) |
| Guardrails: PII redaction, injection and path-traversal blocking, output checks | `agent/guardrails.py`, `mcp_server/server.py::_resolve_doc` |
| Evaluation: offline eval set, independent scorer, mode comparison | `evals/` |
| Observability: per-node spans, token usage, label-change events | `agent/tracing.py`, `traces/` |
| Testing: unit, protocol-level, end-to-end with a scripted LLM | `tests/` |
| Containers, CI, gated eval job | `Dockerfile`, `docker-compose.yml`, `.github/workflows/ci.yml` |

## Quick start

Python 3.11+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q                     # offline: no keys, no servers
```

### Run against a real model

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY (or edit the gateway config for another provider)
pip install "litellm[proxy]"
set -a; source .env; set +a
litellm --config gateway/litellm_config.yaml --port 4000 &

python -m epistemic_agent.cli "My FX-28 boiler shows F22. What should I do?"
python -m epistemic_agent.cli "What's the wheel nut torque on a 2019 Ford Transit Custom?"
python -m epistemic_agent.cli "What's the wheel nut torque on a 2019 Ford Transit Custom?" --mode self_label
python evals/run_evals.py
```

The second question is the interesting one: the knowledge base doesn't cover that van, so any figure comes from memory. Compare how the two modes label it.

No gateway? Set `GATEWAY_URL` to any OpenAI-compatible endpoint and `AGENT_MODEL`/`JUDGE_MODEL` to real model names.

### Use your own documents

Put `.md` or `.txt` files in a folder and pass `--docs ~/manuals` to the CLI (or set `DOCS_DIR`). Some of the bundled samples in `mcp_server/sample_docs/` are **fictional** products with invented specs, so the evals have known answers.

### Run as services

```bash
docker compose up --build
curl -s localhost:8000/ask -H 'content-type: application/json' \
  -d '{"question":"My FX-28 boiler shows F22. What do I do?"}' | jq -r .rendered
```

## The MCP server on its own

```bash
python -m epistemic_agent.mcp_server                                  # stdio
python -m epistemic_agent.mcp_server --transport http --port 8001     # Streamable HTTP at /mcp
python -m epistemic_agent.mcp_server --docs-dir ~/manuals             # your own documents
npx @modelcontextprotocol/inspector python -m epistemic_agent.mcp_server   # poke it in a browser UI
```

