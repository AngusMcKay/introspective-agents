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

The graph (`src/introspective_agent/agent/graph.py`):

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

python -m introspective_agent.cli "My FX-28 boiler shows F22. What should I do?"
python -m introspective_agent.cli "What's the wheel nut torque on a 2019 Ford Transit Custom?"
python -m introspective_agent.cli "What's the wheel nut torque on a 2019 Ford Transit Custom?" --mode self_label
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

This is the part to reuse for a work prototype. It knows nothing about the agent.

```bash
python -m introspective_agent.mcp_server                                  # stdio
python -m introspective_agent.mcp_server --transport http --port 8001     # Streamable HTTP at /mcp
python -m introspective_agent.mcp_server --docs-dir ~/manuals             # your own documents
npx @modelcontextprotocol/inspector python -m introspective_agent.mcp_server   # poke it in a browser UI
```

To use it from Claude Desktop, add to its MCP config:

```json
{ "mcpServers": { "knowledge-base": {
    "command": "/path/to/.venv/bin/python",
    "args": ["-m", "introspective_agent.mcp_server", "--docs-dir", "/path/to/manuals"] } } }
```

It's written for the **v2** Python SDK (`MCPServer`). Most tutorials online use v1 (`FastMCP`); the ideas are the same, but the import names differ.

Design decisions worth being able to explain:
- **IDs, not paths.** `read_document` takes an ID from the index and never builds a file path from input, so neither the model nor an injected instruction can make it read arbitrary files.
- **Unit conversion is a tool.** Arithmetic in a model's head is a classic source of confident errors, and a tool result can be cited and checked.
- **Errors.** `ToolInputError` messages go back to the client so the agent can correct itself. Any other exception reaches the client only as a generic error, so internals don't leak.
- **Retrieved text is untrusted.** A document could contain instructions. Prompts wrap evidence as data and tell the model to ignore instructions inside it.
- **DNS-rebinding protection** on the HTTP transport (`--allowed-host`).

## Suggested order to work through it

1. **Run `pytest` and read `tests/test_graph.py`.** It shows the whole control flow, including the agent catching an invented figure, downgrading a spec recalled from memory, and refusing a "verified" claim with no source.
2. **MCP server.** Add a tool, e.g. a PDF reader. Call it from the Inspector before involving the agent.
3. **Graph.** Read `graph.py` then `nodes.py`. Add a node, e.g. a safety classifier for gas, mains electrics or structural questions.
4. **Gateway.** Point `judge-model` at a different provider from `agent-model`, so the judge doesn't share the drafter's blind spots. Watch the fallback work by giving `agent-model` a bad key.
5. **Evals.** Run both modes. Add questions you've had wrong answers to before. Run `--repeats 3`: results vary between runs, which is itself a finding.
6. **Strictness.** See below.
7. **Observability.** Read a trace file, then swap `tracing.py` for OpenTelemetry + Langfuse.
8. **Retrieval.** Replace keyword search with embeddings without changing the tool's interface.

## Adding a strictness setting

You mentioned wanting to control how harshly the agent questions itself for questions where being right matters. The hooks are in place but it's not implemented. Places it would touch:

- `agent/prompts.py`: `CRITIQUE` and `KNOWLEDGE_JUDGE` wording, e.g. "treat any specific figure recalled from memory as doubtful".
- `agent/nodes.py::verify`: the rules, e.g. at high strictness, any `inherent_knowledge` claim containing a number drops to `derived_inherent` without asking the judge, and "plausible" maps to `unsupported`.
- `agent/state.py` and `api.py`/`cli.py`: carry a `strictness` value through, like `mode`.
- `evals/`: run the suite at each level. Expect a trade-off: stricter means fewer invented specs, but more correct general knowledge labelled weakly. Measuring that trade-off is the interesting result.

## The self-assessment experiment

Models often do correctly retract invented content when challenged. The question is whether that's reliable enough to *label* claims without checking them. The eval reports:

- **verified leak rate:** of numeric "verified" claims, how many have a number appearing in none of the retrieved evidence.
- **figures asserted as inherent_knowledge:** specific numbers presented as strong facts from memory.
- **trap cases:** questions about products the knowledge base doesn't cover, or fault codes the manual doesn't list. They fail if a figure or meaning is presented with a strong label.

Caveats, so the numbers aren't over-read:
- In grounded mode the verifier uses a similar numeric check to the scorer, so its low leak rate is partly by construction.
- The knowledge judge is still a model judging from memory. Using a different model from the drafter helps, but it isn't ground truth.
- The trap and knowledge scoring is rule-based and crude. A hand-labelled set (claim → correct label) with a confusion matrix per mode is the most valuable extension here.

## Known limitations

- No authentication on the API.
- One MCP session per tool call (simple, not efficient).
- Guardrails are regex-based; production would layer a classifier on top.
- Tool calls run sequentially; LangGraph can fan them out in parallel.
- The document index is cached; restart the MCP server after adding documents.
