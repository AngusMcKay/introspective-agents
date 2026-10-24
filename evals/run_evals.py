"""Run the eval suite against a real model (via the gateway) in one or both modes.

    python evals/run_evals.py                          # both modes, all cases
    python evals/run_evals.py --mode grounded --only fact_
    python evals/run_evals.py --repeats 3              # LLMs are stochastic; repeat to see variance

Writes evals/results/<timestamp>.json and prints a comparison table.
The MCP server runs in-process on the sample docs, so only the gateway needs to be up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scoring import score_case  # noqa: E402

from introspective_agent.agent.graph import run_graph  # noqa: E402
from introspective_agent.agent.llm import GatewayLLM  # noqa: E402
from introspective_agent.agent.tools import MCPTools  # noqa: E402
from introspective_agent.agent.tracing import Tracer  # noqa: E402
from introspective_agent.config import SAMPLE_DOCS  # noqa: E402
from introspective_agent.mcp_server.server import build_server  # noqa: E402

HERE = Path(__file__).resolve().parent


def load_cases(only: str | None) -> list[dict]:
    cases = [json.loads(line) for line in (HERE / "cases.jsonl").read_text().splitlines() if line.strip()]
    return [c for c in cases if not only or c["id"].startswith(only)]


async def run(modes: list[str], cases: list[dict], repeats: int) -> list[dict]:
    llm = GatewayLLM()
    tools = MCPTools(build_server(SAMPLE_DOCS))  # expected answers come from the sample docs
    results = []
    for mode in modes:
        for rep in range(repeats):
            for case in cases:
                tracer = Tracer()
                t0 = time.perf_counter()
                try:
                    state = await run_graph(case["question"], llm=llm, tools=tools, mode=mode, tracer=tracer)
                    r = score_case(case, state["final"], state.get("evidence", []))
                except Exception as e:
                    r = {"id": case["id"], "type": case["type"], "mode": mode, "passed": False, "error": repr(e)}
                r.update(repeat=rep, latency_s=round(time.perf_counter() - t0, 2), trace_id=tracer.trace_id,
                         **{k: v for k, v in tracer.summary().items() if k != "trace_id"})
                results.append(r)
                print(f"{mode:10s} {case['id']:28s} {'PASS' if r['passed'] else 'FAIL'}"
                      f"{'  ' + r['error'] if 'error' in r else ''}")
    return results


def summarise(results: list[dict]) -> str:
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_mode[r["mode"]].append(r)
    rows = ["| metric | " + " | ".join(by_mode) + " |", "|---|" + "---|" * len(by_mode)]

    def metric(name: str, fn) -> None:
        rows.append(f"| {name} | " + " | ".join(fn(rs) for rs in by_mode.values()) + " |")

    def rate(rs: list[dict], t: str) -> str:
        xs = [r for r in rs if r["type"] == t]
        return f"{sum(r['passed'] for r in xs)}/{len(xs)}" if xs else "-"

    metric("factual accuracy (from documents)", lambda rs: rate(rs, "factual"))
    metric("general-knowledge cases", lambda rs: rate(rs, "knowledge"))
    metric("trap cases handled", lambda rs: rate(rs, "trap"))
    metric("guardrail cases", lambda rs: rate(rs, "guardrail"))

    def leak(rs: list[dict]) -> str:
        n = sum(r.get("numeric_verified", 0) for r in rs)
        k = sum(r.get("leaked_verified", 0) for r in rs)
        return f"{k}/{n} ({k / n:.0%})" if n else "0/0"

    metric("verified leak rate", leak)
    metric("figures asserted as inherent_knowledge", lambda rs: str(sum(r.get("figures_from_memory", 0) for r in rs)))
    metric("label changes", lambda rs: str(sum(r.get("label_changes", 0) for r in rs)))
    metric("mean latency (s)", lambda rs: f"{sum(r['latency_s'] for r in rs) / len(rs):.1f}")
    metric("mean LLM calls", lambda rs: f"{sum(r.get('llm_calls', 0) for r in rs) / len(rs):.1f}")
    metric("total tokens", lambda rs: str(sum(r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in rs)))
    metric("errors", lambda rs: str(sum("error" in r for r in rs)))
    return "\n".join(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["grounded", "self_label", "both"], default="both")
    p.add_argument("--only", help="run only cases whose id starts with this")
    p.add_argument("--repeats", type=int, default=1)
    args = p.parse_args()

    modes = ["self_label", "grounded"] if args.mode == "both" else [args.mode]
    results = asyncio.run(run(modes, load_cases(args.only), args.repeats))
    table = summarise(results)
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.write_text(json.dumps({"results": results, "summary": table}, indent=2, default=str))
    print("\n" + table + f"\n\nDetail: {out}")


if __name__ == "__main__":
    main()
