"""HTTP API around the agent.

    uvicorn introspective_agent.api:app --port 8000

There's no authentication: fine on your own machine, not fine anywhere shared. If you
expose it, put it behind auth and rate limiting (both commonly done at an API gateway).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from introspective_agent.agent.graph import answer
from introspective_agent.agent.llm import GatewayLLM
from introspective_agent.agent.render import render
from introspective_agent.agent.schemas import FinalAnswer, Mode
from introspective_agent.agent.tools import MCPTools
from introspective_agent.agent.tracing import Tracer
from introspective_agent.config import settings

app = FastAPI(title="Introspective agent", version="0.2.0")
_llm = None
_tools = MCPTools(settings.mcp_url)


def _get_llm() -> GatewayLLM:
    global _llm
    if _llm is None:
        _llm = GatewayLLM()
    return _llm


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=settings.max_question_chars)
    mode: Mode = "grounded"


class AskResponse(BaseModel):
    answer: FinalAnswer
    rendered: str
    trace: dict


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    tracer = Tracer()
    try:
        final = await answer(req.question, llm=_get_llm(), tools=_tools, mode=req.mode, tracer=tracer)
    except Exception as e:  # don't leak internals; the trace file has the detail
        tracer.event("run_error", error=repr(e))
        raise HTTPException(status_code=502, detail=f"agent failed; trace_id={tracer.trace_id}") from e
    return AskResponse(answer=final, rendered=render(final), trace=tracer.summary())
