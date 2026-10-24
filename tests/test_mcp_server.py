"""The MCP server, tested both as plain functions and over a real MCP session (in-process)."""

import anyio
import pytest
from mcp import Client

from introspective_agent.mcp_server.server import (
    ToolInputError, build_server, convert_units, list_documents, read_document, search_documents,
)


def test_list_documents_excludes_readme():
    ids = {d["doc_id"] for d in list_documents()["documents"]}
    assert ids == {"fernhill_fx28_boiler", "corrie_cvan_workshop_notes", "glenlyon_gd18_drill"}


def test_search_finds_the_right_paragraph():
    top = search_documents("wheel nut torque")["results"][0]
    assert top["doc_id"] == "corrie_cvan_workshop_notes" and "160 Nm" in top["text"]


def test_search_rejects_empty_query():
    with pytest.raises(ToolInputError):
        search_documents("the and of")


def test_read_document():
    r = read_document("fernhill_fx28_boiler")
    assert "F22" in r["text"] and r["source"] == "knowledge_base:fernhill_fx28_boiler"


@pytest.mark.parametrize("doc_id", ["../../etc/passwd", "/etc/passwd", "README", "nope"])
def test_read_document_only_reads_indexed_docs(doc_id):
    with pytest.raises(ToolInputError):
        read_document(doc_id)


@pytest.mark.parametrize("value, f, t, expected", [
    (160, "nm", "lbf_ft", 118.01),
    (40, "Nm", "lbf_ft", 29.5),
    (4.1, "bar", "psi", 59.47),
    (20, "c", "f", 68.0),
    (6.2, "l", "us_qt", 6.55),
])
def test_convert_units(value, f, t, expected):
    assert convert_units(value, f, t)["result"] == pytest.approx(expected, abs=0.01)


def test_convert_units_rejects_mismatched_dimensions():
    with pytest.raises(ToolInputError):
        convert_units(1, "nm", "bar")


def test_over_mcp_protocol():
    async def go():
        async with Client(build_server()) as c:
            names = {t.name for t in (await c.list_tools()).tools}
            assert names == {"list_documents", "search_documents", "read_document", "convert_units"}
            ok = await c.call_tool("convert_units", {"value": 160, "from_unit": "nm", "to_unit": "lbf_ft"})
            assert not ok.is_error and ok.structured_content["result"] == 118.01
            bad = await c.call_tool("read_document", {"doc_id": "../../etc/passwd"})
            assert bad.is_error and "invalid document id" in bad.content[0].text  # deliberate ToolError reaches client

    anyio.run(go)
