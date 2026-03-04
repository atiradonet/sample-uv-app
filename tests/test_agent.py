"""Tests for the LLM web search agent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

from main import (
    DUCKDUCKGO_URL,
    SearchInput,
    SearchOutput,
    SearchResult,
    app,
    run_agent,
    web_search,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DDGO_WITH_ABSTRACT = {
    "Heading": "Python (programming language)",
    "AbstractText": "Python is a high-level, general-purpose programming language.",
    "AbstractURL": "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "RelatedTopics": [
        {"Text": "Python Software Foundation", "FirstURL": "https://www.python.org/psf/"},
        {"Text": "CPython — reference implementation", "FirstURL": "https://github.com/python/cpython"},
    ],
}

DDGO_EMPTY = {"Heading": "", "AbstractText": "", "AbstractURL": "", "RelatedTopics": []}


# ---------------------------------------------------------------------------
# Pydantic model tests (sync — no async needed)
# ---------------------------------------------------------------------------


def test_search_input_defaults():
    inp = SearchInput(query="uv package manager")
    assert inp.query == "uv package manager"
    assert inp.max_results == 5


def test_search_input_custom_max_results():
    inp = SearchInput(query="test", max_results=3)
    assert inp.max_results == 3


def test_search_input_rejects_out_of_bounds():
    with pytest.raises(Exception):
        SearchInput(query="test", max_results=0)
    with pytest.raises(Exception):
        SearchInput(query="test", max_results=11)


def test_search_result_model():
    r = SearchResult(title="UV Docs", url="https://docs.astral.sh/uv/", snippet="Fast Python package manager")
    assert r.title == "UV Docs"
    assert r.url == "https://docs.astral.sh/uv/"


def test_search_output_model():
    out = SearchOutput(
        results=[SearchResult(title="T", url="http://example.com", snippet="S")],
        query="test query",
    )
    assert out.query == "test query"
    assert len(out.results) == 1


# ---------------------------------------------------------------------------
# web_search tests (async + mocked HTTP via respx)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_web_search_with_abstract():
    respx.get(DUCKDUCKGO_URL).mock(return_value=httpx.Response(200, json=DDGO_WITH_ABSTRACT))

    result = await web_search("python programming language")

    assert isinstance(result, SearchOutput)
    assert result.query == "python programming language"
    assert len(result.results) >= 1
    assert "Python" in result.results[0].title
    assert "high-level" in result.results[0].snippet


@pytest.mark.asyncio
@respx.mock
async def test_web_search_empty_response():
    respx.get(DUCKDUCKGO_URL).mock(return_value=httpx.Response(200, json=DDGO_EMPTY))

    result = await web_search("xyzzy totally unknown")

    assert isinstance(result, SearchOutput)
    assert result.results == []


@pytest.mark.asyncio
@respx.mock
async def test_web_search_respects_max_results():
    many_topics = [{"Text": f"Topic {i}", "FirstURL": f"https://example.com/{i}"} for i in range(10)]
    respx.get(DUCKDUCKGO_URL).mock(
        return_value=httpx.Response(
            200,
            json={"Heading": "", "AbstractText": "", "AbstractURL": "", "RelatedTopics": many_topics},
        )
    )

    result = await web_search("broad query", max_results=3)

    assert len(result.results) <= 3


@pytest.mark.asyncio
@respx.mock
async def test_web_search_nested_topics():
    response_data = {
        "Heading": "",
        "AbstractText": "",
        "AbstractURL": "",
        "RelatedTopics": [
            {
                "Topics": [
                    {"Text": "Nested topic A", "FirstURL": "https://example.com/a"},
                    {"Text": "Nested topic B", "FirstURL": "https://example.com/b"},
                ]
            }
        ],
    }
    respx.get(DUCKDUCKGO_URL).mock(return_value=httpx.Response(200, json=response_data))

    result = await web_search("nested topics test")

    assert any("Nested topic" in r.title for r in result.results)


@pytest.mark.asyncio
@respx.mock
async def test_web_search_http_error():
    respx.get(DUCKDUCKGO_URL).mock(return_value=httpx.Response(503))

    with pytest.raises(httpx.HTTPStatusError):
        await web_search("any query")


# ---------------------------------------------------------------------------
# run_agent tests (async + mocked AsyncAnthropic)
# ---------------------------------------------------------------------------


def _text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_use_block(tool_id: str, name: str, input_data: dict) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = tool_id
    block.name = name
    block.input = input_data
    return block


def _response(content: list, stop_reason: str = "end_turn") -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.stop_reason = stop_reason
    return resp


@pytest.mark.asyncio
async def test_run_agent_single_turn():
    """Agent returns final answer without calling any tools."""
    mock_response = _response([_text_block("Paris is the capital of France.")])

    with patch("main.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            result = await run_agent("What is the capital of France?", "claude-opus-4-6", 10)

    assert "Paris" in result
    mock_client.messages.create.assert_awaited_once()


@pytest.mark.asyncio
@respx.mock
async def test_run_agent_react_cycle():
    """Agent issues a tool call then produces a final answer."""
    respx.get(DUCKDUCKGO_URL).mock(return_value=httpx.Response(200, json=DDGO_WITH_ABSTRACT))

    first = _response(
        [_tool_use_block("toolu_01", "web_search", {"query": "latest Python version", "max_results": 5})],
        stop_reason="tool_use",
    )
    second = _response([_text_block("The latest Python version is 3.13.")])

    with patch("main.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(side_effect=[first, second])

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            result = await run_agent("What is the latest Python version?", "claude-opus-4-6", 10)

    assert "3.13" in result
    assert mock_client.messages.create.await_count == 2


@pytest.mark.asyncio
async def test_run_agent_missing_api_key():
    with patch.dict("os.environ", {}, clear=True):
        import os
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(Exception, match="ANTHROPIC_API_KEY"):
                await run_agent("anything", "claude-opus-4-6", 10)


@pytest.mark.asyncio
async def test_run_agent_max_iterations():
    """Agent stops after max_iterations even if no end_turn received."""
    # Every response is a tool use with stop_reason=tool_use, so it never ends naturally
    tool_response = _response(
        [_tool_use_block("toolu_01", "web_search", {"query": "loop", "max_results": 5})],
        stop_reason="tool_use",
    )

    with patch("main.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=tool_response)

        with patch("main.web_search", new_callable=AsyncMock) as mock_search:
            mock_search.return_value = SearchOutput(results=[], query="loop")

            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
                result = await run_agent("loop forever", "claude-opus-4-6", max_iterations=2)

    assert "Maximum iterations" in result
    assert mock_client.messages.create.await_count == 2


@pytest.mark.asyncio
async def test_run_agent_passes_model():
    mock_response = _response([_text_block("42")])

    with patch("main.AsyncAnthropic") as mock_cls:
        mock_client = MagicMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
            await run_agent("test", "claude-haiku-4-5-20251001", 10)

    call_kwargs = mock_client.messages.create.call_args.kwargs
    assert call_kwargs["model"] == "claude-haiku-4-5-20251001"
