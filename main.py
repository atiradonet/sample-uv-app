"""LLM Web Search Agent using Anthropic tool use with the ReAct pattern."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx
import typer
from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolResultBlockParam, ToolUseBlock
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

load_dotenv()

app = typer.Typer(help="LLM Web Search Agent powered by Anthropic + DuckDuckGo")
console = Console()


# ---------------------------------------------------------------------------
# Pydantic schemas for tool inputs/outputs
# ---------------------------------------------------------------------------


class SearchInput(BaseModel):
    query: str = Field(..., description="The search query to look up on the web")
    max_results: int = Field(5, description="Maximum number of results to return", ge=1, le=10)


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class SearchOutput(BaseModel):
    results: list[SearchResult]
    query: str


# ---------------------------------------------------------------------------
# DuckDuckGo search (free, no API key required)
# ---------------------------------------------------------------------------

DUCKDUCKGO_URL = "https://api.duckduckgo.com/"


async def web_search(query: str, max_results: int = 5) -> SearchOutput:
    """Fetch results from DuckDuckGo's instant answer API."""
    params = {
        "q": query,
        "format": "json",
        "no_html": "1",
        "skip_disambig": "1",
        "no_redirect": "1",
    }
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(DUCKDUCKGO_URL, params=params)
        response.raise_for_status()
        data = response.json()

    results: list[SearchResult] = []

    # Abstract (top answer)
    if data.get("AbstractText"):
        results.append(
            SearchResult(
                title=data.get("Heading", "DuckDuckGo Abstract"),
                url=data.get("AbstractURL", ""),
                snippet=data["AbstractText"],
            )
        )

    # Related topics
    for topic in data.get("RelatedTopics", []):
        if len(results) >= max_results:
            break
        if "Text" in topic and "FirstURL" in topic:
            results.append(
                SearchResult(
                    title=topic.get("Text", "")[:80],
                    url=topic["FirstURL"],
                    snippet=topic["Text"],
                )
            )
        # Nested topic groups
        for sub in topic.get("Topics", []):
            if len(results) >= max_results:
                break
            if "Text" in sub and "FirstURL" in sub:
                results.append(
                    SearchResult(
                        title=sub.get("Text", "")[:80],
                        url=sub["FirstURL"],
                        snippet=sub["Text"],
                    )
                )

    return SearchOutput(results=results, query=query)


# ---------------------------------------------------------------------------
# Tool definition for Anthropic
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "name": "web_search",
        "description": (
            "Search the web using DuckDuckGo to find current information. "
            "Use this when you need facts, recent events, or information you are unsure about."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (1-10)",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    }
]


# ---------------------------------------------------------------------------
# ReAct agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful research assistant with access to a web search tool.

When answering questions:
1. REASON: Think about what you need to know to answer the question.
2. ACT: If you need current or factual information, call the web_search tool.
3. OBSERVE: Read the search results carefully.
4. REASON again: Integrate what you learned and decide if more searches are needed.
5. ANSWER: Provide a clear, grounded answer based on your research.

Always cite the sources you used in your final answer.
"""


async def run_agent(question: str, model: str, max_iterations: int) -> str:
    """Run the ReAct agent loop and return the final answer."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise typer.BadParameter(
            "ANTHROPIC_API_KEY is not set. Add it to your .env file or environment."
        )

    client = AsyncAnthropic(api_key=api_key)
    messages: list[MessageParam] = [{"role": "user", "content": question}]

    for iteration in range(1, max_iterations + 1):
        console.print(Rule(f"[dim]Iteration {iteration}[/dim]", style="dim"))

        response = await client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        tool_uses: list[ToolUseBlock] = []
        reasoning_parts: list[str] = []

        for block in response.content:
            if block.type == "text":
                reasoning_parts.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(block)

        if reasoning_parts:
            console.print(
                Panel(
                    Markdown("\n".join(reasoning_parts)),
                    title="[bold cyan]Reasoning[/bold cyan]",
                    border_style="cyan",
                    padding=(0, 1),
                )
            )

        if response.stop_reason == "end_turn" and not tool_uses:
            return "\n".join(reasoning_parts)

        if tool_uses:
            messages.append({"role": "assistant", "content": response.content})
            tool_results: list[ToolResultBlockParam] = []

            for tool_use in tool_uses:
                console.print(
                    Panel(
                        Text(f"query: {tool_use.input.get('query', '')}", style="yellow"),
                        title=f"[bold yellow]Tool Call: {tool_use.name}[/bold yellow]",
                        border_style="yellow",
                        padding=(0, 1),
                    )
                )

                try:
                    search_input = SearchInput(**tool_use.input)
                    output = await web_search(search_input.query, search_input.max_results)
                    result_text = json.dumps(output.model_dump(), indent=2)
                    is_error = False
                except Exception as exc:
                    result_text = f"Search failed: {exc}"
                    is_error = True

                if not is_error:
                    results_display = "\n".join(
                        f"• **{r['title']}**\n  {r['snippet'][:120]}…\n  {r['url']}"
                        for r in json.loads(result_text)["results"]
                    )
                    console.print(
                        Panel(
                            Markdown(results_display) if results_display else Text("No results found."),
                            title="[bold green]Search Results[/bold green]",
                            border_style="green",
                            padding=(0, 1),
                        )
                    )
                else:
                    console.print(
                        Panel(
                            Text(result_text, style="red"),
                            title="[bold red]Search Error[/bold red]",
                            border_style="red",
                            padding=(0, 1),
                        )
                    )

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": result_text,
                        "is_error": is_error,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "end_turn":
            return "\n".join(reasoning_parts)

    return "Maximum iterations reached without a conclusive answer."


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question to research and answer"),
    model: str = typer.Option("claude-opus-4-6", "--model", "-m", help="Anthropic model to use"),
    max_iterations: int = typer.Option(10, "--max-iter", help="Maximum ReAct iterations"),
):
    """Ask the LLM web search agent a question."""
    console.print()
    console.print(
        Panel(
            Text(question, style="bold white"),
            title="[bold blue]Question[/bold blue]",
            border_style="blue",
            padding=(0, 1),
        )
    )
    console.print()

    try:
        final_answer = asyncio.run(run_agent(question, model, max_iterations))
    except typer.BadParameter as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise typer.Exit(1)
    except Exception as exc:
        console.print(f"[bold red]Unexpected error:[/bold red] {exc}")
        raise typer.Exit(1)

    console.print()
    console.print(Rule("[bold blue]Final Answer[/bold blue]", style="blue"))
    console.print()
    console.print(Markdown(final_answer))
    console.print()


if __name__ == "__main__":
    app()
