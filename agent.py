import asyncio
from dataclasses import dataclass
from datetime import datetime
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import KnownModelName

from models import ResearchReport, SubagentFindings, SubagentTasks
from prompts import LEAD_AGENT_PROMPT, SUBAGENT_PROMPT
from tools import fetch, search

MODEL: KnownModelName = "openai:gpt-4o"


@dataclass
class DateDeps:
    current_date: str


lead_agent = Agent[DateDeps, ResearchReport](
    model=MODEL,
    output_type=ResearchReport,
    output_retries=2,
)

sub_agent = Agent[DateDeps, SubagentFindings](
    model=MODEL,
    output_type=SubagentFindings,
    output_retries=2,
    model_settings={"parallel_tool_calls": True},
)


@lead_agent.instructions
def set_instruction(ctx: RunContext):
    return LEAD_AGENT_PROMPT.format(CURRENT_DATE=ctx.deps.current_date)


@sub_agent.instructions
def set_instruction(ctx: RunContext[DateDeps]):
    return SUBAGENT_PROMPT.format(CURRENT_DATE=ctx.deps.current_date)


@lead_agent.tool
async def run_subagent(ctx: RunContext[DateDeps], tasks: SubagentTasks):
    """Run subagents in parallel. Each task should be specific and focused."""
    print(f"🚀 Running {len(tasks.tasks)} subagents...")
    results = await asyncio.gather(
        *[
            sub_agent.run(
                f"Research Task: {task.description}\nFocus Area: {task.focus_area}",
                deps=ctx.deps,
            )
            for task in tasks.tasks
        ]
    )
    print(f"✅ Completed {len(results)} subagent tasks")
    return [result.output for result in results]


@sub_agent.tool_plain
async def web_search(
    query: str, count: int = 10, country: str = "us", search_lang: str = "en"
):
    """
    Search the web using Brave Search API for research information.

    Args:
        query: The search query to execute
        count: Number of search results to return (1-20, default 10)
        country: Country code for localized results (default "us")
        search_lang: Language for search results (default "en")

    Returns:
        YAML-formatted search results with titles, URLs, descriptions, and dates
    """
    print(f"🔍 Searching: {query}")
    try:
        return await search(query, count, country, search_lang)
    except Exception as e:
        return f"Ran into an error: {e}. Please try again!"


@sub_agent.tool_plain
async def web_fetch(url: str, timeout: int = 30, headers: dict | None = None):
    """
    Fetch and extract text content from a specific URL for detailed analysis.

    Args:
        url: The URL to fetch content from
        timeout: Request timeout in seconds (default 30)
        headers: Optional HTTP headers to include in the request

    Returns:
        YAML-formatted response with URL and clean text content extracted from HTML
    """
    print(f"📄 Fetching: {url}")
    try:
        return await fetch(url, timeout, headers)
    except Exception as e:
        return f"Ran into an error: {e}. Please try again!"
