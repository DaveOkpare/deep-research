# Building a Multi-Agent Research System

This weekend I decided to build a multi-agent research system based on Anthropic's approach to demonstrate how these architectures work in practice. The system breaks research queries into focused subtasks, runs specialized agents on each, then synthesizes results into structured reports.

The system uses an orchestrator-worker pattern. The lead agent analyzes queries, breaks them into subtasks, and coordinates sub-agents. Sub-agents handle focused research areas and return structured findings.

Key benefits observed:
- **Task separation**: Each agent gets a specific research focus
- **Context isolation**: Sub-agents maintain dedicated context windows  
- **Parallel execution**: Multiple research threads run simultaneously

This mirrors Anthropic's [multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) approach and treats sub-agents as intelligent tool calls rather than autonomous agents.

The core insight of the orchestrator-worker pattern is that the sub-agents work as intelligent tool calls rather than independent agents.

```python
# Traditional approach: Simple tool calls
search_results = web_search("AI agents")
content = fetch_url("https://example.com")

# Our approach: Sub-agents as intelligent tool calls  
research_findings = await sub_agent.run("Research current AI agent architectures")
```

This reframing changes everything. Instead of managing complex inter-agent communication, we have one lead agent that uses other agents as sophisticated, reasoning tools.

The lead agent handles query analysis, task decomposition, and result synthesis. Sub-agents focus on specific research areas and return structured findings with their own context windows.

```python
from pydantic_ai import Agent
from models import ResearchReport, SubagentFindings

# Lead agent configuration
lead_agent = Agent[DateDeps, ResearchReport](
    model='openai:gpt-4o',
    output_type=ResearchReport,
    output_retries=2,
)

# Sub-agent configuration  
sub_agent = Agent[DateDeps, SubagentFindings](
    model='openai:gpt-4o',
    output_type=SubagentFindings,
    output_retries=2,
)
```

The orchestration works through a tool called `run_subagent` where the lead agent provides a list of tasks. These tasks get distributed among subagents and run concurrently, with results gathered and returned to the lead agent as a tool result.

```python
@lead_agent.tool
async def run_subagent(ctx: RunContext[DateDeps], tasks: SubagentTasks):
    """Run subagents concurrently. Each task should be specific and focused."""
    results = await asyncio.gather(
        *[
            sub_agent.run(
                f"Research Task: {task.description}\nFocus Area: {task.focus_area}",
                deps=ctx.deps,
            )
            for task in tasks.tasks
        ]
    )
    return [result.output for result in results]
```

## Execution Strategy

Sub-agents run concurrently, and each sub-agent uses tools designed for concurrent/parallel calls internally.

```python
# Each sub-agent can make multiple searches simultaneously
@sub_agent.tool_plain
async def web_search(query: str, count: int = 10):
    """Search the web for research information."""
    return await search(query, count)

@sub_agent.tool_plain  
async def web_fetch(url: str, timeout: int = 30):
    """Fetch content from a specific URL."""
    return await fetch(url, timeout)
```

## Structured Outputs

Using Pydantic models for all agent outputs ensures consistency across multiple agents. When sub-agents return unstructured text, the lead agent must parse and interpret varying formats, which introduces errors and inconsistency. Structured outputs guarantee that each agent returns data in the expected format, making synthesis reliable. This becomes critical when coordinating multiple agents - the lead agent can confidently access specific fields like `key_insights` or `confidence_level` without parsing natural language responses.

```python
class SubagentFindings(BaseModel):
    task_description: str
    summary: str
    key_insights: List[str]
    sources_found: int
    confidence_level: str  # "high", "medium", "low"

class ResearchReport(BaseModel):
    title: str
    executive_summary: str
    sections: List[ResearchSection]
    key_takeaways: List[str]
    
    def to_markdown(self) -> str:
        # Built-in conversion for streaming
```

This approach provides consistent structure across all agents, type safety to catch errors early, streaming support for real-time updates, and predictable integration with frontend components.

## Streaming Implementation

The streaming system converts partial Pydantic objects to incremental markdown as research progresses:

```python
async def stream_research_response(user_prompt: str, message_id: str):
    previous_markdown = ""
    
    async with lead_agent.run_stream(user_prompt, deps=deps) as result:
        async for partial_output in result.stream():
            current_markdown = stream_to_markdown(partial_output)
            delta = current_markdown[len(previous_markdown):]
            
            if delta:
                yield f"data: {json.dumps({'content': delta})}\n\n"
                previous_markdown = current_markdown
```

## Results

Key findings:
- Intelligent tool calls more reliable than parallel tool calls
- Orchestration works better as coordination rather than autonomous communication  
- Structured outputs become essential with multiple agents

## Use Cases

Multi-agent orchestration fits:
- Open-ended research questions
- Tasks needing multiple information sources
- Complex problems with unpredictable subtasks  

Single agents work better for:
- Well-defined queries
- Linear workflows
- Simple information retrieval

## Implementation

The system is built on Pydantic AI for structured agent outputs, FastAPI for streaming responses, and async processing throughout for efficient coordination. The complete implementation is available at [https://github.com/daveokpare/deep-research](https://github.com/daveokpare/deep-research).

These patterns extend beyond research to any complex, multi-step AI tasks requiring reliable coordination between multiple agents.

## Future Work

This project lacks proper citations and comprehensive evaluation metrics comparing single vs multi-agent performance.

Adding evaluation metrics would make this a more complete demo project by measuring research quality, completion time, and accuracy against baselines.