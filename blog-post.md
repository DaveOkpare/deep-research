# Building a Multi-Agent Research System: From Idea to Implementation

Every AI engineer should build a multi-agent system at least once. It's one of those experiences that fundamentally changes how you think about AI applications, revealing both the incredible potential and the subtle complexities of coordinating multiple AI agents.

When Anthropic released their research on multi-agent systems, we knew we had to dig deeper. What exactly makes multi-agent architectures effective? What are their real limitations? And most importantly, what would we learn by building one ourselves?

We decided to replicate something similar to Anthropic's research agent as our learning vehicle. The goal wasn't just to build a useful tool, but to understand multi-agent systems from the ground up through hands-on implementation.

## Understanding Multi-Agent Systems: Benefits and Limitations

Before diving into implementation, we needed to understand what makes multi-agent systems special. The core insight is deceptively simple: some problems are better solved by multiple specialized agents working together than by one generalist agent trying to do everything.

**The key benefits we discovered:**

**Parallel Processing**: Multiple agents can explore different aspects of a problem simultaneously, dramatically reducing total processing time.

**Specialized Expertise**: Each agent can be optimized for specific tasks, leading to better performance than a single general-purpose agent.

**Fault Tolerance**: If one agent fails or gets stuck, others can continue working, making the system more robust.

**Natural Task Decomposition**: Complex problems often break down naturally into subtasks that can be distributed across agents.

**But multi-agent systems also have significant limitations:**

**Coordination Complexity**: Managing communication and task distribution between agents adds substantial complexity.

**Potential for Redundancy**: Without careful orchestration, agents might duplicate work or pursue conflicting approaches.

**Debugging Challenges**: When something goes wrong, it's harder to trace the issue across multiple agents.

**Resource Intensive**: Running multiple agents simultaneously consumes more tokens and API calls.

**Orchestration Overhead**: You need a sophisticated coordinator to manage the agent team effectively.

Anthropic's research helped us see that multi-agent systems excel for "open-ended problems where it's very difficult to predict the required steps in advance." Research tasks fit this description perfectly, making them an ideal testing ground for our learning experiment.

## Designing the Agent Architecture

The solution we settled on follows what Anthropic calls the orchestrator-worker pattern, but with a key insight: the "sub-agents" are essentially sophisticated tool calls orchestrated by the main agent.

**The Lead Agent** is the main thread of execution. It analyzes queries, develops research strategies, breaks down complex questions into focused subtasks, and synthesizes everything into a final report.

**Sub-Agents** are really specialized tool calls with their own context and capabilities. Instead of calling a simple web search function, the lead agent "calls" a sub-agent that can reason about what to search for, evaluate results, and return structured findings.

```python
# Traditional approach: Simple tool calls
search_results = web_search("AI agents")
content = fetch_url("https://example.com")

# Our approach: Sub-agents as intelligent tool calls  
research_findings = await sub_agent.run("Research current AI agent architectures")
```

This reframing clarified our thinking significantly. We're not managing communication between truly independent agents. Instead, we have one main agent that uses other agents as intelligent, context-aware tools.

This design gives us several advantages:
- **Intelligent tool execution**: Each "tool call" can reason about how to accomplish its task
- **Structured outputs**: Tool calls return rich, formatted data instead of raw text
- **Focused context**: Each sub-agent has its own context window dedicated to a specific task
- **Sequential control**: The main agent maintains full control over the research flow

The key insight is that this works well for tasks where you need intelligent decomposition and execution, but don't need truly autonomous agent collaboration.

## Implementation Journey: Starting Simple

Like most good software projects, we started with the simplest possible implementation and gradually added complexity. Our first version was just a single Pydantic AI agent with web search capabilities. Here's what that looked like:

```python
from pydantic_ai import Agent
from models import ResearchReport

simple_agent = Agent[None, ResearchReport](
    model='openai:gpt-4o',
    output_type=ResearchReport,
)
```

This worked for basic queries, but we quickly hit limitations. The agent would make one or two searches, grab some content, and call it done. It wasn't thorough enough for complex research tasks, and there was no way to explore multiple angles simultaneously.

The breakthrough came when we realized we needed structure at two levels: structured outputs to ensure consistent report quality, and structured orchestration to coordinate multiple agents effectively.

For structured outputs, we leaned heavily on Pydantic models:

```python
class ResearchReport(BaseModel):
    title: str
    executive_summary: str
    sections: List[ResearchSection]
    key_takeaways: List[str]
    
    def to_markdown(self) -> str:
        # Built-in markdown conversion for streaming
```

This ensures every research report follows the same format, making them predictable and easy to work with. But the real magic happened when we implemented the orchestration layer.

## Building the Orchestration Logic

The orchestration logic is where our system really comes alive. The lead agent follows a sophisticated research process that mirrors how human researchers actually work:

1. **Query Analysis**: First, it understands the scope and complexity of the research question
2. **Task Decomposition**: It breaks the query into 2-4 focused subtasks that can be researched independently
3. **Sequential Execution**: Sub-agents are deployed one at a time to prevent API overload
4. **Gap Analysis**: After initial findings, the lead agent reviews results for missing information
5. **Iterative Research**: Up to 2 follow-up rounds if gaps are identified
6. **Synthesis**: Finally, it creates a comprehensive report combining all findings

Here's how the lead and sub-agents are configured:

```python
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
```

The key insight here is the `parallel_tool_calls` setting for sub-agents. This allows each sub-agent to make multiple web searches and content fetches simultaneously, dramatically improving research efficiency.

We learned that sequential execution of sub-agents, combined with parallel tool calls within each sub-agent, gives us the best balance of thoroughness and rate limiting. Each sub-agent can explore 3-6 sources in parallel, but we only run one sub-agent at a time to avoid overwhelming external APIs.

## A Key Design Insight: Intelligent Tool Calls vs Parallel Tool Calls

One of the most important lessons we learned was about how to achieve effective parallelism. Initially, we tried to rely on the `parallel_tool_calls` feature that many LLMs support, expecting a single agent to automatically make multiple tool calls simultaneously.

This approach was unreliable in practice. The agent would sometimes make parallel calls, but other times would inexplicably fall back to sequential execution, ignoring the parallel capability entirely. The behavior was inconsistent and difficult to predict.

Our breakthrough came when we reframed the problem: **instead of trying to make dumb tools work in parallel, we create intelligent tool calls that handle their own execution strategy**.

```python
# Instead of: One agent making parallel dumb tool calls (unreliable)
web_search("AI agents")
web_search("multi-agent systems") 
web_search("agent orchestration")

# We use: Intelligent tool calls via sub-agents (reliable)
subtasks = [
    "Research the technical architecture of AI agents",
    "Find recent developments in multi-agent systems", 
    "Analyze different agent orchestration patterns"
]

# Each sub-agent is an intelligent tool call
for subtask in subtasks:
    findings = await sub_agent.run(subtask)  # Sub-agent decides its own search strategy
    research_results.append(findings)
```

The key insight is that **intelligent tool calls (sub-agents) that can reason about their execution are more reliable than trying to parallelize simple tool calls**. Each sub-agent can decide internally whether to make parallel web searches, how to evaluate results, and when it has gathered sufficient information.

This pattern - treating sub-agents as sophisticated, reasoning tool calls rather than autonomous agents - turned out to be one of our most important architectural decisions.

## Streaming the Research Process

One of the most satisfying parts of the system is watching research unfold in real-time. Instead of waiting for a complete report, users see progress as it happens: new sections appearing, findings being added, and the report gradually taking shape.

This required building a streaming API that could handle partial Pydantic objects. Our FastAPI implementation looks like this:

```python
@app.post("/api/chat")
async def chat(request: Request):
    if web_search:
        streaming_generator = stream_research_response(user_prompt, message_id)
    else:
        streaming_generator = stream_response(message_history, user_prompt, message_id)
    
    return StreamingResponse(
        streaming_generator, 
        media_type="application/json"
    )
```

The real complexity is in converting partial research reports to incremental markdown:

```python
async def stream_research_response(user_prompt: str, message_id: str):
    previous_markdown = ""
    
    async for partial_result in agent_runner:
        current_markdown = stream_to_markdown(partial_result)
        delta = current_markdown[len(previous_markdown):]
        
        if delta:
            yield create_streaming_chunk(delta, message_id)
            previous_markdown = current_markdown
```

This approach lets users see research progress in real-time as the system finds new information, adds sections to reports, and updates existing content. The frontend uses Vercel's AI Elements with the `streamdown` library to render markdown as it arrives.

## The Magic of Structured Outputs

One decision that paid dividends throughout development was using Pydantic models for all agent outputs. This might seem like overkill for a research tool, but it provides several crucial benefits:

**Consistency**: Every research report follows exactly the same structure, making them easy to parse, display, and work with programmatically.

**Type Safety**: We catch data structure errors at runtime instead of discovering them in production.

**Stream-Friendly**: Partial objects can be serialized progressively, enabling our real-time streaming functionality.

**Easy Integration**: The frontend can reliably expect certain fields and structure, simplifying UI development.

Here's our complete research report model:

```python
class ResearchReport(BaseModel):
    title: str
    executive_summary: str
    sections: List[ResearchSection]
    key_takeaways: List[str]
    sources: List[str]
    
    def to_markdown(self) -> str:
        lines = [f"# {self.title}", "", "## Executive Summary", self.executive_summary, ""]
        
        for section in self.sections:
            lines.extend([f"## {section.title}", section.content, ""])
            
        lines.extend(["## Key Takeaways"] + [f"- {takeaway}" for takeaway in self.key_takeaways])
        
        return "\n".join(lines)
```

The `to_markdown()` method is particularly useful for streaming. We can convert partial reports to markdown incrementally, sending only the new content to users as research progresses.

## What We Learned About Multi-Agent Systems

Building this system taught us several fundamental lessons about multi-agent architectures that every AI engineer should know:

**Intelligent tool calls beat parallel tool calls**: As we discovered, treating sub-agents as reasoning tool calls that can handle their own execution strategy is far more reliable than trying to parallelize simple tool calls.

**Sequential coordination, parallel execution**: Running sub-agents one at a time while allowing each to use parallel tools internally gives the best balance of thoroughness and reliability. Trying to run everything in parallel often backfires due to rate limits and coordination complexity.

**Structured outputs become essential at scale**: With multiple agents producing results, the overhead of defining Pydantic models pays off quickly. Without structured outputs, coordinating agent results becomes a nightmare.

**Prompt engineering becomes critical**: In single-agent systems, you can often get away with rough prompts. But in multi-agent systems, the lead agent's instructions for task decomposition and synthesis directly determine system quality. Small prompt changes can have dramatic effects.

**Error handling complexity multiplies**: With multiple agents, you need sophisticated strategies for handling partial failures, retries, and graceful degradation. What works for single agents often breaks down in multi-agent scenarios.

**The coordination overhead is real**: Multi-agent systems require significantly more tokens and API calls than equivalent single-agent approaches. The benefits need to justify this cost.

**But the results can be genuinely better**: For complex, open-ended tasks like research, the combination of parallel exploration and intelligent synthesis produces results that single agents struggle to match.

The most valuable insight for other AI engineers is this: **start simple and add agents only when you hit clear limitations**. Multi-agent systems solve real problems, but they introduce genuine complexity that isn't always worth it.

## When to Choose Orchestration vs Single Agents

Through building and using this system, we've developed intuition about when multi-agent orchestration makes sense:

**Use orchestration for**:
- Open-ended research questions
- Tasks requiring multiple perspectives or sources
- Complex problems where you can't predict subtasks in advance
- Situations where parallel exploration adds value

**Stick with single agents for**:
- Well-defined, straightforward questions
- Tasks with predictable, linear workflows  
- Simple information retrieval
- Cases where added complexity isn't justified

The orchestrator-worker pattern shines when you need adaptive, thorough exploration of complex topics. But for many use cases, a well-designed single agent is simpler and more appropriate.

## Taking It Further

This system opens up interesting possibilities for future development. We're considering:

**Specialized sub-agents**: Different agents optimized for academic papers, news articles, technical documentation, or social media analysis.

**Persistent research**: Saving research sessions and allowing iterative refinement over time.

**Collaborative features**: Multiple users contributing to and refining research reports.

**Integration capabilities**: APIs for embedding research functionality in other applications.

The foundation we've built with Pydantic AI's structured outputs and FastAPI's streaming capabilities makes these extensions straightforward to implement.

## Why Every AI Engineer Should Try This

Building this multi-agent research system turned out to be one of the most educational AI projects we've undertaken. It's less about advanced techniques and more about understanding how AI agents can work together effectively.

The experience taught us fundamental lessons about:
- When multi-agent architectures provide real value vs unnecessary complexity
- How to design reliable coordination between AI agents  
- The practical challenges of orchestrating multiple LLMs
- The importance of structured outputs and careful prompt engineering at scale

These insights have changed how we approach AI system design more broadly, even for single-agent applications.

**For other AI engineers considering similar projects**: Start with replicating something like Anthropic's research agent. The domain is complex enough to surface real multi-agent challenges, but well-defined enough to make progress quickly. You'll learn more about AI systems in a few weeks than months of single-agent work.

The key patterns we discovered, especially treating sub-agents as intelligent tool calls rather than autonomous agents, apply far beyond research agents. They're fundamental principles for any system where you need AI agents to handle complex, multi-step tasks.

**The broader lesson**: Multi-agent systems aren't just about building more complex AI applications. They're about understanding how to decompose complex problems in ways that AI can solve reliably. That's a skill that will become increasingly valuable as AI capabilities continue to grow.

Our implementation is available in the repository, and we'd love to see what variations and improvements others build. Multi-agent systems are still evolving rapidly, and there's huge value in more AI engineers getting hands-on experience with these patterns.

What complex problem would you tackle with a team of coordinated AI agents?