# Building a Multi-Agent Research System: From Idea to Implementation

What if you could have a team of AI researchers working on your questions in parallel? Each one diving deep into different aspects of your query, coordinating their findings, and synthesizing everything into a comprehensive report. That's exactly what we set out to build, and the journey taught us more about agent orchestration than we ever expected.

When we started this project, we were frustrated with the limitations of traditional AI interactions. Ask a question, get an answer. Simple, but not very useful for complex research tasks that require exploring multiple angles, cross-referencing sources, and synthesizing disparate information. Real research isn't linear. It's messy, iterative, and often leads down unexpected but valuable paths.

## The Problem with Linear Research

Traditional AI systems, despite their impressive capabilities, have a fundamental limitation when it comes to research: they operate in a single-threaded manner. You ask a question, the model thinks about it, and gives you one response. But anyone who has done serious research knows that's not how discovery works.

Real research involves starting with broad questions, finding initial sources, discovering new angles you hadn't considered, diving deeper into promising directions, and occasionally hitting dead ends that require pivoting to entirely new approaches. It's inherently parallel and adaptive.

This challenge isn't unique to us. The team at Anthropic articulated it perfectly in their work on multi-agent research systems: "Research work involves open-ended problems where it's very difficult to predict the required steps in advance." Their insight was that multi-agent systems provide the flexibility to "pivot or explore tangential connections" dynamically.

That became our north star. We wanted to build a system that could break down complex research questions into focused subtasks, explore them in parallel, and synthesize findings into coherent reports. But how do you coordinate multiple AI agents effectively?

## Designing the Agent Architecture

The solution we settled on follows what Anthropic calls the orchestrator-worker pattern. Instead of trying to build one super-intelligent agent that does everything, we created a hierarchical system with specialized roles:

**The Lead Agent** acts as a research coordinator. It analyzes incoming queries, develops research strategies, breaks down complex questions into focused subtasks, and ultimately synthesizes everything into a final report.

**Sub-Agents** are the specialized research workers. Each one receives a specific research task from the lead agent and uses web search and content extraction tools to gather information. They can make multiple tool calls in parallel, efficiently exploring different sources simultaneously.

This design gives us several advantages over a single-agent approach:
- **Better fault tolerance**: If one sub-agent encounters an error, the others continue working
- **Parallel exploration**: Multiple research angles can be pursued simultaneously  
- **Focused expertise**: Each agent can be optimized for its specific role
- **Natural scalability**: Adding more sub-agents is straightforward

The key insight from Anthropic's work is that this pattern works best for "complex tasks where you can't predict the subtasks needed." That perfectly describes open-ended research.

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

## Lessons Learned & Reflections

Building this system taught us several things we didn't expect:

**Sequential sub-agent execution works better than parallel**: Our initial instinct was to run all sub-agents simultaneously for maximum speed. But this overwhelmed external APIs and didn't significantly improve research quality. Sequential execution with parallel tool calls within each sub-agent proved more reliable and just as effective.

**Rate limiting is crucial**: Web search APIs have strict limits, and respecting them is essential for reliability. We built in automatic delays and retry logic with exponential backoff.

**Structured outputs are worth the complexity**: The initial overhead of defining Pydantic models pays off quickly in terms of reliability, debugging ease, and feature development speed.

**Real-time feedback transforms the experience**: Users engage differently with research when they can see it happening. The streaming functionality turns what could be a frustrating wait into an engaging process.

**Agent orchestration requires careful prompt engineering**: The lead agent's instructions for task decomposition and synthesis are critical. Small changes in prompts can dramatically affect research quality.

The system handles a surprising variety of research tasks well, from technical deep-dives to market research to academic literature reviews. The key seems to be the combination of parallel exploration (via multiple sub-agents) and intelligent synthesis (via the lead agent).

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

## Wrapping Up

Building a multi-agent research system turned out to be less about advanced AI techniques and more about thoughtful system design. The orchestrator-worker pattern, structured outputs, and streaming updates combine to create something that feels genuinely useful in ways that traditional AI interactions don't.

The key insights were borrowed directly from Anthropic's excellent work on multi-agent systems: start simple, add complexity only when it demonstrably improves outcomes, and design for tasks where you can't predict the required steps in advance.

If you're working on similar problems, we'd encourage you to start with single agents and add orchestration only when you hit clear limitations. But when you do need coordination between multiple agents, the patterns we've shared here provide a solid foundation.

The code for our implementation is available in the repository, and we're excited to see what others build with these concepts. Multi-agent systems are still in their early days, but they're pointing toward more flexible and capable AI applications.

What would you research if you had a team of AI agents at your disposal?