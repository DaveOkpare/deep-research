from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
from datetime import datetime
import json

from agent import lead_agent, DateDeps
from models import ResearchReport


# Initialize Pydantic AI Agent
agent = Agent(
    OpenAIModel("gpt-4o-mini"),
    system_prompt="You are a helpful AI assistant. Provide clear and concise responses.",
)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def convert_messages_to_pydantic(messages):
    """Convert AI SDK messages to Pydantic AI message format"""
    converted = []

    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or "role" not in msg:
            continue

        role = msg["role"]
        content = ""

        # Extract text from parts
        if "parts" in msg and msg["parts"]:
            for part in msg["parts"]:
                if part.get("type") == "text":
                    content = part.get("text", "")
                    break

        # Fallback to direct content field
        if not content:
            content = msg.get("content", msg.get("text", ""))

        if not content:
            continue

        # Convert to proper Pydantic AI message format
        if role == "user":
            converted.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        elif role == "assistant":
            converted.append(ModelResponse(parts=[TextPart(content=content)]))

    return converted


def stream_to_markdown(partial_report: dict) -> str:
    """Convert partial ResearchReport to complete markdown."""
    try:
        md = ""

        # Add title if available
        if "title" in partial_report and partial_report["title"]:
            md += f"# {partial_report['title']}\n\n"

        # Add executive summary if available
        if (
            "executive_summary" in partial_report
            and partial_report["executive_summary"]
        ):
            md += f"## Executive Summary\n\n{partial_report['executive_summary']}\n\n"

        # Add sections if available
        if "sections" in partial_report and partial_report["sections"]:
            for section in partial_report["sections"]:
                if not isinstance(section, dict):
                    continue

                section_title = section.get("title", "")
                section_content = section.get("content", "")

                if section_title:
                    md += f"## {section_title}\n\n"

                if section_content:
                    md += f"{section_content}\n\n"

                # Handle subsections
                if "subsections" in section and section["subsections"]:
                    for subsection in section["subsections"]:
                        if not isinstance(subsection, dict):
                            continue
                        sub_title = subsection.get("title", "")
                        sub_content = subsection.get("content", "")

                        if sub_title:
                            md += f"### {sub_title}\n\n"

                        if sub_content:
                            md += f"{sub_content}\n\n"

        # Add key takeaways if available
        if "key_takeaways" in partial_report and partial_report["key_takeaways"]:
            md += "## Key Takeaways\n\n"
            for i, takeaway in enumerate(partial_report["key_takeaways"], 1):
                md += f"{i}. {takeaway}\n"

        return md
    except Exception:
        return ""


async def stream_response(message_history, user_prompt, message_id):
    """Stream AI response with conversation history"""
    try:
        yield f'data: {{"type":"text-start","id":"{message_id}"}}\n\n'

        previous_text = ""
        async with agent.run_stream(
            user_prompt, message_history=message_history if message_history else None
        ) as result:
            async for message in result.stream():
                delta = (
                    message[len(previous_text) :]
                    if message.startswith(previous_text)
                    else message
                )
                if delta:
                    chunk_data = json.dumps(
                        {"type": "text-delta", "id": message_id, "delta": delta}
                    )
                    yield f"data: {chunk_data}\n\n"
                    previous_text = message

        yield f'data: {{"type":"text-end","id":"{message_id}"}}\n\n'
        yield f"data: [DONE]\n\n"

    except Exception as e:
        error_data = json.dumps(
            {"type": "text-delta", "id": message_id, "delta": f"Error: {str(e)}"}
        )
        yield f"data: {error_data}\n\n"
        yield f'data: {{"type":"text-end","id":"{message_id}"}}\n\n'
        yield f"data: [DONE]\n\n"


async def stream_research_response(user_prompt: str, message_id: str):
    """Stream structured research response as markdown."""
    try:
        yield f'data: {{"type":"text-start","id":"{message_id}"}}\n\n'

        current_date = datetime.now().strftime("%Y-%m-%d")
        deps = DateDeps(current_date=current_date)

        previous_markdown = ""

        async with lead_agent.run_stream(user_prompt, deps=deps) as result:
            async for partial_output in result.stream():
                if partial_output:
                    # Convert partial output to dict if it's a structured object
                    if hasattr(partial_output, "model_dump"):
                        partial_dict = partial_output.model_dump()
                    elif isinstance(partial_output, dict):
                        partial_dict = partial_output
                    else:
                        # Fallback for string or other types
                        current_markdown = str(partial_output)
                        if current_markdown != previous_markdown:
                            delta = current_markdown[len(previous_markdown) :]
                            if delta:
                                chunk_data = json.dumps(
                                    {
                                        "type": "text-delta",
                                        "id": message_id,
                                        "delta": delta,
                                    }
                                )
                                yield f"data: {chunk_data}\n\n"
                                previous_markdown = current_markdown
                        continue

                    # Generate complete markdown from partial structured output
                    current_markdown = stream_to_markdown(partial_dict)

                    # Only send delta if there's new content
                    if current_markdown != previous_markdown:
                        delta = current_markdown[len(previous_markdown) :]
                        if delta:
                            chunk_data = json.dumps(
                                {"type": "text-delta", "id": message_id, "delta": delta}
                            )
                            yield f"data: {chunk_data}\n\n"
                        previous_markdown = current_markdown

        yield f'data: {{"type":"text-end","id":"{message_id}"}}\n\n'
        yield f"data: [DONE]\n\n"

    except Exception as e:
        error_data = json.dumps(
            {"type": "text-delta", "id": message_id, "delta": f"Error: {str(e)}"}
        )
        yield f"data: {error_data}\n\n"
        yield f'data: {{"type":"text-end","id":"{message_id}"}}\n\n'
        yield f"data: [DONE]\n\n"


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        messages = body.get("messages", [])

        if not messages:
            raise HTTPException(400, "No messages provided")

        # Extract webSearch flag from request body
        web_search = body.get("webSearch", False)

        # Get the current user prompt (last message)
        last_message = messages[-1]
        user_prompt = ""

        # Extract text from the last message
        if "parts" in last_message and last_message["parts"]:
            for part in last_message["parts"]:
                if part.get("type") == "text":
                    user_prompt = part.get("text", "")
                    break

        if not user_prompt:
            user_prompt = last_message.get("content", last_message.get("text", ""))

        if not user_prompt:
            raise HTTPException(400, "No user message found")

        # Convert previous messages (excluding current) to history
        message_history = convert_messages_to_pydantic(messages[:-1])

        # Get message ID
        message_id = last_message.get("id")
        if not message_id:
            raise HTTPException(400, "No message ID provided")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Invalid request format: {str(e)}")

    # Route to appropriate streaming function based on webSearch flag
    if web_search:
        streaming_generator = stream_research_response(user_prompt, message_id)
    else:
        streaming_generator = stream_response(message_history, user_prompt, message_id)

    response = StreamingResponse(streaming_generator, media_type="text/event-stream")
    response.headers["x-vercel-ai-data-stream"] = "v1"
    response.headers["Cache-Control"] = "no-cache"
    response.headers["Connection"] = "keep-alive"
    return response
