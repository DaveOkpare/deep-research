from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.messages import ModelRequest, ModelResponse, UserPromptPart, TextPart
import json


# Initialize Pydantic AI Agent
agent = Agent(
    OpenAIModel('gpt-4o-mini'),
    system_prompt="You are a helpful AI assistant. Provide clear and concise responses."
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


async def stream_response(message_history, user_prompt, message_id):
    """Stream AI response with conversation history"""
    try:
        yield f'data: {{"type":"text-start","id":"{message_id}"}}\n\n'
        
        previous_text = ""
        async with agent.run_stream(
            user_prompt,
            message_history=message_history if message_history else None
        ) as result:
            async for message in result.stream():
                delta = message[len(previous_text):] if message.startswith(previous_text) else message
                if delta:
                    chunk_data = json.dumps({"type": "text-delta", "id": message_id, "delta": delta})
                    yield f'data: {chunk_data}\n\n'
                    previous_text = message
        
        yield f'data: {{"type":"text-end","id":"{message_id}"}}\n\n'
        yield f'data: [DONE]\n\n'
        
    except Exception as e:
        error_data = json.dumps({"type": "text-delta", "id": message_id, "delta": f"Error: {str(e)}"})
        yield f'data: {error_data}\n\n'
        yield f'data: {{"type":"text-end","id":"{message_id}"}}\n\n'
        yield f'data: [DONE]\n\n'


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
    
    response = StreamingResponse(
        stream_response(message_history, user_prompt, message_id),
        media_type="text/event-stream"
    )
    response.headers['x-vercel-ai-data-stream'] = 'v1'
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    return response