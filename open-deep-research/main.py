import os
from typing import TypedDict
from openai import OpenAI
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor
from pydantic import BaseModel, Field

tracer_provider = register(
    project_name="open-deep-research",
    endpoint="https://app.phoenix.arize.com/s/okparedave/v1/traces",
    auto_instrument=True,
)

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
model = "gpt-4.1-mini"


class AgentState(TypedDict, total=False):  # total=False makes all keys optional
    messages: list
    research_brief: str


class ClarifyWithUser(BaseModel):
    needs_clarification: bool = Field(
        description="Whether the user needs to be asked a clarifying question."
    )
    question: str = Field(
        description="A question to ask the user to clarify the research scope."
    )
    verification: str = Field(
        description="Verify message that we will proceed after the user has provided the necessary information."
    )


def clarify_with_user(state: AgentState, query: str) -> str:
    instructions = f"""You are a research query analyzer. Your task is to determine if a user's research query contains sufficient information to proceed with comprehensive research, or if clarification is needed.

Analyze the user's query and consider these factors:
1. **Specificity**: Is the research topic clearly defined? Are there vague terms that could mean multiple things?
2. **Scope**: Is the research scope reasonable and well-bounded? Is it too broad or too narrow?
3. **Context**: Is there enough context to understand what kind of research is needed?
4. **Intent**: Is the user's goal for the research clear?
5. **Actionability**: Can meaningful research be conducted with the given information?

Decision Process:
- If the query has sufficient specificity, scope, context, intent, and actionability → needs_clarification = False
- If any critical information is missing or unclear → needs_clarification = True

Output Requirements:
- Always set needs_clarification (True or False)
- If needs_clarification = True: Return a focused question to clarify the most important missing information
- If needs_clarification = False: Return a verification message confirming what research will be conducted

Output Format Examples:

When clarification is needed:
{{
  "needs_clarification": true,
  "question": "[specific question to clarify the most important missing information]",
  "verification": ""
}}

When query is sufficient:
{{
  "needs_clarification": false,
  "question": "",
  "verification": "[confirmation message describing what research will be conducted]"
}}
    """

    state["messages"].extend([{"role": "user", "content": query}])

    response: ClarifyWithUser = client.responses.parse(
        model=model,
        instructions=instructions,
        input=state["messages"],
        text_format=ClarifyWithUser,
    ).output_parsed

    if response.needs_clarification:
        return response.question
    else:
        return response.verification

