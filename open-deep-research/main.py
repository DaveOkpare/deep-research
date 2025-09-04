import os
from openai import OpenAI
from arize.otel import register
from dotenv import load_dotenv
from openinference.instrumentation.openai import OpenAIInstrumentor

load_dotenv()

tracer_provider = register(
    space_id = os.getenv("ARIZE_SPACE_ID"), # in app space settings page
    api_key = os.getenv("ARIZE_API_KEY"), # in app space settings page
    project_name = "open-deep-research", 
)

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

client = OpenAI()

response = client.responses.create(
  model="gpt-4.1",
  input="Tell me a three sentence bedtime story about a unicorn."
)

print(response)
