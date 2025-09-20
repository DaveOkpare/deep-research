import os
from openai import OpenAI
from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

tracer_provider = register(
    project_name="open-deep-research",
    endpoint="https://app.phoenix.arize.com/s/okparedave/v1/traces",
    auto_instrument=True,
)

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
