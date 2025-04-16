from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig,
    set_tracing_disabled,
)
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv())

google_api_key = os.getenv("GOOGLE_API_KEY")

set_tracing_disabled(disabled=True)


# externel client
externel_client = AsyncOpenAI(
    api_key=google_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash", openai_client=externel_client
)

# Config
config = RunConfig(model=model, model_provider=externel_client)

# Agent
agent = Agent(name="Assistant", instructions="You are my personal assistant")

# Result
result = Runner.run_sync(
    starting_agent=agent, input="Write hello world five time", run_config=config
)

# Result Output
print(result.final_output)
