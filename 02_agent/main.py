from dataclasses import dataclass
from typing_extensions import TypedDict
from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    set_tracing_disabled,
    function_tool,
)
from dotenv import load_dotenv, find_dotenv
import os

from pydantic import BaseModel

load_dotenv(find_dotenv())

set_tracing_disabled(disabled=True)

gemini_api_key = os.getenv("GOOGLE_API_KEY")

# externel client
externel_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Model
model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash", openai_client=externel_client
)

# Config
config = RunConfig(model=model, model_provider=externel_client)


# tool
@function_tool
def get_weather(city: str) -> str:
    return f"The weather in {city} is sunny"

@dataclass
class AgentOutput:
    agent_output: str

# Agent
assistant_agent = Agent(
    name="Assistant",
    instructions="You are phd math teacher.you answer only about math if user ask you other topic do not answer and  say i have no knowledge",
    model=model,
    output_type=AgentOutput
)

# Runner
result = Runner.run_sync(
    starting_agent=assistant_agent, input="Hi", run_config=config
)

print(result.final_output)

