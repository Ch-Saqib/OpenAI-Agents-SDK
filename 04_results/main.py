from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    set_tracing_disabled,
    ModelSettings,
    function_tool,
)
from dotenv import load_dotenv, find_dotenv
import os

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


def main():
    agent = Agent(name="Assistant", instructions="You are a helpful assistant")

    result = Runner.run_sync(
        agent,
        "Write a haiku about recursion in programming.",
        run_config=config,
    )

    print(result.raw_responses)


print(main())
