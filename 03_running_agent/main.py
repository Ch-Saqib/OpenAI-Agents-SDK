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


@function_tool
async def data():
    return {"Message": "Hello World"}


def main():
    agent = Agent(
        name="Assistant", instructions="You are a helpful assistant", tools=[data]
    )

    result = Runner.run_sync(
        agent,
        "Write a haiku about recursion in programming.",
        run_config=config,
    )

    print("Agent 01  :",result.final_output)

    new_input = result.to_input_list() + [
        {"role": "user", "content": "Write hello world 2  time"}
    ]
    result = Runner.run_sync(
        agent,
        new_input,
        run_config=config,
    )
    print("Agent 02  :",result.final_output)


# Call for Async  
# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())

# Call For Sync
print(main())
