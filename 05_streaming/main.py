import random
from agents import (
    Agent,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig,
    Runner,
    set_tracing_disabled,
    function_tool,
    ItemHelpers,
)
from openai.types.responses import ResponseTextDeltaEvent
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
def how_many_jokes() -> int:
    return random.randint(1, 10)


agent = Agent(name="Assistant", instructions="First call the `how_many_jokes` tool, then tell that many jokes.",tools=[how_many_jokes])


# async def main():
#     response = ""
#     result = Runner.run_streamed(starting_agent=agent, input="HI", run_config=config)

#     async for events in result.stream_events():
#         if events.type == "raw_response_event" and isinstance(events.data,ResponseTextDeltaEvent):
#             chunks = events.data.delta
#             response += chunks
#     return response


# import asyncio

# print(asyncio.run(main()))


async def main():
    result = Runner.run_streamed(starting_agent=agent, input="Hi", run_config=config)

    async for events in result.stream_events():
        if events.type == "raw_response_event":
            continue
        elif events.type == "agent_updated_stream_event":
            print(f"Agent updated: {events.new_agent.name}")
            continue
        elif events.type == "run_item_stream_event":
            if events.item.type == "tool_call_item":
                print("-- Tool was called")
            elif events.item.type == "tool_call_output_item":
                print(f"-- Tool output: {events.item.output}")
            elif events.item.type == "message_output_item":
                print(
                    f"-- Message output:\n {ItemHelpers.text_message_output(events.item)}"
                )
            else:
                pass


print("=== Run complete ===")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
