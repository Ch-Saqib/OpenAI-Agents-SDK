from agents import (
    Agent,
    Runner,
    AsyncOpenAI,
    OpenAIChatCompletionsModel,
    RunConfig,
    set_tracing_disabled,
)
from dotenv import load_dotenv, find_dotenv
import chainlit as cl
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


# UI
@cl.on_message
async def chat_agent(message: cl.Message):
    result = await Runner.run(starting_agent=agent, input=message.content, run_config=config)
    await cl.Message(content=result.final_output).send()
