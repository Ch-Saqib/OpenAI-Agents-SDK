import json
from typing import Any
from agents import (
    Agent,
    AsyncOpenAI,
    FunctionTool,
    OpenAIChatCompletionsModel,
    RunConfig,
    RunContextWrapper,
    Runner,
    set_tracing_disabled,
    function_tool,
    enable_verbose_stdout_logging,
)
from dotenv import load_dotenv, find_dotenv
import os

from pydantic import BaseModel

load_dotenv(find_dotenv())

set_tracing_disabled(disabled=True)

enable_verbose_stdout_logging()

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


@function_tool(name_override="Get_Data", is_enabled=True, use_docstring_info=False,failure_error_function=None)
def get_data(name: str):
    """
    this function  is used for get data

    Args:
    name :str

    """
    return {"message": f"Hello World {name.name}"}


@function_tool(name_override="fetch_data")
def read_file(ctx: RunContextWrapper[Any], path: str, directory: str | None = None) -> str:
    """Read the contents of a file.

    Args:
        path: The path to the file to read.
        directory: The directory to read the file from.
    """
    # In real life, we'd read the file from the file system
    return "<file contents>"


def do_some_work(data: str) -> str:
    return "done"


class FunctionArgs(BaseModel):
    username: str
    age: int


async def run_function(ctx: RunContextWrapper[Any], args: str) -> str:
    parsed = FunctionArgs.model_validate_json(args)
    return do_some_work(data=f"{parsed.username} is {parsed.age} years old")


tool = FunctionTool(
    name="process_user",
    description="Processes extracted user data",
    params_json_schema=FunctionArgs.model_json_schema(),
    on_invoke_tool=run_function,
)

french_agent = Agent(
    name="French agent",
    instructions="You translate the user's message to French",
)

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant",
    tools=[get_data,read_file,french_agent.as_tool(tool_name="French",tool_description="You translate the user's message to French")],
)


result = Runner.run_sync(
    starting_agent=agent, input=" call get data and my name  is john", run_config=config
)

print(result.final_output)
