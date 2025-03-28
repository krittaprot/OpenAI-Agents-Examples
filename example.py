import asyncio
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent

local_model = OpenAIChatCompletionsModel(
    model="qwen2.5-14b-instruct-1m",
    openai_client=AsyncOpenAI(base_url="http://localhost:1234/v1", api_key='lm-studio')
)

agent = Agent(
    name="Assistant",
    instructions="You are a helpful reasoning assistant",
    model=local_model
)

async def stream_response():
    result = Runner.run_streamed(agent, "Create a simple SwiftUI animation that moves a rectangle from point a to b repeatedly.")
    # Use stream_events() instead of stream
    async for event in result.stream_events():
        # Handle raw response events for text streaming
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end="", flush=True)

if __name__ == "__main__":
    # Run the async function in the event loop
    asyncio.run(stream_response())