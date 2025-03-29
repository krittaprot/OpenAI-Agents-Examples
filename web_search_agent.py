# %% [markdown]
# # OpenAI Agents SDK with SearXNG Web Search Tool

# %% Imports
import os
import json
import asyncio
import requests
from typing import List, Dict, Union, Optional

from pydantic import BaseModel
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    AsyncOpenAI,
    function_tool,
    GuardrailFunctionOutput,
    RunContextWrapper,
    input_guardrail,
)
from openai.types.responses import (
    ResponseTextDeltaEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
    ResponseCreatedEvent,
)

from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY in .env file")

# %% [markdown]
# ## Configuration
# **IMPORTANT:** Set your API keys and SearXNG instance URL below.

# %% Configuration
# --- OpenAI/Local LLM Configuration ---
# Use "lm-studio" or your actual OpenAI key if using their service
from agents import set_tracing_export_api_key
set_tracing_export_api_key(os.getenv("OPENAI_API_KEY"))

# Replace with your local model identifier if using LM Studio or similar
LOCAL_MODEL_NAME = "qwen2.5-14b-instruct@iq4_xs"
# Replace with the base URL of your local LLM API endpoint
LOCAL_API_BASE_URL = "http://localhost:1234/v1"

# --- SearXNG Configuration ---
# !!! REPLACE THIS WITH THE BASE URL OF YOUR SEARXNG INSTANCE !!!
# Example: "http://192.168.1.63:4000" or "http://127.0.0.1:8080"
# Do NOT include '/search' at the end.
SEARXNG_BASE_URL = "http://192.168.1.63:4000" # <--- CHANGE THIS

# Check if SearXNG URL is set
if not SEARXNG_BASE_URL or SEARXNG_BASE_URL == "YOUR_SEARXNG_INSTANCE_URL_HERE":
    print("⚠️ WARNING: SEARXNG_BASE_URL is not set. The search tool will not work.")
    print("Please edit the script and set the SEARXNG_BASE_URL variable.")
    # You might want to exit or disable the tool if the URL isn't set
    # exit()


# %% [markdown]
# ## SearXNG Web Search Tool Definition

# %% SearXNG Tool Function
# Define default headers - can be overridden by passing 'headers' to the function
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
}

@function_tool
def search_searxng(
    query: str,
    language: str = 'en',
    safesearch: int = 0, # 0: none, 1: moderate, 2: strict
    max_results: Optional[int] = 5, # Limit results for LLM context
    timeout: int = 15,
    headers: Optional[Dict[str, str]] = None,
    categories: str = 'general' # Use 'general', 'news', 'images', etc.
) -> Dict[str, Union[List[Dict[str, str]], str]]:
    """
    Performs a web search using a configured SearXNG instance to find up-to-date
    information or answer questions about recent events, facts, or topics not
    likely in the LLM's training data.

    Args:
        query: The specific search query string.
        language: Language code for search results (e.g., 'en', 'de'). Defaults to 'en'.
        safesearch: SafeSearch level (0=off, 1=moderate, 2=strict). Defaults to 0.
        max_results: Max number of results to return. Defaults to 5.
        categories: Search category (e.g., 'general', 'news'). Defaults to 'general'.
        timeout: Request timeout in seconds. (Internal parameter, not for LLM)
        headers: Custom HTTP headers. (Internal parameter, not for LLM)


    Returns:
        A dictionary containing either 'results' (a list of {title, url, snippet} dicts)
        or 'error' (a string describing the issue).
    """
    # Use the globally configured SEARXNG_BASE_URL
    base_url = SEARXNG_BASE_URL
    if not base_url or base_url == "YOUR_SEARXNG_INSTANCE_URL_HERE":
        return {"error": "SearXNG base URL is not configured in the environment."}

    search_endpoint = f"{base_url.rstrip('/')}/search"
    search_headers = headers if headers else DEFAULT_HEADERS

    params = {
        'q': query,
        'categories': categories,
        'language': language,
        'safesearch': str(safesearch),
        'format': 'json',
    }

    print(f"\n[SYSTEM: Calling SearXNG Tool]")
    print(f"DEBUG: Querying SearXNG endpoint: {search_endpoint}")
    print(f"DEBUG: Using parameters: {params}")

    response = None
    try:
        response = requests.get(
            search_endpoint,
            params=params,
            headers=search_headers,
            timeout=timeout
        )
        response.raise_for_status()
        search_data = response.json()

        formatted_results = []

        # Process standard 'results'
        if 'results' in search_data and isinstance(search_data['results'], list):
            results_list = search_data['results']
            if max_results is not None:
                results_list = results_list[:max_results]
            for result in results_list:
                formatted_results.append({
                    "title": result.get('title', 'No title'),
                    "url": result.get('url', 'No URL'),
                    "snippet": result.get('content', 'No snippet.')
                })

        # Process 'infoboxes'
        if 'infoboxes' in search_data and isinstance(search_data['infoboxes'], list):
             for box in search_data['infoboxes']:
                 title = f"Infobox ({box.get('engine', 'Source Unknown')})"
                 content = box.get('content', 'No content.')
                 infobox_url = None
                 if box.get('urls') and isinstance(box['urls'], list) and box['urls']:
                     infobox_url = box['urls'][0].get('url')
                 formatted_results.append({
                     "title": title,
                     "url": infobox_url or box.get('infobox_url', 'No URL'),
                     "snippet": content
                 })
                 if max_results is not None and len(formatted_results) >= max_results:
                     break # Stop if limit reached

        if not formatted_results:
             print("DEBUG: No results or infoboxes found.")
             return {"results": []} # Return success with empty list

        print(f"DEBUG: Found {len(formatted_results)} results/infoboxes.")
        return {"results": formatted_results}

    except requests.exceptions.Timeout:
        error_msg = f"Error: SearXNG request timed out ({search_endpoint})."
        print(error_msg)
        return {"error": error_msg}
    except requests.exceptions.ConnectionError:
        error_msg = f"Error: Could not connect to SearXNG at {search_endpoint}."
        print(error_msg)
        return {"error": error_msg}
    except requests.exceptions.HTTPError as e:
        error_msg = f"Error: SearXNG HTTP error {e.response.status_code}."
        print(error_msg)
        try:
            error_details = e.response.text[:200]
            error_msg += f" Response: {error_details}"
            print(f"Response snippet: {error_details}")
        except Exception: pass
        return {"error": error_msg}
    except requests.exceptions.JSONDecodeError:
        error_msg = "Error: Failed to decode JSON from SearXNG."
        print(error_msg)
        if response: print("Response Text Snippet:", response.text[:200])
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"An unexpected error occurred during search: {e}"
        import traceback
        print(error_msg)
        print(traceback.format_exc())
        return {"error": error_msg}


# %% [markdown]
# ## Optional Guardrail Definition (Example)

# %% Guardrail Code (Optional - Not enabled by default in main loop)
# define structure of output for any guardrail agents
class GuardrailOutput(BaseModel):
    is_triggered: bool
    reasoning: str

# define an agent that checks if user is asking about political opinions
politics_agent = Agent(
    name="Politics check",
    instructions="Check if the user is asking you about political opinions. Respond ONLY with the JSON structure containing 'is_triggered' (boolean) and 'reasoning' (string).",
    output_type=GuardrailOutput,
    model= OpenAIChatCompletionsModel( # Use a separate model instance if needed
        model=LOCAL_MODEL_NAME,
        openai_client=AsyncOpenAI(base_url=LOCAL_API_BASE_URL)
    )
)

# this is the guardrail function that returns GuardrailFunctionOutput object
@input_guardrail
async def politics_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str,
) -> GuardrailFunctionOutput:
    print("\n[SYSTEM: Running Politics Guardrail]")
    # run agent to check if guardrail is triggered
    response = await Runner.run(starting_agent=politics_agent, input=input)
    # format response into GuardrailFunctionOutput
    if response.final_output and isinstance(response.final_output, GuardrailOutput):
        print(f"[Guardrail Check: Triggered={response.final_output.is_triggered}, Reason='{response.final_output.reasoning}']")
        return GuardrailFunctionOutput(
            output_info=response.final_output,
            tripwire_triggered=response.final_output.is_triggered,
        )
    else:
        print("[Guardrail Check: Error processing guardrail output]")
        # Default to not triggered if guardrail fails
        return GuardrailFunctionOutput(tripwire_triggered=False)


# %% [markdown]
# ## Main Agent Definition and Chat Loop

# %% Main Agent and Interaction Loop
async def main():
    print("Initializing LLM client...")
    # Setup the main LLM client
    local_model = OpenAIChatCompletionsModel(
        model=LOCAL_MODEL_NAME,
        openai_client=AsyncOpenAI(base_url=LOCAL_API_BASE_URL)
    )

    print("Initializing Assistant Agent...")
    # Define the main assistant agent with the search tool
    assistant_agent = Agent(
        name="Assistant",
        instructions=(
            "You are a helpful and informative assistant. "
            "If you need current information, facts you don't know, "
            "or details about recent events, use the 'search_searxng' tool. "
            "Clearly state when you are searching and summarize the findings "
            "from the search results to answer the user's query. "
            "Always cite the source URL when using information from a search result. "
            "Be conversational and friendly."
        ),
        model=local_model,
        tools=[search_searxng], # Add the search tool here
        # input_guardrails=[politics_guardrail], # <-- Uncomment to enable guardrail
    )

    print("Starting interactive chat session...")
    print("Type 'quit' or 'exit' to end the session.")

    chat_history = [] # To store conversation history

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ["quit", "exit"]:
                print("Ending chat session. Goodbye!")
                break

            # Prepare input for the agent, including history
            current_input = chat_history + [{"role": "user", "content": user_input}]

            print("Assistant: ", end="", flush=True)

            # Run the agent with streaming
            response = Runner.run_streamed(
                starting_agent=assistant_agent,
                input=current_input
            )

            full_response_content = ""
            tool_outputs = [] # Store tool outputs for history

            # Process streamed events
            async for event in response.stream_events():
                if event.type == "raw_response_event":
                    if isinstance(event.data, ResponseTextDeltaEvent):
                        token = event.data.delta
                        print(token, end="", flush=True)
                        full_response_content += token
                    elif isinstance(event.data, ResponseFunctionCallArgumentsDeltaEvent):
                        # Optionally print tool arguments as they stream
                        # print(event.data.delta, end="", flush=True)
                        pass # Usually handled by the 'tool_called' event below
                elif event.type == "run_item_stream_event":
                    if event.name == "tool_called":
                        # Tool call details (name and full arguments) are available here
                        tool_name = event.item.raw_item.name
                        tool_args = event.item.raw_item.arguments
                        print(f"\n[SYSTEM: Tool '{tool_name}' called with args: {tool_args}]", flush=True)
                        print("Assistant: ...thinking...", end="", flush=True) # Indicate processing
                    elif event.name == "tool_output":
                        # Tool execution result is available here
                        tool_output_data = event.item.raw_item['output']
                        print(f"\n[SYSTEM: Tool returned output.]", flush=True)
                        # Store for history, but don't print the raw output directly unless debugging
                        tool_outputs.append(event.item.raw_item) # Store the whole item for history context
                        print("Assistant: ", end="", flush=True) # Resume printing assistant response

            print() # Newline after the assistant finishes

            # Update history *after* the full response stream is processed
            # Use the response object's method to get the correct history format
            if response:
                 chat_history = response.to_input_list()
                 # Optional: Prune history if it gets too long
                 # MAX_HISTORY_TURNS = 5
                 # if len(chat_history) > MAX_HISTORY_TURNS * 2:
                 #    chat_history = chat_history[-(MAX_HISTORY_TURNS * 2):]

        except Exception as e:
            print(f"\nAn error occurred: {e}")
            import traceback
            traceback.print_exc()
            # Optionally reset history or try to recover
            # chat_history = []


if __name__ == "__main__":
    # Ensure SearXNG URL is set before running
    if not SEARXNG_BASE_URL or SEARXNG_BASE_URL == "YOUR_SEARXNG_INSTANCE_URL_HERE":
        print("ERROR: Cannot start chat. SEARXNG_BASE_URL is not configured.")
        print("Please edit the script and set the correct URL.")
    else:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\nCaught interrupt, exiting.")