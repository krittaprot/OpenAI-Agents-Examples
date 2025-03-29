import os
import json
import asyncio
import requests
import streamlit as st
from typing import List, Dict, Union, Optional
from datetime import datetime

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
    set_tracing_export_api_key
)
from openai.types.responses import (
    ResponseTextDeltaEvent,
    ResponseFunctionCallArgumentsDeltaEvent,
)

from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY in .env file")

# --- Page Configuration ---
st.set_page_config(
    page_title="SearXNG Web Search Assistant",
    page_icon="🔍",
    layout="wide",
)

# --- Sidebar for Configuration ---
st.sidebar.title("Configuration")

# OpenAI API Key input
openai_api_key = st.sidebar.text_input(
    "OpenAI API Key",
    value=os.environ.get("OPENAI_API_KEY", ""),
    type="password"
)

# SearXNG URL input
searxng_base_url = st.sidebar.text_input(
    "SearXNG Base URL",
    value=os.environ.get("SEARXNG_BASE_URL", "http://192.168.1.63:4000")
)

# LLM Model selection
local_model_name = st.sidebar.text_input(
    "LLM Model Name",
    value="qwen2.5-14b-instruct@iq4_xs"
)

# LLM API Base URL
local_api_base_url = st.sidebar.text_input(
    "LLM API Base URL",
    value="http://localhost:1234/v1"
)

# Enable guardrail option
enable_guardrail = st.sidebar.checkbox("Enable Politics Guardrail", value=False)

# Apply configuration button
if st.sidebar.button("Apply Configuration"):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    set_tracing_export_api_key(openai_api_key)
    st.sidebar.success("Configuration applied!")

# --- SearXNG Tool Function ---
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
}

@function_tool
def search_searxng(
    query: str,
    language: str = 'en',
    safesearch: int = 0,
    max_results: Optional[int] = 5,
    timeout: int = 15,
    headers: Optional[Dict[str, str]] = None,
    categories: str = 'general'
) -> Dict[str, Union[List[Dict[str, str]], str]]:
    """
    Performs a web search using a configured SearXNG instance to find up-to-date
    information or answer questions about recent events, facts, or topics not
    likely in the LLM's training data.
    """
    # Use the configured SEARXNG_BASE_URL
    base_url = searxng_base_url
    if not base_url:
        return {"error": "SearXNG base URL is not configured."}

    search_endpoint = f"{base_url.rstrip('/')}/search"
    search_headers = headers if headers else DEFAULT_HEADERS

    params = {
        'q': query,
        'categories': categories,
        'language': language,
        'safesearch': str(safesearch),
        'format': 'json',
    }

    st.session_state['status_message'] = f"Searching for: {query}"

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
                     break

        if not formatted_results:
            st.session_state['status_message'] = "No search results found."
            return {"results": []}

        st.session_state['status_message'] = f"Found {len(formatted_results)} results."
        return {"results": formatted_results}

    except Exception as e:
        error_msg = f"Error during search: {str(e)}"
        st.session_state['status_message'] = error_msg
        return {"error": error_msg}

# --- Guardrail Definition ---
class GuardrailOutput(BaseModel):
    is_triggered: bool
    reasoning: str

# Define an agent that checks if user is asking about political opinions
def create_politics_agent(model_config):
    return Agent(
        name="Politics check",
        instructions="Check if the user is asking you about political opinions. Respond ONLY with the JSON structure containing 'is_triggered' (boolean) and 'reasoning' (string).",
        output_type=GuardrailOutput,
        model=model_config
    )

@input_guardrail
async def politics_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str,
) -> GuardrailFunctionOutput:
    st.session_state['status_message'] = "Running politics guardrail check..."
    politics_agent = create_politics_agent(agent.model)
    response = await Runner.run(starting_agent=politics_agent, input=input)
    
    if response.final_output and isinstance(response.final_output, GuardrailOutput):
        st.session_state['status_message'] = f"Guardrail check: {response.final_output.reasoning}"
        return GuardrailFunctionOutput(
            output_info=response.final_output,
            tripwire_triggered=response.final_output.is_triggered,
        )
    else:
        st.session_state['status_message'] = "Error processing guardrail output"
        return GuardrailFunctionOutput(tripwire_triggered=False)

# --- Streamlit App Functions ---
def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'status_message' not in st.session_state:
        st.session_state.status_message = "Ready"

async def process_message(user_input):
    """Process user input and get AI response"""
    # Update status
    st.session_state['status_message'] = "Processing your message..."
    status_placeholder = st.empty()
    status_placeholder.markdown(f"*Status: {st.session_state['status_message']}*")
    
    # Get current model configuration
    model_config = OpenAIChatCompletionsModel(
        model=local_model_name,
        openai_client=AsyncOpenAI(base_url=local_api_base_url)
    )
    
    # Create assistant agent with search tool
    tools = [search_searxng]
    guardrails = [politics_guardrail] if enable_guardrail else []
    
    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    assistant_agent = Agent(
        name="Assistant",
        instructions=(
            f"You are a helpful and informative assistant. Today's date and time is {current_datetime}. "
            "If you need current information, facts you don't know, "
            "or details about recent events, use the 'search_searxng' tool. "
            "Clearly state when you are searching and summarize the findings "
            "from the search results to answer the user's query. "
            "Always cite the source URL when using information from a search result. "
            "Be conversational and friendly."
        ),
        model=model_config,
        tools=tools,
        input_guardrails=guardrails,
    )

    
    # Prepare input for the agent, including history
    current_input = st.session_state.chat_history + [{"role": "user", "content": user_input}]
    
    # Use a placeholder for the assistant's response that we'll update
    assistant_response_placeholder = st.empty()
    
    # Run the agent with streaming
    response = Runner.run_streamed(
        starting_agent=assistant_agent,
        input=current_input
    )
    
    full_response_content = ""
    
    # Process streamed events
    async for event in response.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent):
                token = event.data.delta
                full_response_content += token
                # Update the placeholder with the current full response
                assistant_response_placeholder.markdown(full_response_content + "▌")
        elif event.type == "run_item_stream_event":
            if event.name == "tool_called":
                # Tool call details
                tool_name = event.item.raw_item.name
                st.session_state['status_message'] = f"Using tool: {tool_name}"
                status_placeholder.markdown(f"*Status: {st.session_state['status_message']}*")
            elif event.name == "tool_output":
                st.session_state['status_message'] = "Tool execution complete"
                status_placeholder.markdown(f"*Status: {st.session_state['status_message']}*")
    
    # Final response without cursor
    assistant_response_placeholder.markdown(full_response_content)
    
    # Update history after the full response is processed
    if response:
        st.session_state.chat_history = response.to_input_list()
    
    # Update status
    st.session_state['status_message'] = "Ready"
    status_placeholder.markdown(f"*Status: {st.session_state['status_message']}*")
    
    return full_response_content

# --- Main Streamlit App ---
def main():
    st.title("SearXNG Web Search Assistant")
    
    # Initialize session state
    initialize_session_state()
    
    # Display status message
    st.markdown(f"*Status: {st.session_state['status_message']}*")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Check for configuration
        if not openai_api_key or not searxng_base_url:
            with st.chat_message("assistant"):
                st.markdown("⚠️ Please set your OpenAI API Key and SearXNG URL in the sidebar before continuing.")
            return
        
        # Get and display assistant response
        with st.chat_message("assistant"):
            # Run the assistant in the background
            response = asyncio.run(process_message(user_input))
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

if __name__ == "__main__":
    main()
