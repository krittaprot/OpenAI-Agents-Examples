# searxng_mcp_server.py
import os
import json
import requests
from typing import List, Dict, Union, Optional
import logging

from mcp.server.fastmcp import FastMCP

# Configure basic logging for the server process
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- MCP Server Definition ---
mcp = FastMCP("SearXNG Search Server")

# --- SearXNG Tool Function (MCP Version) ---
DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
}

@mcp.tool()
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
    likely in the LLM's training data. Reads SEARXNG_BASE_URL from environment.
    """
    # Read the base URL from the environment variable passed by the client
    base_url = os.environ.get("SEARXNG_BASE_URL")
    if not base_url:
        error_msg = "SearXNG base URL (SEARXNG_BASE_URL) is not configured in the server environment."
        logger.error(error_msg)
        return {"error": error_msg}

    logger.info(f"Received search request for query: '{query}' using base URL: {base_url}")

    search_endpoint = f"{base_url.rstrip('/')}/search"
    search_headers = headers if headers else DEFAULT_HEADERS

    params = {
        'q': query,
        'categories': categories,
        'language': language,
        'safesearch': str(safesearch),
        'format': 'json',
    }

    try:
        logger.info(f"Sending request to SearXNG: {search_endpoint} with params: {params}")
        response = requests.get(
            search_endpoint,
            params=params,
            headers=search_headers,
            timeout=timeout
        )
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        search_data = response.json()
        logger.info(f"Received {len(search_data.get('results', []))} results and {len(search_data.get('infoboxes', []))} infoboxes from SearXNG.")

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

        # Process 'infoboxes' (ensure we don't exceed max_results if combined)
        if 'infoboxes' in search_data and isinstance(search_data['infoboxes'], list):
             for box in search_data['infoboxes']:
                 if max_results is not None and len(formatted_results) >= max_results:
                     break # Stop if we've reached max_results

                 title = f"Infobox ({box.get('engine', 'Source Unknown')})"
                 content = box.get('content', 'No content.')
                 infobox_url = None
                 if box.get('urls') and isinstance(box['urls'], list) and box['urls']:
                     # Try to get a URL from the list
                     infobox_url = box['urls'][0].get('url')
                 # Fallback URL if 'urls' is missing or empty
                 infobox_url = infobox_url or box.get('infobox_url', 'No URL')

                 formatted_results.append({
                     "title": title,
                     "url": infobox_url,
                     "snippet": content
                 })


        if not formatted_results:
            logger.warning(f"No search results found for query: '{query}'")
            return {"results": []}

        logger.info(f"Formatted {len(formatted_results)} results for query: '{query}'")
        # Limit again just in case infobox processing exceeded slightly
        if max_results is not None:
            formatted_results = formatted_results[:max_results]

        return {"results": formatted_results}

    except requests.exceptions.RequestException as e:
        error_msg = f"Error during SearXNG request: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    except json.JSONDecodeError as e:
        error_msg = f"Error decoding JSON response from SearXNG: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        # Catch any other unexpected errors
        error_msg = f"Unexpected error during search: {str(e)}"
        logger.exception(error_msg) # Log exception traceback
        return {"error": error_msg}

# --- Run the MCP Server ---
if __name__ == "__main__":
    logger.info("Starting SearXNG MCP server...")
    # This will start the server using the stdio transport when executed directly
    mcp.run()