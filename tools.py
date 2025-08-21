import asyncio
import os
import aiohttp
from bs4 import BeautifulSoup

brave_api_key = os.getenv("BRAVE_API_KEY")


async def search(
    query: str, count: int = 10, country: str = "us", search_lang: str = "en"
) -> str:
    """
    Search the web using Brave Search API.

    Args:
        query: The search query to execute
        count: Number of search results to return (1-20)
        country: Country code for localized results
        search_lang: Language for search results

    Returns:
        str: YAML-formatted search results including query and web results
    """
    print("Running search on: ", query)

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "X-Subscription-Token": brave_api_key,
                },
                params={
                    "q": query,
                    "count": count,
                    "country": country,
                    "search_lang": search_lang,
                    "result_filter": "web",
                },
            ) as response:
                # Handle HTTP error status codes
                if response.status >= 400:
                    if response.status == 429:  # Rate limited
                        await asyncio.sleep(2)
                        raise ValueError(
                            f"Rate limited (429), retrying search for: {query}"
                        )
                    elif response.status >= 500:  # Server errors
                        await asyncio.sleep(3)
                        raise ValueError(
                            f"Server error ({response.status}), retrying search for: {query}"
                        )

                json_data = await response.json()

                # Extract web results if they exist
                if "web" in json_data and "results" in json_data["web"]:
                    web_results = json_data["web"]["results"]
                    total_count = len(web_results)

                    # Format results as YAML
                    results_yaml = "results:\n"
                    for result in web_results:
                        title = result.get("title", "").replace('"', '\\"')
                        url = result.get("url", "")
                        description = result.get("description", "").replace('"', '\\"')
                        age = result.get("age", "")

                        results_yaml += f"""  - title: "{title}"
    url: "{url}"
    description: "{description}"
    date: "{age}"
"""

                return f"""query: "{query}"
total_count: {total_count}
{results_yaml}"""

    except aiohttp.ClientError as e:
        # Network/connection errors - retry with delay
        await asyncio.sleep(2)
        raise ValueError(f"Network error during search, retrying: {str(e)}")
    except Exception as e:
        # For other exceptions, return error without retry
        error_msg = str(e).replace('"', '\\"')
        return f"""query: "{query}"
error: "{error_msg}\""""


async def fetch(url: str, timeout: int = 30, headers: dict | None = None) -> str:
    """
    Fetch content from a URL.

    Args:
        url: The URL to fetch content from
        timeout: Request timeout in seconds
        headers: Optional headers to include in the request

    Returns:
        str: YAML-formatted response data including status, content, and URL
    """
    print(f"Fetching URL: {url}")
    try:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as session:
            async with session.get(url, headers=headers or {}) as response:
                html_content = await response.text()
                text_content = extract_text_content(html_content)
                clean_content = text_content.replace('"', '\\"').replace("\n", "\\n")

                return f"""url: "{str(response.url)}"
content: "{clean_content}\""""
    except Exception as e:
        error_msg = str(e).replace('"', '\\"')
        print(f"Error fetching {url}: {str(e)}")
        return f"""url: "{url}"
status_code: "error"
error: "{error_msg}\""""


def extract_text_content(html_content: str) -> str:
    """Extract clean readable text from HTML content using BeautifulSoup."""
    soup = BeautifulSoup(html_content, "html.parser")

    # Remove unwanted elements
    for element in soup.find_all(
        ["script", "style", "nav", "header", "footer", "aside", "menu"]
    ):
        element.decompose()

    # Extract clean text with proper spacing
    text = soup.get_text(separator=" ", strip=True)

    return text
