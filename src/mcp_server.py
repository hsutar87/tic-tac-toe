from fastmcp import FastMCP
from typing import List, Dict, Any
from .vector_store import VectorStoreManager

mcp = FastMCP("Silly-Knowledge-Server")
store = VectorStoreManager()


@mcp.tool()
def say_hello(name: str) -> str:
    """
    A simple tool that greets the user by name.

    Parameters:
        name (str): The name of the user to greet.

    Returns:
        str: A greeting message.
    """
    return f"Hello, {name}! How can I assist you today?"
    

@mcp.tool()
def search_local_docs(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search the local OpenSearch-backed vector store for the given query.

    Parameters:
        query (str): The text query to search for.
        limit (int): Maximum number of results to return (default 5).

    Returns:
        List[Dict[str, Any]]: A list of result objects containing:
            - content (str): The document chunk text
            - score (float): Relevance score from OpenSearch
            - source (str): Source filename (e.g., "file.pdf")
    """
    results = store.search(query, limit=limit)
    return results or []


@mcp.tool()
def list_available_files() -> List[str]:
    """
    List all files currently indexed in the vector store.

    Returns:
        List[str]: Sorted list of filenames indexed in OpenSearch.
    """
    body = {
        "size": 0,
        "aggs": {
            "unique_sources": {
                "terms": {
                    "field": "source.keyword",
                    "size": 1000
                }
            }
        }
    }

    response = store.client.search(index=store.index_name, body=body)
    buckets = response.get("aggregations", {}).get("unique_sources", {}).get("buckets", [])
    sources = [b["key"] for b in buckets]

    return sorted(sources)


if __name__ == "__main__":
    # Start SSE server on port 8000
    mcp.run(transport="sse", port=8000)
