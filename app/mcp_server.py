import os
import json
from mcp.server.fastmcp import FastMCP
from app.retrieve import Retriever
from app.answer import AnswerGenerator

# Initialize FastMCP server
mcp = FastMCP("Docs Search Server")

retriever = Retriever()
generator = AnswerGenerator()

@mcp.tool()
def search_docs(query: str) -> str:
    """
    Search Anthropic documentation for the given query and return an answer with citations.
    """
    try:
        chunks = retriever.retrieve_context(query, k=5)
        result = generator.generate_answer(query, chunks)
        
        # Format output for MCP
        citations_text = "\n".join([f"- [{c['id']}] {c['title']}: {c['url']}" for c in result['citations']])
        return f"Answer: {result['answer']}\n\nCitations:\n{citations_text}"
    except Exception as e:
        return f"Error searching documentation: {str(e)}"

if __name__ == "__main__":
    mcp.run()
