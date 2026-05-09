import os
from typing import List, Dict, Any, Optional
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field
from app.retrieve import Retriever
from app.answer import AnswerGenerator

# Initialize FastMCP server
mcp = FastMCP("FastAPI Docs Assistant")

retriever = Retriever()
generator = AnswerGenerator()

class SearchInput(BaseModel):
    query: str = Field(..., description="The question or search query for the documentation")
    k: int = Field(5, description="Number of context chunks to retrieve")

class Citation(BaseModel):
    id: int
    title: str
    url: str

class SearchResponse(BaseModel):
    answer: str
    citations: List[Citation]
    provider: str

@mcp.tool()
def search_docs(query: str, k: int = 5) -> str:
    """
    Search the FastAPI documentation and generate an answer using RAG.
    Returns a comprehensive answer with inline citations and a source list.
    """
    try:
        chunks = retriever.retrieve_context(query, k=k)
        result = generator.generate_answer(query, chunks)
        
        # Format output for clear display in Claude
        citations_text = "\n".join([f"[{c['id']}] {c['title']}: {c['url']}" for c in result['citations']])
        return f"### Answer\n{result['answer']}\n\n### Sources\n{citations_text}\n\n*Provider: {result['provider']}*"
    except Exception as e:
        return f"Error searching documentation: {str(e)}"

@mcp.tool()
def format_citations(citations: List[Dict[str, Any]]) -> str:
    """
    Utility tool to format a list of raw citation objects into a clean Markdown list.
    """
    if not citations:
        return "No citations provided."
    
    formatted = "### Documentation Sources\n"
    for c in citations:
        title = c.get('title', 'Untitled')
        url = c.get('url', '#')
        cid = c.get('id', '?')
        formatted += f"- **[{cid}]** [{title}]({url})\n"
    return formatted

if __name__ == "__main__":
    mcp.run()
