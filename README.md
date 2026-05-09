# Documentation AI Assistant: Production-Grade RAG with LLM-as-Judge Evaluation

A sophisticated Retrieval-Augmented Generation (RAG) system designed for technical documentation Q&A, featuring a robust ingestion pipeline, multi-provider LLM failover, and a comprehensive evaluation dashboard.

**Qualification Choice: Option C - MCP Server Implementation**

## 🚀 Key Features

### 1. Robust Ingestion Pipeline (Playwright)
- **Dynamic Rendering**: Uses Playwright with headless Chromium to fetch fully rendered JavaScript-enabled documentation pages.
- **Smart Extraction**: Targets `main`, `article`, and `section` tags to isolate meaningful content.
- **Content Validation**: Heuristic-based validation to skip shallow pages or loading states.

### 2. Intelligent Q&A & Embeddings
- **Voyage AI Embeddings**: Uses `voyage-large-2` (1536 dims), the gold standard for Anthropic-based pipelines.
- **Vector Retrieval**: Powered by Supabase `pgvector` with a text-search fallback for maximum reliability.
- **Multi-Provider Failover**: Primarily uses **Anthropic Claude 3.5 Sonnet** and automatically fails over to **Groq (Llama 3.3 70B)** if it encounters API errors or credit limits.
- **Citations**: Automatically extracts and displays source URLs for every statement.

### 3. Deep Integration: MCP Server (Option C)
This project includes a full Model Context Protocol (MCP) server integration, allowing any Claude client (like Claude Desktop) to use the documentation as a live tool.
- **Tools Exposed**: `search_docs` (RAG search) and `format_citations` (Source formatting).
- **Architecture**: Built with `fastmcp` for high performance and clear schema definitions.

### 4. Production-Grade Evaluation Dashboard
- **LLM-as-Judge**: Automatically scores every response using a separate LLM judge.
- **Metrics**: Precision@3, Citation Accuracy, Answer Relevance (1-5), and Faithfulness (1-5).
- **Interactive UI**: A dedicated dashboard tab with metric cards and a 12-question test suite.

---

## ⚙️ Setup Instructions

### 1. Environment Configuration
Create a `.env` file:
```env
ANTHROPIC_API_KEY=...
GROQ_API_KEY=...
VOYAGE_API_KEY=...
SUPABASE_URL=...
SUPABASE_SERVICE_KEY=...
```

### 2. Installation
```bash
pip install -r requirements.txt
playwright install chromium
```

### 3. Running the App
- **Ingestion**: `python -m app.ingest`
- **FastAPI**: `uvicorn app.api:app --reload`
- **MCP Server**: `python -m app.mcp_server`

### 4. MCP Demo Instructions
To connect this to Claude Desktop, add the following to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "fastapi-docs": {
      "command": "python",
      "args": ["-m", "app.mcp_server"],
      "env": {
        "ANTHROPIC_API_KEY": "...",
        "VOYAGE_API_KEY": "...",
        "SUPABASE_URL": "...",
        "SUPABASE_SERVICE_KEY": "..."
      }
    }
  }
}
```

---

## 🤝 Project Background
This project was built as a qualification showcase for a Senior AI Builder role, emphasizing architectural cleanliness, robust error handling, and quantitative evaluation of AI responses.
