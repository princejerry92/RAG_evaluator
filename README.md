# Documentation AI Assistant: Production-Grade RAG with LLM-as-Judge Evaluation

A sophisticated Retrieval-Augmented Generation (RAG) system designed for technical documentation Q&A, featuring a robust ingestion pipeline, multi-provider LLM failover, and a comprehensive evaluation dashboard.

## 🚀 Key Features

### 1. Robust Ingestion Pipeline (Playwright)
- **Dynamic Rendering**: Uses Playwright with headless Chromium to fetch fully rendered JavaScript-enabled documentation pages.
- **Smart Extraction**: Targets `main`, `article`, and `section` tags to isolate meaningful content while stripping navigation headers, footers, and scripts.
- **Content Validation**: Heuristic-based validation to skip shallow pages, loading states, or shell-only content.
- **Stable Source**: Pre-configured to index high-quality documentation (e.g., FastAPI tutorial).

### 2. Intelligent Question Answering
- **Vector Retrieval**: Powered by Supabase `pgvector` for high-performance semantic search.
- **Multi-Provider Failover**: Implements a provider-abstraction layer. It primarily uses **Anthropic Claude 3.5 Sonnet** and automatically fails over to **Groq (Llama 3.3 70B)** if it encounters API errors, timeouts, or credit limits.
- **Citations**: Automatically extracts and displays source URLs for every statement made by the LLM.

### 3. Production-Grade Evaluation Dashboard
- **LLM-as-Judge**: Automatically scores every response using a separate LLM judge (configured via the `Evaluator` class).
- **Core Metrics**:
  - **Precision@3**: Measures retrieval relevance.
  - **Citation Accuracy**: Validates if citations correctly point to the used documents.
  - **Answer Relevance**: LLM-judge score (1-5) for answer sufficiency.
  - **Faithfulness**: LLM-judge score (1-5) for groundedness and absence of hallucinations.
- **Interactive UI**: A dedicated "Evaluation Dashboard" tab with metric cards, score bars, and a detailed results table.

---

## 🛠 Tech Stack

- **Backend**: FastAPI (Python 3.11+)
- **Vector DB**: Supabase (PostgreSQL with `pgvector`)
- **LLMs**: Anthropic Claude API & Groq API
- **Scraper**: Playwright & BeautifulSoup4
- **Frontend**: Single-page Tailwind CSS (Modern Dark Mode)

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.11+
- A Supabase project with the `vector` extension enabled.

### 2. Environment Configuration
Create a `.env` file in the root directory:
```env
ANTHROPIC_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_service_role_key
```

### 3. Database Setup
Run the following SQL in the Supabase SQL Editor to create the schema and the search function:
```sql
create extension if not exists vector;

create table docs_chunks (
  id bigserial primary key,
  content text,
  source_url text,
  title text,
  embedding vector(1536)
);

create or replace function match_docs (
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
returns table (
  id bigint,
  content text,
  source_url text,
  title text,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    docs_chunks.id,
    docs_chunks.content,
    docs_chunks.source_url,
    docs_chunks.title,
    1 - (docs_chunks.embedding <=> query_embedding) as similarity
  from docs_chunks
  where 1 - (docs_chunks.embedding <=> query_embedding) > match_threshold
  order by docs_chunks.embedding <=> query_embedding
  limit match_count;
end;
$$;
```

### 4. Installation
```bash
pip install -r requirements.txt
playwright install chromium
```

### 5. Ingestion
Initialize the vector database with documentation:
```bash
python -m app.ingest
```

### 6. Run the App
```bash
uvicorn app.api:app --reload
```
Navigate to `http://127.0.0.1:8000` to access the chat and evaluation interfaces.

---

## 📊 Evaluation Results
The system includes a pre-defined test suite in `tests/eval_questions.json`. Open the **Evaluation Dashboard** in the browser to run these tests and view live performance metrics.

---

## 🤝 Project Background
This project was built as a qualification showcase for a Senior AI Builder role, emphasizing architectural cleanliness, robust error handling, and quantitative evaluation of AI responses.
