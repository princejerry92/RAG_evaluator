from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from app.retrieve import Retriever
from app.answer import AnswerGenerator
import os
import uvicorn

app = FastAPI(title="Anthropic Docs RAG API")

retriever = Retriever()
generator = AnswerGenerator()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    provider: str

@app.get("/")
async def get_index():
    return FileResponse("app/index.html")

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/ask", response_model=AnswerResponse)
async def ask(request: QuestionRequest):
    try:
        chunks = retriever.retrieve_context(request.question)
        result = generator.generate_answer(request.question, chunks)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/evaluate")
async def evaluate():
    from app.evaluate import Evaluator
    evaluator = Evaluator()
    results = evaluator.run_evaluation()
    return results

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
