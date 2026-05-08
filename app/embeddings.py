import os
from typing import List, Dict, Any
from supabase import create_client, Client
import math
import hashlib
import anthropic
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class EmbeddingManager:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        self.anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase credentials missing in .env")
        if not self.anthropic_key:
            raise ValueError("Anthropic API key missing in .env")
            
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        self.client = anthropic.Anthropic(api_key=self.anthropic_key)

    def get_embedding(self, text: str) -> List[float]:
        """
        Since Claude doesn't have a direct public embedding endpoint like OpenAI (Voyage is often recommended for Claude),
        for this qualification project we will assume the user might want Voyage or standardizing on a common model.
        However, the prompt says 'Tech Stack: Anthropic Claude API'. 
        Anthropic actually recommends Voyage AI for embeddings.
        If the user didn't specify voyage-ai in dependencies, I'll use a placeholder or use a simple library if needed.
        Wait, Anthropic has a partnership with Voyage. 
        Let's check if the requirements.txt had something for embeddings. It had `numpy`.
        
        Actually, for a real production RAG with Claude, Voyage is the standard.
        I will implement a placeholder for now or use a basic sentence-transformer if the user prefers, 
        but I'll stick to the idea of 'Anthropic Claude API' being the core.
        
        CORRECTION: I'll use a mock/simple embedding if voyage is not installed, or ask if they want voyage.
        Actually, I'll just use a generic 'embedding_placeholder' for the logic phase and note it.
        """
        # Placeholder: In a real app, you'd call Voyage AI or OpenAI
        # For this demo, let's pretend we have an embedding function.
        # I'll use a deterministic hash-based "embedding" for local testing if needed,
        # but the code will be structured for real API usage.
        
        # In a real scenario:
        # response = voyage_client.embed([text], model="voyage-large-2")
        # return response.embeddings[0]
        
        # Deterministic dummy embedding for logic validation (1536 dims)
        # Using a non-zero vector avoids NaN cosine distance issues in pgvector
        hashed = hashlib.md5(text.encode('utf-8')).hexdigest()
        seed = int(hashed, 16) % 10000
        np.random.seed(seed)
        vec = np.random.rand(1536) - 0.5
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist()

    def store_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Store chunks in Supabase pgvector 'docs_chunks' table.
        """
        for chunk in chunks:
            # In production, you'd batch these
            embedding = self.get_embedding(chunk["content"])
            data = {
                "content": chunk["content"],
                "source_url": chunk["source_url"],
                "title": chunk["title"],
                "embedding": embedding
            }
            try:
                self.supabase.table("docs_chunks").insert(data).execute()
            except Exception as e:
                print(f"Error storing chunk: {e}")

    def setup_database(self):
        """
        SQL to run in Supabase SQL Editor:
        
        create extension if not exists vector;
        
        create table docs_chunks (
          id bigserial primary key,
          content text,
          source_url text,
          title text,
          embedding vector(1536)
        );
        """
        print("Please ensure the pgvector extension and docs_chunks table are set up in Supabase.")

if __name__ == "__main__":
    manager = EmbeddingManager()
    manager.setup_database()
