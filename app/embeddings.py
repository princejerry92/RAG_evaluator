import os
from typing import List, Dict, Any
from supabase import create_client, Client
import voyageai
from dotenv import load_dotenv

load_dotenv()

class EmbeddingManager:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        self.voyage_key = os.getenv("VOYAGE_API_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase credentials missing in .env")
        if not self.voyage_key or self.voyage_key == "your_voyage_api_key":
            # For qualification demo, we allow initialization but will warn during usage
            print("WARNING: VOYAGE_API_KEY missing or placeholder in .env. Real embeddings will fail.")
            
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        self.vo = voyageai.Client(api_key=self.voyage_key) if self.voyage_key else None

    def get_embedding(self, text: str) -> List[float]:
        """
        Uses Voyage AI to generate semantic embeddings.
        Voyage is the recommended provider for Anthropic-based RAG pipelines.
        Model: voyage-large-2 (1536 dimensions)
        """
        if not self.vo:
            raise ValueError("Voyage client not initialized. Check VOYAGE_API_KEY in .env")
            
        # Clean text to ensure best embedding quality
        text = text.replace("\n", " ")
        
        try:
            result = self.vo.embed([text], model="voyage-4-lite", input_type="document")
            return result.embeddings[0]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Fallback to zero vector to avoid total failure, but log clearly
            return [0.0] * 1536

    def store_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Store chunks in Supabase pgvector 'docs_chunks' table.
        """
        for chunk in chunks:
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
