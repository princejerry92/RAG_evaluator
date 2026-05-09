import os
import logging
from typing import List, Dict, Any
from supabase import create_client, Client
from dotenv import load_dotenv
from app.embeddings import EmbeddingManager

load_dotenv()

logger = logging.getLogger(__name__)

class Retriever:
    def __init__(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
        self.embedding_manager = EmbeddingManager()

    def retrieve_context(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve top-k chunks from Supabase using vector similarity,
        with a text-search fallback if vector search returns empty.
        """
        query_embedding = self.embedding_manager.get_embedding(question)
        
        # Attempt 1: Vector similarity via RPC
        try:
            rpc_params = {
                "query_embedding": query_embedding,
                "match_threshold": 0.1,  # Tuned for Voyage semantic embeddings
                "match_count": k,
            }
            response = self.supabase.rpc("match_docs", rpc_params).execute()
            logger.info(f"Vector search returned {len(response.data)} results")
            
            if response.data:
                return response.data
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
        
        # Attempt 2: Fallback to simple text search on the content column
        logger.info("Falling back to text-based search")
        try:
            response = (
                self.supabase.table("docs_chunks")
                .select("id, content, source_url, title")
                .ilike("content", f"%{question.split()[0]}%")  # Search for the first keyword
                .limit(k)
                .execute()
            )
            logger.info(f"Text fallback returned {len(response.data)} results")
            return response.data
        except Exception as e:
            logger.error(f"Text fallback failed: {e}")
            return []

if __name__ == "__main__":
    retriever = Retriever()
    results = retriever.retrieve_context("How do I use Claude?")
    print(results)
