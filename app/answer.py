import os
import time
import logging
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import anthropic
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_content: str) -> str:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

class ClaudeProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = "claude-3-5-sonnet-20240620"

    def get_name(self) -> str:
        return "claude"

    def generate(self, system_prompt: str, user_content: str) -> str:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}]
        )
        return message.content[0].text

class GroqProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.model = "llama-3.3-70b-versatile"

    def get_name(self) -> str:
        return "groq-llama3"

    def generate(self, system_prompt: str, user_content: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            max_tokens=1024
        )
        return completion.choices[0].message.content

class AnswerGenerator:
    def __init__(self):
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        
        self.providers: List[LLMProvider] = []
        
        if anthropic_key and anthropic_key != "your_anthropic_api_key":
            self.providers.append(ClaudeProvider(anthropic_key))
        
        if groq_key and groq_key != "your_groq_api_key":
            self.providers.append(GroqProvider(groq_key))
            
        if not self.providers:
            # For qualification demo purposes, we will initialize with dummy keys 
            # if real ones aren't provided yet to avoid init errors
            logger.warning("No valid API keys found in .env. Initializing with placeholders.")

    def generate_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate an answer based on retrieved context with citations and failover support.
        """
        if not context_chunks:
            return {
                "answer": "I'm sorry, I couldn't find any relevant information in the documentation to answer your question.",
                "citations": [],
                "provider": "none"
            }

        context_text = ""
        citations = []
        for i, chunk in enumerate(context_chunks):
            context_text += f"\n---\nSource [{i+1}]: {chunk['source_url']}\nContent: {chunk['content']}\n"
            citations.append({
                "id": i + 1,
                "url": chunk['source_url'],
                "title": chunk['title']
            })

        system_prompt = """You are a documentation assistant. 
Answer the user's question ONLY using the provided context chunks.
If the answer is not contained in the context, clearly state that you don't have enough information.
Always cite your sources using the [n] format where n corresponds to the source index provided in the context.
Keep your answer professional, concise, and technically accurate."""

        user_content = f"Question: {question}\n\nContext:\n{context_text}"

        # Failover logic: loop through providers
        last_error = None
        for provider in self.providers:
            try:
                logger.info(f"Attempting generation with provider: {provider.get_name()}")
                answer_text = provider.generate(system_prompt, user_content)
                
                return {
                    "answer": answer_text,
                    "citations": citations,
                    "provider": provider.get_name()
                }
            except Exception as e:
                logger.error(f"Provider {provider.get_name()} failed: {str(e)}")
                last_error = e
                # Fall through to next provider

        # If all providers fail or no providers available
        error_msg = f"All providers failed. Last error: {str(last_error)}" if last_error else "No LLM providers configured."
        return {
            "answer": f"An error occurred while generating the answer: {error_msg}",
            "citations": [],
            "provider": "failed"
        }

if __name__ == "__main__":
    generator = AnswerGenerator()
    # Dummy test
    print(generator.generate_answer("What is Claude?", [{"content": "Claude is an AI...", "source_url": "...", "title": "..."}]))
