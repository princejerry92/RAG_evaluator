import re
import time
from typing import List, Dict, Any, Optional
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from app.embeddings import EmbeddingManager

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

class RecursiveCharacterChunker:
    def __init__(self, chunk_size: int = 3000, overlap: int = 500):
        # 600-800 tokens is roughly 2400-3200 characters (assuming 4 chars/token)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        return self._split(text, self.chunk_size)

    def _split(self, text: str, chunk_size: int) -> List[str]:
        if len(text) <= chunk_size:
            return [text]
        
        separator = self.separators[-1]
        for s in self.separators:
            if s == "":
                separator = s
                break
            if s in text:
                separator = s
                break
                
        splits = text.split(separator) if separator else list(text)
        
        chunks = []
        current_chunk = []
        current_len = 0
        
        for split in splits:
            split_len = len(split) + (len(separator) if separator else 0)
            if current_len + split_len > chunk_size and current_chunk:
                chunks.append(separator.join(current_chunk))
                # rough overlap
                while current_len > self.overlap and len(current_chunk) > 1:
                    popped = current_chunk.pop(0)
                    current_len -= (len(popped) + len(separator))
            current_chunk.append(split)
            current_len += split_len
            
        if current_chunk:
            chunks.append(separator.join(current_chunk))
            
        return chunks

def is_valid_content(title: str, text: str) -> bool:
    """
    Validates if the scraped content is useful and not a placeholder or shell.
    """
    if len(text) < 300:
        print(f"Skipping '{title}': Content too short ({len(text)} chars)")
        return False
        
    if "Loading..." in text and text.count("Loading...") > 3:
        print(f"Skipping '{title}': Appears to be a loading state")
        return False
        
    # Basic heuristic for meaningful paragraph text
    if text.count(".") < 5:
        print(f"Skipping '{title}': Lacks meaningful sentence structure")
        return False
        
    return True

def scrape_page(page, url: str) -> Optional[Dict[str, Any]]:
    try:
        print(f"Navigating to: {url}")
        # 'domcontentloaded' is faster and more reliable than 'networkidle' for static site docs
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        
        # Give a small buffer for any hydration/late rendering
        page.wait_for_timeout(1000)
        
        content = page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract title
        title = page.title() or url
        
        # Remove noise
        for element in soup(["nav", "footer", "header", "script", "style", "aside", "menu"]):
            element.decompose()
            
        # Prefer main tags
        main_content = soup.find('main') or soup.find('article') or soup.find('section') or soup.body
        
        if not main_content:
            print(f"No main content found for {url}")
            return None
            
        # Get cleaned text
        full_text = main_content.get_text(separator='\n\n', strip=True)
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        
        if not is_valid_content(title, full_text):
            return None
            
        print(f"Successfully extracted {len(full_text)} characters from '{title}'")
        if len(full_text) > 0:
            print(f"DEBUG: First 500 chars:\n{full_text[:500]}...")
        
        return {
            "title": title,
            "source_url": url,
            "content": full_text
        }
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def fetch_doc_urls() -> List[str]:
    """
    Returns a list of stable documentation URLs to scrape.
    Switching to FastAPI docs as requested.
    """
    return [
        "https://fastapi.tiangolo.com/",
        "https://fastapi.tiangolo.com/features/",
        "https://fastapi.tiangolo.com/tutorial/first-steps/",
        "https://fastapi.tiangolo.com/tutorial/path-params/",
        "https://fastapi.tiangolo.com/tutorial/query-params/",
        "https://fastapi.tiangolo.com/tutorial/body/",
        "https://fastapi.tiangolo.com/advanced/custom-response/",
        "https://fastapi.tiangolo.com/advanced/middleware/",
    ]

def process_docs(max_pages: int = 20) -> List[Dict[str, Any]]:
    urls = fetch_doc_urls()
    if max_pages:
        urls = urls[:max_pages]
        
    chunker = RecursiveCharacterChunker()
    all_chunks = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
        page = context.new_page()
        
        for i, url in enumerate(urls):
            print(f"[{i+1}/{len(urls)}] Processing: {url}")
            page_data = scrape_page(page, url)
            
            if not page_data:
                continue
                
            text_chunks = chunker.split_text(page_data["content"])
            
            for chunk in text_chunks:
                if len(chunk.strip()) > 50:
                    all_chunks.append({
                        "title": page_data["title"],
                        "source_url": page_data["source_url"],
                        "content": chunk.strip()
                    })
            
            print(f"Extracted {len(text_chunks)} chunks from '{page_data['title']}'")
        
        browser.close()
            
    print(f"Total chunks generated: {len(all_chunks)}")
    return all_chunks

if __name__ == "__main__":
    # Test ingestion locally
    chunks = process_docs(max_pages=5)
    
    if chunks:
        print(f"Connecting to Supabase to store {len(chunks)} chunks...")
        manager = EmbeddingManager()
        manager.store_chunks(chunks)
        print("Successfully stored chunks in Supabase!")
    else:
        print("No chunks were generated.")
