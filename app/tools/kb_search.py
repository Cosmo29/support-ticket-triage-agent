"""
Support Knowledge Base Search Tool.
Retrieves grounded runbooks, policies, and troubleshooting articles using vector/TF-IDF similarity.
"""

import os
import glob
import time
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from app.models.ticket import KBCitation, ToolCallRecord
from app.core.logging import logger

KB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/knowledge_base"))

class KBSearchEngine:
    def __init__(self, kb_directory: str = KB_DIR):
        self.kb_directory = kb_directory
        self.documents: List[dict] = []
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = None
        self._index_articles()

    def _index_articles(self):
        """Indexes all markdown articles in the knowledge base."""
        self.documents = []
        md_files = glob.glob(os.path.join(self.kb_directory, "*.md"))
        
        for filepath in md_files:
            filename = os.path.basename(filepath)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                # Extract Title
                title = filename.replace(".md", "").replace("_", " ").title()
                for line in content.splitlines():
                    if line.startswith("# "):
                        title = line.replace("# ", "").strip()
                        break
                        
                # Chunk document by H2 sections
                sections = content.split("## ")
                for i, sec in enumerate(sections):
                    chunk_text = sec.strip()
                    if not chunk_text:
                        continue
                    sec_title = title if i == 0 else f"{title} - {chunk_text.splitlines()[0]}"
                    self.documents.append({
                        "filename": filename,
                        "title": sec_title,
                        "content": chunk_text
                    })
            except Exception as e:
                logger.error(f"Error reading KB file {filepath}: {e}")

        if self.documents:
            corpus = [f"{doc['title']} {doc['content']}" for doc in self.documents]
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
            logger.info(f"Knowledge Base indexed with {len(self.documents)} sections.")

    def search(self, query: str, top_k: int = 3) -> Tuple[List[KBCitation], ToolCallRecord]:
        """Searches indexed KB chunks against ticket query text."""
        start_time = time.time()
        citations: List[KBCitation] = []
        
        if not self.documents or not query.strip() or self.tfidf_matrix is None:
            latency_ms = round((time.time() - start_time) * 1000, 2)
            record = ToolCallRecord(
                tool_name="kb_search",
                input_params={"query": query[:60], "top_k": top_k},
                output_summary="0 articles found (empty index or query)",
                latency_ms=latency_ms
            )
            return citations, record

        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            top_indices = similarities.argsort()[::-1][:top_k]
            
            for idx in top_indices:
                score = float(similarities[idx])
                if score > 0.05:  # Relevance threshold
                    doc = self.documents[idx]
                    snippet = doc["content"][:300].strip() + ("..." if len(doc["content"]) > 300 else "")
                    citations.append(KBCitation(
                        article_title=doc["title"],
                        filename=doc["filename"],
                        relevance_score=round(score, 3),
                        snippet=snippet
                    ))
        except Exception as e:
            logger.error(f"KB search execution error: {e}")

        latency_ms = round((time.time() - start_time) * 1000, 2)
        summary = f"Found {len(citations)} relevant articles: " + ", ".join([c.article_title for c in citations]) if citations else "No matching KB articles found"
        
        record = ToolCallRecord(
            tool_name="kb_search",
            input_params={"query": query[:80], "top_k": top_k},
            output_summary=summary,
            latency_ms=latency_ms
        )
        return citations, record

# Global KB instance
kb_engine = KBSearchEngine()

def kb_search(query: str, top_k: int = 2) -> Tuple[List[KBCitation], ToolCallRecord]:
    return kb_engine.search(query, top_k)
