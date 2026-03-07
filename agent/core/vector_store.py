"""
Simple Vector Store for Persistent Research Memory
A lightweight fallback that uses numpy for cosine similarity
Bypasses ChromaDB/Pydantic v1 issues on Python 3.14
"""
import logging
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = "./data/vectorstore"):
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.storage_file = os.path.join(persist_directory, "simple_store.json")
        self.embeddings_file = os.path.join(persist_directory, "embeddings.npy")
        
        # Load or initialize model
        try:
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("SentenceTransformer model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None

        self.data: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self._load()

    def _load(self):
        """Load data from disk"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r', encoding='utf-8') as f:
                    self.data = json.load(f)
                if os.path.exists(self.embeddings_file):
                    self.embeddings = np.load(self.embeddings_file)
                logger.info(f"Loaded {len(self.data)} items from {self.storage_file}")
            except Exception as e:
                logger.error(f"Error loading vector store: {e}")
                self.data = []
                self.embeddings = None

    def _save(self):
        """Save data to disk"""
        try:
            with open(self.storage_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2)
            if self.embeddings is not None:
                np.save(self.embeddings_file, self.embeddings)
            logger.info(f"Saved {len(self.data)} items to {self.storage_file}")
        except Exception as e:
            logger.error(f"Error saving vector store: {e}")

    def add_facts(self, facts: List[Dict[str, Any]], session_id: str):
        """Add verified facts to the vector store with guards"""
        if not facts:
            return
            
        if self.model is None:
            logger.warning("VectorStore: Model not loaded, skipping semantic indexing but saving raw data.")
            self.data.extend([{
                "id": f"{session_id}_{len(self.data) + i}",
                "text": f.get("fact", ""),
                "session_id": session_id,
                "sources": f.get("sources", []),
                "timestamp": str(datetime.now().isoformat())
            } for i, f in enumerate(facts)])
            self._save()
            return

        new_texts = []
        new_items = []
        for fact_data in facts:
            fact_text = fact_data.get("fact", "")
            if not fact_text:
                continue
            
            new_texts.append(fact_text)
            new_items.append({
                "id": f"{session_id}_{len(self.data) + len(new_items)}",
                "text": fact_text,
                "session_id": session_id,
                "confidence": fact_data.get("confidence", 0.0),
                "sources": fact_data.get("sources", []),
                "timestamp": str(os.path.getmtime(self.persist_directory))
            })

        if new_texts:
            new_embeddings = self.model.encode(new_texts)
            
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
            self.data.extend(new_items)
            self._save()
            logger.info(f"Added {len(new_texts)} facts to VectorStore")

    def query_related_knowledge(self, query: str, n_results: int = 5) -> List[str]:
        """Retrieve semantically related facts using cosine similarity"""
        if not self.data or self.embeddings is None or self.model is None:
            return []

        try:
            query_embedding = self.model.encode([query])[0]
            
            # Use numpy for cosine similarity
            # (embeddings / norm) dot (query / norm)
            norms = np.linalg.norm(self.embeddings, axis=1)
            query_norm = np.linalg.norm(query_embedding)
            
            if query_norm == 0:
                return []
                
            similarities = np.dot(self.embeddings, query_embedding) / (norms * query_norm + 1e-9)
            
            # Get top indices
            top_indices = np.argsort(similarities)[::-1][:n_results]
            
            results = [self.data[i]["text"] for i in top_indices if similarities[i] > 0.3]
            return results
        except Exception as e:
            logger.error(f"Error querying VectorStore: {e}")
            # Fallback: simple keyword match if semantic search fails
            keywords = query.lower().split()
            return [d["text"] for d in self.data[:n_results] if any(k in d["text"].lower() for k in keywords)]

    def get_session_memory(self, session_id: str) -> List[str]:
        """Retrieve all facts associated with a specific session"""
        return [item["text"] for item in self.data if item.get("session_id") == session_id]
