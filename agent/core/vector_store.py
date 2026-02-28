"""
Vector Store for Persistent Research Memory
Uses ChromaDB for local semantic storage and retrieval
"""
import chromadb
from chromadb.utils import embedding_functions
import logging
import os
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, persist_directory: str = "./data/vectorstore"):
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Use a lightweight sentence-transformer for local embeddings
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        self.collection = self.client.get_or_create_collection(
            name="research_memory",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"VectorStore initialized at {persist_directory}")

    def add_facts(self, facts: List[Dict[str, Any]], session_id: str):
        """Add verified facts to the vector store"""
        if not facts:
            return

        ids = []
        documents = []
        metadatas = []

        for i, fact_data in enumerate(facts):
            fact_text = fact_data.get("fact", "")
            if not fact_text:
                continue
            
            # Create a unique ID for the fact
            fact_id = f"{session_id}_{i}"
            ids.append(fact_id)
            documents.append(fact_text)
            metadatas.append({
                "session_id": session_id,
                "confidence": fact_data.get("confidence", 0.0),
                "sources": ",".join(fact_data.get("sources", [])),
                "timestamp": str(os.path.getmtime(os.getcwd())) # placeholder for time
            })

        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Added {len(ids)} facts to VectorStore")

    def query_related_knowledge(self, query: str, n_results: int = 5) -> List[str]:
        """Retrieve semantically related facts from past research"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # extract documents
            if results and results['documents']:
                return results['documents'][0]
            return []
        except Exception as e:
            logger.error(f"Error querying VectorStore: {e}")
            return []

    def get_session_memory(self, session_id: str) -> List[str]:
        """Retrieve all facts associated with a specific session"""
        results = self.collection.get(
            where={"session_id": session_id}
        )
        return results['documents'] if results else []
