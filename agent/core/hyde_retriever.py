"""
HyDE (Hypothetical Document Embeddings) Enhanced Retriever
Generates hypothetical answers to improve retrieval quality
"""
from typing import List, Dict, Optional, Tuple
import numpy as np # type: ignore
from dataclasses import dataclass
import asyncio
import logging

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    text: str
    score: float
    source: str
    method: str  # "hyde", "semantic", "keyword"


class HyDERetriever:
    """
    Advanced retrieval using Hypothetical Document Embeddings
    This is what makes RAG actually impressive
    """
    
    def __init__(self, llm_client, embedder=None, bm25_index=None):
        self.llm = llm_client
        self.embedder = embedder
        self.bm25 = bm25_index
    
    async def retrieve(
        self, 
        query: str, 
        k: int = 5,
        use_hyde: bool = True,
        use_multi_query: bool = True
    ) -> List[RetrievalResult]:
        """
        Multi-strategy retrieval with HyDE enhancement
        
        1. Generate hypothetical answer
        2. Create multiple query perspectives  
        3. Combine semantic + keyword search
        4. Rerank with cross-encoder
        """
        
        results = []
        
        # Strategy 1: HyDE - Generate hypothetical perfect answer
        if use_hyde and self.llm:
            hyde_results = await self._hyde_retrieval(query, k)
            results.extend(hyde_results)
        
        # Strategy 2: Multi-Query - Different perspectives
        if use_multi_query and self.llm:
            multi_results = await self._multi_query_retrieval(query, k)
            results.extend(multi_results)
        
        # Strategy 3: Direct retrieval (fallback)
        direct_results = await self._direct_retrieval(query, k)
        results.extend(direct_results)
        
        # Deduplicate and rerank
        final_results = self._deduplicate_and_rerank(results, query, k)
        
        return final_results
    
    async def _hyde_retrieval(self, query: str, k: int) -> List[RetrievalResult]:
        """
        Generate hypothetical document and use it for retrieval
        This often retrieves more relevant documents than the query alone
        """
        
        # Generate hypothetical answer
        hyde_prompt = f"""Write a detailed, factual answer to this question:
{query}

Provide a comprehensive response as if you were writing a Wikipedia article section.
Include specific details, examples, and technical terms that would appear in authoritative sources."""
        
        try:
            hypothetical = await self.llm.send_prompt(hyde_prompt, temperature=0.3)
            
            # Embed the hypothetical answer
            if self.embedder:
                hyde_embedding = self.embedder.embed_one(hypothetical)
                
                # Search using hypothetical embedding
                # This would connect to your FAISS/vector store
                results = []  # self.vector_store.search(hyde_embedding, k)
                
                return [
                    RetrievalResult(
                        text=r.get("text", ""),
                        score=r.get("score", 0),
                        source=r.get("source", ""),
                        method="hyde"
                    )
                    for r in results
                ]
        except Exception as e:
            logger.error(f"HyDE retrieval failed: {e}")
            return []
        
        return []
    
    async def _multi_query_retrieval(self, query: str, k: int) -> List[RetrievalResult]:
        """
        Generate multiple query perspectives for better coverage
        """
        
        multi_prompt = f"""Rewrite this query from 3 different perspectives:
{query}

1. Technical/Expert perspective:
2. Beginner/ELI5 perspective:  
3. Practical/Application perspective:

Write only the rewritten queries, one per line."""
        
        try:
            variations = await self.llm.send_prompt(multi_prompt, temperature=0.5)
            queries = [q.strip() for q in variations.split('\n') if q.strip()][:3] # type: ignore
            
            all_results = []
            for q in queries:
                results = await self._direct_retrieval(q, k // 3)
                all_results.extend(results)
            
            return all_results
        except:
            return []
    
    async def _direct_retrieval(self, query: str, k: int) -> List[RetrievalResult]:
        """
        Standard retrieval using query directly
        Combines semantic and keyword search
        """
        results = []
        
        # Semantic search
        if self.embedder:
            query_embedding = self.embedder.embed_one(query)
            # semantic_results = self.vector_store.search(query_embedding, k)
            # Add to results...
        
        # BM25 keyword search
        if self.bm25:
            # keyword_results = self.bm25.search(query, k)
            # Add to results...
            pass
        
        # No mock results to ensure transparency
        pass
        
        return results
    
    def _deduplicate_and_rerank(
        self, 
        results: List[RetrievalResult], 
        query: str,
        k: int
    ) -> List[RetrievalResult]:
        """
        Remove duplicates and rerank using multiple signals
        """
        
        # Deduplicate by text similarity
        seen_texts = set()
        unique_results = []
        
        for result in results:
            # Simple dedup - in production use fuzzy matching
            text_key = result.text[:100].lower() # type: ignore
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_results.append(result)
        
        # Rerank using multiple factors
        for result in unique_results:
            # Boost HyDE results slightly
            if result.method == "hyde":
                result.score *= 1.1
            
            # Boost results that contain query terms
            query_terms = set(query.lower().split())
            result_terms = set(result.text.lower().split())
            overlap = len(query_terms & result_terms) / len(query_terms)
            result.score *= (1 + overlap * 0.2)
        
        # Sort by score and return top k
        unique_results.sort(key=lambda x: x.score, reverse=True)
        return unique_results[:k] # type: ignore


class CrossEncoderReranker:
    """
    Neural reranker for final result ordering
    Uses a cross-encoder model to score query-document pairs
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
    
    def load_model(self):
        """Load cross-encoder model (requires sentence-transformers)"""
        try:
            from sentence_transformers import CrossEncoder # type: ignore
            self.model = CrossEncoder(self.model_name)
        except ImportError:
            logger.warning("CrossEncoder requires sentence-transformers. Reranking will use original order.")
            self.model = None
    
    def rerank(
        self, 
        query: str, 
        documents: List[str], 
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents using cross-encoder
        Returns list of (index, score) tuples
        """
        if not self.model:
            # Fallback to returning original order
            return [(i, 1.0 - i * 0.1) for i in range(min(top_k, len(documents)))]
        
        # Score all query-document pairs
        pairs = [[query, doc] for doc in documents]
        scores = self.model.predict(pairs) # type: ignore
        
        # Sort by score
        scored_docs = [(i, score) for i, score in enumerate(scores)]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k] # type: ignore


# Demo showing the difference HyDE makes
async def compare_retrieval_methods():
    """Show how HyDE improves retrieval quality"""
    
    query = "How does attention mechanism work in transformers?"
    
    print("Query:", query)
    print("\n" + "="*50)
    
    # Standard retrieval would just embed the query
    print("\n1. STANDARD RETRIEVAL:")
    print("   Embeds: 'How does attention mechanism work in transformers?'")
    print("   Problem: Short query, might miss detailed documents")
    
    # HyDE generates a hypothetical answer first
    print("\n2. HYDE RETRIEVAL:")
    print("   First generates hypothetical answer:")
    print("   'The attention mechanism in transformers is a key component that allows...")
    print("   the model to weigh the importance of different parts of the input...")
    print("   It uses Query, Key, and Value matrices to compute attention scores...'")
    print("   Then embeds this detailed answer to find similar documents")
    print("   Result: Finds more technical, detailed documents")
    
    # Multi-query explores different angles
    print("\n3. MULTI-QUERY RETRIEVAL:")
    print("   Generates variations:")
    print("   - 'Transformer architecture self-attention mathematical formulation'")
    print("   - 'What is attention in neural networks simple explanation'")
    print("   - 'Implementing multi-head attention in practice'")
    print("   Result: Broader coverage of the topic")


if __name__ == "__main__":
    asyncio.run(compare_retrieval_methods())
