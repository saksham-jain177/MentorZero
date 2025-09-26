from __future__ import annotations

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

from agent.embeddings.service import EmbeddingService
from agent.vectorstore.faiss_store import FaissStore
from agent.llm.ollama_client import OllamaClient
from agent.config import get_settings


@dataclass
class RetrievedChunk:
    score: float
    text: str
    meta: Dict[str, object]


class AgenticRAGService:
    """Lightweight Agentic RAG service with:
    - ingestion (chunk -> embed -> index)
    - LLM query rewrite (optional)
    - retrieval with MMR diversification
    - citation formatting for prompts
    """

    def __init__(
        self,
        embedder: EmbeddingService,
        store: FaissStore,
        llm: Optional[OllamaClient] = None,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self._rewrite_cache: Dict[str, Tuple[float, str]] = {}
        self._rewrite_ttl_seconds: float = 300.0
        # BM25 state
        self._bm25_docs: List[str] = []
        self._bm25_doc_freqs: Dict[str, int] = {}
        self._bm25_avg_len: float = 0.0

    # -------- ingestion ---------
    def ingest_text(self, session_id: str, raw_text: str) -> int:
        chunks = self._chunk_text(raw_text)
        if not chunks:
            return 0
        metas = [
            {"session_id": session_id, "source": "upload", "i": i}
            for i, _ in enumerate(chunks)
        ]
        self.store.add_texts(self.embedder, chunks, metas)
        # Update BM25
        for ch in chunks:
            self._bm25_add_doc(ch)
        return len(chunks)

    def _chunk_text(self, text: str, target_chars: int | None = None, overlap: int | None = None) -> List[str]:
        s = get_settings()
        if target_chars is None:
            target_chars = int(getattr(s, 'rag_chunk_chars', 800) or 800)
        if overlap is None:
            overlap = int(getattr(s, 'rag_chunk_overlap', 120) or 120)
        text = (text or "").strip()
        if not text:
            return []
        # naive paragraph split, then windowed merge
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paras:
            paras = [text]
        chunks: List[str] = []
        buf: List[str] = []
        size = 0
        for p in paras:
            if size + len(p) + 2 <= target_chars:
                buf.append(p)
                size += len(p) + 2
            else:
                if buf:
                    chunks.append("\n\n".join(buf))
                # start new window with overlap from previous tail
                if chunks:
                    tail = chunks[-1][-overlap:]
                    buf = [tail, p]
                    size = len(tail) + len(p) + 2
                else:
                    buf = [p]
                    size = len(p)
        if buf:
            chunks.append("\n\n".join(buf))
        return chunks

    # -------- retrieval ---------
    async def retrieve(
        self,
        query: str,
        k: int | None = None,
        diversify: bool = True,
        lambda_mult: float | None = None,
        hybrid_alpha: float | None = None,
    ) -> List[RetrievedChunk]:
        s = get_settings()
        if k is None:
            k = int(getattr(s, 'rag_top_k', 5) or 5)
        if lambda_mult is None:
            lambda_mult = float(getattr(s, 'rag_lambda_mult', 0.65) or 0.65)
        if hybrid_alpha is None:
            hybrid_alpha = float(getattr(s, 'rag_hybrid_alpha', 0.30) or 0.30)
        q = query
        if self.llm is not None:
            try:
                q = await self._rewrite_query_cached(query)
            except Exception:
                q = query

        # fetch from dense; blend with BM25
        prelim = self.store.similarity_search(self.embedder, q, k=max(20, k * 4))
        prelim_chunks: List[RetrievedChunk] = [
            RetrievedChunk(score=s, text=(m.get("text") or ""), meta=m) for s, m in prelim
        ]
        bm = self._bm25_rank(q, top_n=max(20, k * 4))
        bm_max = max((s for s, _ in bm), default=1.0)
        bm_norm = {t: (s / bm_max if bm_max > 0 else 0.0) for s, t in bm}
        if prelim_chunks or bm_norm:
            dense_map: Dict[str, float] = {}
            for ch in prelim_chunks:
                dense_map[ch.text] = max(dense_map.get(ch.text, 0.0), ch.score)
            all_texts = set(dense_map.keys()) | set(bm_norm.keys())
            blended: List[RetrievedChunk] = []
            for t in all_texts:
                ds = dense_map.get(t, 0.0)
                bs = bm_norm.get(t, 0.0)
                blended.append(RetrievedChunk(score=(1-hybrid_alpha)*ds + hybrid_alpha*bs, text=t, meta={"text": t}))
            blended.sort(key=lambda c: c.score, reverse=True)
            prelim_chunks = blended
        if not prelim_chunks:
            return []
        if not diversify or len(prelim_chunks) <= k:
            return prelim_chunks[:k]
        chosen = self._mmr(self.embedder, q, prelim_chunks, k, lambda_mult=lambda_mult)
        return chosen

    async def _rewrite_query(self, query: str) -> str:
        prompt = (
            "Rewrite the query for dense retrieval. Keep it concise and specific. "
            "Return only the rewritten query without quotes.\nQuery: " + query
        )
        resp = await self.llm.send(prompt=prompt, temperature=0.0)
        text = (resp or {}).get("text", "").strip()
        return text or query

    async def _rewrite_query_cached(self, query: str) -> str:
        now = time.time()
        hit = self._rewrite_cache.get(query)
        if hit and (now - hit[0]) < self._rewrite_ttl_seconds:
            return hit[1]
        rewritten = await self._rewrite_query(query)
        self._rewrite_cache[query] = (now, rewritten)
        return rewritten

    def _mmr(
        self,
        embedder: EmbeddingService,
        query: str,
        candidates: List[RetrievedChunk],
        k: int,
        lambda_mult: float = 0.65,
    ) -> List[RetrievedChunk]:
        # cosine sim in embedding space
        from math import sqrt
        import numpy as np  # type: ignore

        def cos(a: List[float], b: List[float]) -> float:
            aa = np.array(a, dtype="float32")
            bb = np.array(b, dtype="float32")
            return float(aa.dot(bb) / (np.linalg.norm(aa) * np.linalg.norm(bb) + 1e-12))

        qv = embedder.embed_one(query)
        qv = FaissStore._normalize([qv])[0]  # type: ignore
        cand_vecs = [FaissStore._normalize([embedder.embed_one(c.text)])[0] for c in candidates]

        selected: List[int] = []
        remaining = list(range(len(candidates)))
        while len(selected) < min(k, len(candidates)) and remaining:
            best_idx = None
            best_score = -1e9
            for idx in remaining:
                relevance = cos(qv, cand_vecs[idx])
                diversity = 0.0
                if selected:
                    diversity = max(cos(cand_vecs[idx], cand_vecs[j]) for j in selected)
                score = lambda_mult * relevance - (1 - lambda_mult) * diversity
                if score > best_score:
                    best_score = score
                    best_idx = idx
            selected.append(best_idx)  # type: ignore
            remaining.remove(best_idx)  # type: ignore
        return [candidates[i] for i in selected]

    def format_citations(self, chunks: List[RetrievedChunk]) -> str:
        lines = []
        for i, ch in enumerate(chunks, start=1):
            lines.append(f"[CITATION {i}] {ch.text}")
        return "\n\n".join(lines)

    # --- BM25 helpers ---
    def _tokenize(self, text: str) -> List[str]:
        import re
        return [t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if t]

    def _bm25_add_doc(self, text: str) -> None:
        tokens = self._tokenize(text)
        self._bm25_docs.append(text)
        seen = set()
        for tok in tokens:
            if tok in seen:
                continue
            self._bm25_doc_freqs[tok] = self._bm25_doc_freqs.get(tok, 0) + 1
            seen.add(tok)
        total_len = sum(len(self._tokenize(d)) for d in self._bm25_docs)
        self._bm25_avg_len = total_len / max(1, len(self._bm25_docs))

    def _bm25_rank(self, query: str, top_n: int = 20) -> List[Tuple[float, str]]:
        import math
        s = get_settings()
        N = max(1, len(self._bm25_docs))
        k1 = float(getattr(s, 'rag_bm25_k1', 1.5) or 1.5)
        b = float(getattr(s, 'rag_bm25_b', 0.75) or 0.75)
        q_tokens = self._tokenize(query)
        scores: List[Tuple[float, str]] = []
        for doc in self._bm25_docs:
            d_tokens = self._tokenize(doc)
            dl = len(d_tokens)
            tf: Dict[str, int] = {}
            for t in d_tokens:
                tf[t] = tf.get(t, 0) + 1
            score = 0.0
            for t in q_tokens:
                df = self._bm25_doc_freqs.get(t, 0)
                if df == 0:
                    continue
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                freq = tf.get(t, 0)
                denom = freq + k1 * (1 - b + b * (dl / max(1.0, self._bm25_avg_len)))
                score += idf * ((freq * (k1 + 1)) / max(1e-9, denom))
            if score > 0:
                scores.append((score, doc))
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[:top_n]


