from __future__ import annotations
from sqlalchemy.orm import Session

from agent.config import get_settings
from agent.llm.ollama_client import OllamaClient
from agent.services.teaching import TeachingService
from agent.services.rag import AgenticRAGService
from agent.embeddings.service import EmbeddingService
from agent.vectorstore.faiss_store import FaissStore
from agent.db.base import get_session_factory
from agent.services.background_processor import background_processor


def get_ollama_client() -> OllamaClient:
	s = get_settings()
	return OllamaClient(host=s.ollama_host, model=s.ollama_model, timeout_seconds=(s.ollama_timeout_seconds or 60.0))

def get_judge_ollama_client() -> OllamaClient:
	s = get_settings()
	model = s.judge_model or s.ollama_model
	return OllamaClient(host=s.ollama_host, model=model, timeout_seconds=(s.ollama_timeout_seconds or 60.0))

def get_teaching_service() -> TeachingService:
    # Build Agentic RAG stack lazily
    s = get_settings()
    embedder = EmbeddingService(s.embedding_model_name)
    store = FaissStore(dim=embedder.dim, index_path=s.faiss_index_path, meta_path=s.faiss_meta_path)
    rag = AgenticRAGService(embedder=embedder, store=store, llm=get_ollama_client())
    return TeachingService(get_ollama_client()).with_rag(rag)


def get_db_session():
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
	