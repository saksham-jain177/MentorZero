from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict


@dataclass
class AZLConfig:
    pass_threshold: float
    max_attempts: int
    daily_budget: int
    score_weights: Dict[str, float]


def _parse_weights(raw: str | None) -> Dict[str, float]:
    if not raw:
        # Safe defaults; override via env AZL_SCORE_WEIGHTS (JSON)
        return {
            "length": 0.1,
            "duplicate": 0.1,
            "consistency": 0.2,
            "roundtrip": 0.2,
            "toxicity": 0.2,
            "placeholder": 0.2,
            "judge": 0.2,
        }
    try:
        weights = json.loads(raw)
        if isinstance(weights, dict):
            return weights
    except Exception:
        pass
    return {
        "length": 0.1,
        "duplicate": 0.1,
        "consistency": 0.2,
        "roundtrip": 0.2,
        "toxicity": 0.2,
        "placeholder": 0.2,
        "judge": 0.2,
    }


def get_azl_config() -> AZLConfig:
    return AZLConfig(
        pass_threshold=float(os.getenv("AZL_PASS_THRESHOLD", "0.75")),
        max_attempts=int(os.getenv("AZL_MAX_ATTEMPTS", "2")),
        daily_budget=int(os.getenv("AZL_DAILY_BUDGET", "200")),
        score_weights=_parse_weights(os.getenv("AZL_SCORE_WEIGHTS")),
    )

"""
Configuration module for MentorZero.
"""
import os
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Settings for MentorZero."""
    
    app_env: str = "dev"
    
    # Ollama settings
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama2-uncensored:7b"
    ollama_timeout_seconds: Optional[float] = 90.0
    # Dedicated judge model (optional). If set, used by AZLScorer/validators.
    judge_model: Optional[str] = None
    # Optional second-stage heavy judge model for cascade
    judge_model_heavy: Optional[str] = None
    # Cascade settings
    judge_margin: float = 0.1  # within margin of threshold triggers heavy judge
    judge_timeout_fast: float = 8.0
    judge_timeout_heavy: float = 30.0
    
    # Database settings
    db_path: str = "./data/mentorzero.db"
    
    # Embeddings settings (default to deterministic to avoid native crashes on some setups)
    embedding_model_name: str = "deterministic"
    
    # FAISS settings
    faiss_index_path: str = "./data/faiss.index"
    faiss_meta_path: str = "./data/faiss_meta.json"
    faiss_index_type: str = "flat"  # flat | hnsw
    faiss_hnsw_ef_search: int = 50
    
    # Voice settings
    whisper_model: Optional[str] = "tiny"
    tts_settings: Optional[str] = '{"model": "facebook/mms-tts-eng"}'
    tts_device: Optional[str] = "cpu"
    
    # AZL settings
    azl_threshold: float = 0.8

    # Generation settings
    generation_temperature: float = 0.0

    # RAG defaults
    rag_top_k: int = 5
    rag_lambda_mult: float = 0.65
    rag_hybrid_alpha: float = 0.30
    rag_chunk_chars: int = 800
    rag_chunk_overlap: int = 120
    rag_bm25_k1: float = 1.5
    rag_bm25_b: float = 0.75
    rag_semantic_chunking: bool = True
    rag_use_reranker: bool = False
    rag_disable_dense: bool = True
    
    # Pydantic v2 configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MZ_",
        extra="allow"
    )


@lru_cache()
def get_settings() -> Settings:
    """
    Get the settings for MentorZero.
    
    Returns:
        Settings: The settings for MentorZero.
    """
    return Settings()