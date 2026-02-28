import sqlite3
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Any

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Manages durable persistence for API responses and expensive computations.
    """
    def __init__(self, db_path: str = "./data/cache.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Initialize the cache table"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS api_cache (
                        cache_key TEXT PRIMARY KEY,
                        response_json TEXT,
                        provider TEXT,
                        created_at TIMESTAMP
                    )
                """)
                # Index for faster retrieval by provider
                conn.execute("CREATE INDEX IF NOT EXISTS idx_provider ON api_cache(provider)")
        except Exception as e:
            logger.error(f"Failed to initialize cache DB: {e}")

    def get(self, cache_key: str, ttl_hours: int = 24) -> Optional[Any]:
        """Retrieve a cached response if within TTL"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT response_json, created_at FROM api_cache WHERE cache_key = ?", 
                    (cache_key,)
                ).fetchone()
                
                if row:
                    created_at = datetime.fromisoformat(row["created_at"])
                    if datetime.now() - created_at < timedelta(hours=ttl_hours):
                        return json.loads(row["response_json"])
                    else:
                        # Expired
                        conn.execute("DELETE FROM api_cache WHERE cache_key = ?", (cache_key,))
        except Exception as e:
            logger.error(f"Error reading from cache: {e}")
        return None

    def set(self, cache_key: str, response: Any, provider: str = "generic"):
        """Save a response to the persistent cache"""
        try:
            response_json = json.dumps(response)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO api_cache (cache_key, response_json, provider, created_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        response_json = excluded.response_json,
                        created_at = excluded.created_at
                """, (cache_key, response_json, provider, datetime.now().isoformat()))
        except Exception as e:
            logger.error(f"Error writing to cache: {e}")

# Global instance
cache_manager = CacheManager()
