import httpx
import json
import logging
import asyncio
import random
import os
from datetime import datetime
from typing import Optional, Dict, List, Any
from agent.config import get_settings

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for interacting with Ollama or other LLM providers.
    Provides a consistent interface for the MentorZero agents.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.ollama_host
        self.model = self.settings.ollama_model
        # Use a single client for connection pooling
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.settings.ollama_timeout_seconds or 90.0)
        )

    async def _request_with_retry(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Helper to perform requests with exponential backoff retries"""
        max_retries = 3
        base_delay = 2.0
        for attempt in range(max_retries):
            try:
                response = await self.client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
            except (httpx.HTTPStatusError, httpx.RequestError) as e:
                # Only retry on 5xx or connection errors
                is_server_error = isinstance(e, httpx.HTTPStatusError) and e.response.status_code >= 500
                is_network_error = isinstance(e, httpx.RequestError)
                
                if (is_server_error or is_network_error) and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt) + random.uniform(0, 1.0)
                    logger.warning(f"LLM {method} failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {delay:.2f}s...")
                    await asyncio.sleep(delay)
                    continue
                raise e

    async def send_prompt(self, prompt: str, temperature: float = 0.0) -> str:
        """
        Send a prompt to the LLM and return the generated text.
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            response = await self._request_with_retry("POST", "/api/generate", json=payload)
            data = response.json()
            # Log for auditing
            self._log_request(prompt, data.get("response", ""))
            
            return data.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"LLM request finally failed after retries: {e}")
            return f"[Error: {e}]"

    async def generate(self, prompt: str, images: Optional[List[str]] = None, model: Optional[str] = None) -> str:
        """
        Generate a response, optionally with vision capabilities.
        """
        try:
            active_model = model or self.model
            payload = {
                "model": active_model,
                "prompt": prompt,
                "stream": False,
                "images": images or []
            }
            
            response = await self._request_with_retry("POST", "/api/generate", json=payload)
            data = response.json()
            return data.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"Vision/Generation failed after retries: {e}")
            return f"[Error: {e}]"

    def _log_request(self, prompt: str, response: str):
        """Log the request and response for quality monitoring"""
        try:
            log_dir = "./data"
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "ollama_requests.log")
            
            with open(log_file, "a", encoding="utf-8") as f:
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "prompt_head": prompt[:100],
                    "response_head": response[:100]
                }
                f.write(json.dumps(log_entry) + "\n")
        except:
            pass

    async def close(self):
        """Close the underlying HTTP client"""
        await self.client.aclose()
