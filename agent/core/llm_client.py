import httpx
import json
import logging
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
            
            response = await self.client.post("/api/generate", json=payload)
            response.raise_for_status()
            
            data = response.json()
            # Log for auditing (matches the existing log pattern)
            self._log_request(prompt, data.get("response", ""))
            
            return data.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"LLM request failed: {e}")
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
            
            response = await self.client.post("/api/generate", json=payload)
            response.raise_for_status()
            
            data = response.json()
            return data.get("response", "").strip()
            
        except Exception as e:
            logger.error(f"Vision/Generation failed: {e}")
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
