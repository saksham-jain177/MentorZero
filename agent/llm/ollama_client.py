from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, host: Optional[str], model: Optional[str], timeout_seconds: float = 10.0) -> None:
        self.host = host
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.base_url = host
        # Simple circuit breaker
        self._cb_failures = 0
        self._cb_threshold = 3
        self._cb_open_until: float = 0.0
        self._cb_cooldown_seconds = 10.0

    def health(self) -> Dict[str, Any]:
        if not self.host or not self.model:
            return {
                "host": self.host,
                "model": self.model,
                "ready": False,
                "reason": "missing configuration",
            }
        try:
            with httpx.Client(base_url=self.host, timeout=self.timeout_seconds) as client:
                # Lightweight call: list tags to verify daemon is up
                _resp = client.get("/api/tags")
                ready = _resp.status_code == 200
                return {
                    "host": self.host,
                    "model": self.model,
                    "ready": bool(ready),
                }
        except Exception as exc:  # noqa: BLE001
            return {
                "host": self.host,
                "model": self.model,
                "ready": False,
                "reason": str(exc),
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Async version of health check for use in FastAPI routes"""
        if not self.host or not self.model:
            return {
                "host": self.host,
                "model": self.model,
                "ready": False,
                "reason": "missing configuration",
            }
        try:
            async with httpx.AsyncClient(base_url=self.host, timeout=self.timeout_seconds) as client:
                # Lightweight call: list tags to verify daemon is up
                _resp = await client.get("/api/tags")
                ready = _resp.status_code == 200
                return {
                    "host": self.host,
                    "model": self.model,
                    "ready": bool(ready),
                }
        except Exception as exc:  # noqa: BLE001
            return {
                "host": self.host,
                "model": self.model,
                "ready": False,
                "reason": str(exc),
            }

    def send(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        if not self.host or not self.model:
            raise ValueError("Ollama host/model not configured")

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "options": {"temperature": temperature},
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if context:
            payload["context"] = context

        with httpx.Client(base_url=self.host, timeout=self.timeout_seconds) as client:
            resp = client.post("/api/generate", json=payload)
            resp.raise_for_status()
            data = resp.json()
            text = data.get("response") or data.get("text") or ""
            confidence = data.get("confidence", 0.0)
            return {"text": text, "confidence": float(confidence)}
    
    async def send(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Async version of send for use in FastAPI routes"""
        if not self.host or not self.model:
            logger.error(f"Ollama not configured properly: host={self.host}, model={self.model}")
            raise ValueError("Ollama host/model not configured")
        # Circuit breaker: short-circuit if open
        import time
        now = time.time()
        if now < self._cb_open_until:
            raise RuntimeError("ollama_circuit_open")

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "options": {"temperature": temperature},
            "stream": False,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if context:
            payload["context"] = context
            
        logger.info(f"Sending request to Ollama at {self.host} for model {self.model}")
        logger.debug(f"Payload: {payload}")

        try:
            async with httpx.AsyncClient(base_url=self.host, timeout=self.timeout_seconds) as client:
                logger.info(f"Making POST request to {self.host}/api/generate")
                resp = await client.post("/api/generate", json=payload)
                resp.raise_for_status()
                data = resp.json()
                logger.info("Successfully received response from Ollama")
                logger.debug(f"Response data: {data}")
                text = data.get("response") or data.get("text") or ""
                confidence = data.get("confidence", 0.0)
                # Reset breaker on success
                self._cb_failures = 0
                self._cb_open_until = 0.0
                return {"text": text, "confidence": float(confidence)}
        except Exception as e:
            logger.error(f"Error sending request to Ollama: {str(e)}")
            # Update breaker
            self._cb_failures += 1
            if self._cb_failures >= self._cb_threshold:
                self._cb_open_until = time.time() + self._cb_cooldown_seconds
                logger.warning("Circuit opened for Ollama client due to repeated failures")
            raise