from __future__ import annotations

from fastapi import FastAPI
from sqlalchemy import Engine

from agent.db.base import get_engine
from agent.db.models import Base
from agent.config import get_azl_config
from agent.llm.ollama_client import OllamaClient


def init_db(engine: Engine | None = None) -> None:
	eng = engine or get_engine()
	Base.metadata.create_all(bind=eng)


def register_startup(app: FastAPI) -> None:
	@app.on_event("startup")
	def _on_startup() -> None:  # noqa: ANN001
		init_db()
		# Optional light warm-up; avoid heavy background jobs by default
		try:
			_ = get_azl_config()
			client = OllamaClient()
			import asyncio
			async def _warm() -> None:
				try:
					await client.send(prompt="ping", temperature=0.0)
				except Exception:
					pass
			try:
				asyncio.create_task(_warm())
			except Exception:
				pass
		except Exception:
			pass

