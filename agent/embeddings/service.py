from __future__ import annotations

from typing import Iterable, List

import hashlib
import numpy as np


class EmbeddingService:
	def __init__(self, model_name: str) -> None:
		self.model_name = model_name
		self._model = None
		self.dim = 384  # default for MiniLM

	def _ensure_model(self):
		if self._model is not None:
			return
		try:
			from sentence_transformers import SentenceTransformer  # type: ignore

			self._model = SentenceTransformer(self.model_name, device="cpu")
			self.dim = self._model.get_sentence_embedding_dimension()  # type: ignore
		except Exception as exc:  # noqa: BLE001
			# Fallback: mark model as None and log once; we'll use hash embeds
			print("[EmbeddingService] WARNING: sentence-transformers load failed, using hash embeddings", exc)
			self._model = None

	def _hash_embed(self, text: str) -> List[float]:
		# simple deterministic hash to vector
		digest = hashlib.sha256(text.encode("utf-8")).digest()
		vec = np.frombuffer(digest, dtype=np.uint8)[: self.dim]
		if vec.size < self.dim:
			vec = np.pad(vec, (0, self.dim - vec.size))
		vec = vec.astype("float32")
		vec = vec / (np.linalg.norm(vec) + 1e-12)
		return vec.tolist()

	def embed(self, texts: Iterable[str]) -> List[List[float]]:
		self._ensure_model()
		lst = list(texts)
		if self._model is None:
			return [self._hash_embed(t) for t in lst]
		# Retry with backoff
		attempts = 0
		last_err: Exception | None = None
		while attempts < 3:
			try:
				embs = self._model.encode(lst, normalize_embeddings=True)  # type: ignore[attr-defined]
				return embs.tolist()
			except Exception as exc:  # noqa: BLE001
				last_err = exc
				import time
				time.sleep(0.2 * (2 ** attempts))
				attempts += 1
		# Fallback to hash embeddings on persistent failure
		print("[EmbeddingService] ERROR: encode failed, falling back to hash embeddings", last_err)
		return [self._hash_embed(t) for t in lst]

	def embed_one(self, text: str) -> List[float]:
		return self.embed([text])[0]

