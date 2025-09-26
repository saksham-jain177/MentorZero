from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, List, Tuple

from agent.embeddings.service import EmbeddingService

try:
	import faiss  # type: ignore
except Exception as exc:  # noqa: BLE001
	raise RuntimeError(
		"faiss-cpu is required for vector search. Please install it via requirements."
	) from exc


class FaissStore:
	def __init__(self, dim: int, index_path: str, meta_path: str) -> None:
		self.dim = dim
		self.index_path = index_path
		self.meta_path = meta_path
		self.meta: Dict[int, Dict[str, Any]] = {}
		self._load()

	def _new_index(self):
		base = faiss.IndexFlatIP(self.dim)
		return faiss.IndexIDMap(base)

	def _load(self) -> None:
		if os.path.exists(self.index_path):
			self.index = faiss.read_index(self.index_path)
		else:
			self.index = self._new_index()
		if os.path.exists(self.meta_path):
			with open(self.meta_path, "r", encoding="utf-8") as f:
				data = json.load(f)
				self.meta = {int(k): v for k, v in data.items()}
		else:
			self.meta = {}

	@property
	def next_id(self) -> int:
		return (max(self.meta.keys()) + 1) if self.meta else 0

	def _persist(self) -> None:
		os.makedirs(os.path.dirname(self.index_path) or ".", exist_ok=True)
		faiss.write_index(self.index, self.index_path)
		with open(self.meta_path, "w", encoding="utf-8") as f:
			json.dump({str(k): v for k, v in self.meta.items()}, f)

	@staticmethod
	def _normalize(vecs: List[List[float]]) -> List[List[float]]:
		import numpy as np  # type: ignore

		arr = np.array(vecs, dtype="float32")
		norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
		arr = arr / norms
		return arr.tolist()

	def add_texts(
		self,
		embedder: EmbeddingService,
		texts: Iterable[str],
		metadatas: Iterable[Dict[str, Any]] | None = None,
	) -> List[int]:
		texts_list = list(texts)
		metas_list = list(metadatas or ({} for _ in texts_list))
		# ensure text is present in metadata
		for i in range(len(texts_list)):
			if 'text' not in metas_list[i]:
				metas_list[i]['text'] = texts_list[i]
		vectors = embedder.embed(texts_list)
		vectors = self._normalize(vectors)
		import numpy as np  # type: ignore

		ids = [self.next_id + i for i in range(len(texts_list))]
		self.index.add_with_ids(np.array(vectors, dtype="float32"), np.array(ids, dtype="int64"))
		for i, meta in zip(ids, metas_list):
			self.meta[i] = meta
		self._persist()
		return ids

	def similarity_search(
		self,
		embedder: EmbeddingService,
		query: str,
		k: int = 5,
	) -> List[Tuple[float, Dict[str, Any]]]:
		if self.index.ntotal == 0:
			return []
		qvec = embedder.embed_one(query)
		qvec = self._normalize([qvec])[0]
		import numpy as np  # type: ignore

		D, I = self.index.search(np.array([qvec], dtype="float32"), k)
		results: List[Tuple[float, Dict[str, Any]]] = []
		for score, idx in zip(D[0].tolist(), I[0].tolist()):
			if idx == -1:
				continue
			results.append((float(score), self.meta.get(int(idx), {})))
		return results

