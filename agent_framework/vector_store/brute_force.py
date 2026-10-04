from typing import List, Tuple, Optional, Dict
import numpy as np
from .base import VectorStore


class BruteForceStore(VectorStore):
    """暴力搜索：100% 召回基线。"""

    def __init__(self, dim: int):
        self.dim = dim
        self.ids: List[str] = []
        self.metadata: List[Dict] = []
        self.matrix: Optional[np.ndarray] = None  # (N, D) float32
        self._id_to_idx: Dict[str, int] = {}

    def add(self, ids, vectors, metadata=None):
        vectors = self.normalize(np.asarray(vectors, dtype=np.float32))
        if self.matrix is None:
            self.matrix = vectors
        else:
            self.matrix = np.vstack([self.matrix, vectors])
        start = len(self.ids)
        for i, _id in enumerate(ids):
            self._id_to_idx[_id] = start + i
        self.ids.extend(ids)
        self.metadata.extend(metadata or [{}] * len(ids))

    def search(self, query, k, filter=None):
        if self.matrix is None or len(self.ids) == 0:
            return []
        q = self.normalize(np.asarray(query, dtype=np.float32).reshape(1, -1))[0]
        scores = self.matrix @ q  # (N,)
        if filter:
            mask = np.array([all(m.get(kk) == vv for kk, vv in filter.items())
                             for m in self.metadata])
            scores = np.where(mask, scores, -np.inf)
        k = min(k, len(self.ids))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(self.ids[i], float(scores[i])) for i in idx]

    def delete(self, ids):
        drop = set(ids)
        keep = [i for i, _id in enumerate(self.ids) if _id not in drop]
        self.ids = [self.ids[i] for i in keep]
        self.metadata = [self.metadata[i] for i in keep]
        self.matrix = self.matrix[keep] if self.matrix is not None else None
        self._id_to_idx = {_id: i for i, _id in enumerate(self.ids)}

    def __len__(self):
        return len(self.ids)