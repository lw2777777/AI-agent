from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict
import numpy as np


class VectorStore(ABC):
    """向量存储抽象接口。所有后端实现这个接口。"""

    @abstractmethod
    def add(self, ids: List[str], vectors: np.ndarray,
            metadata: Optional[List[Dict]] = None) -> None:
        """批量加入向量。vectors: (N, D) float32，需已归一化。"""
        ...

    @abstractmethod
    def search(self, query: np.ndarray, k: int,
               filter: Optional[Dict] = None) -> List[Tuple[str, float]]:
        """返回 [(id, score), ...]，按 score 降序（cosine 相似度）。"""
        ...

    @abstractmethod
    def delete(self, ids: List[str]) -> None:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """L2 归一化，归一化后内积 == cosine。"""
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        norm[norm == 0] = 1e-10
        return (v / norm).astype(np.float32)