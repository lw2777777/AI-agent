"""
embedding_provider.py — 文本向量化

支持：
  - DeepSeek Embedding（通过 OpenAI 兼容接口）
  - BGE（本地，sentence-transformers）
  - Mock（无依赖，用于跑通闭环）

设计：
  - 统一接口 embed(texts: List[str]) -> List[List[float]]
  - 带 LRU 缓存（同一文本不重复 embedding）
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
from collections import OrderedDict
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProvider:
    """抽象接口"""
    dim: int = 0

    def embed(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


class MockEmbeddingProvider(EmbeddingProvider):
    """
    Mock：用 hash 生成确定性伪向量。
    无 API key 时跑通闭环。
    """
    def __init__(self, dim: int = 256):
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        out = []
        for t in texts:
            h = hashlib.md5(t.encode()).digest()
            # 用 hash 字节生成伪随机向量
            rng = np.random.default_rng(int.from_bytes(h[:8], "big"))
            v = rng.normal(0, 1, self.dim)
            v = v / (np.linalg.norm(v) + 1e-8)
            out.append(v.tolist())
        return out


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    用 OpenAI 兼容接口（DeepSeek 也支持）。
    模型名如 "text-embedding-3-small" / "deepseek-embedding"
    """
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
        dim: int = 1536,
    ):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.dim = dim

    def embed(self, texts: List[str]) -> List[List[float]]:
        # 批量，OpenAI 限制单次 2048
        out: List[List[float]] = []
        for i in range(0, len(texts), 100):
            batch = texts[i : i + 100]
            resp = self._client.embeddings.create(model=self.model, input=batch)
            out.extend([d.embedding for d in resp.data])
        return out


class BGEEmbeddingProvider(EmbeddingProvider):
    """本地 BGE 模型（无需 API,但需要 sentence-transformers）"""
    def __init__(self, model_name: str = "BAAI/bge-small-zh-v1.5"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise RuntimeError("pip install sentence-transformers")
        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return vecs.tolist()


class CachedEmbeddingProvider(EmbeddingProvider):
    """包装任意 provider，加 LRU 缓存"""
    def __init__(self, inner: EmbeddingProvider, cache_size: int = 10000):
        self._inner = inner
        self.dim = inner.dim
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[Optional[List[float]]] = [None] * len(texts)
        to_compute: List[str] = []
        to_compute_idx: List[int] = []

        with self._lock:
            for i, t in enumerate(texts):
                key = hashlib.md5(t.encode()).hexdigest()
                if key in self._cache:
                    self._cache.move_to_end(key)
                    out[i] = self._cache[key]
                else:
                    to_compute.append(t)
                    to_compute_idx.append(i)

        if to_compute:
            vecs = self._inner.embed(to_compute)
            with self._lock:
                for idx, t, v in zip(to_compute_idx, to_compute, vecs):
                    key = hashlib.md5(t.encode()).hexdigest()
                    self._cache[key] = v
                    out[idx] = v
                    if len(self._cache) > self._cache_size:
                        self._cache.popitem(last=False)

        return out  # type: ignore


def get_embedding_provider(use_real: bool = False) -> EmbeddingProvider:
    """工厂：默认 Mock，有 key 用 OpenAI"""
    if use_real:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if api_key:
            return CachedEmbeddingProvider(
                OpenAIEmbeddingProvider(
                    api_key=api_key,
                    model="text-embedding-3-small",
                    base_url="https://api.deepseek.com",
                    dim=1536,
                )
            )
    return CachedEmbeddingProvider(MockEmbeddingProvider(dim=256))