"""
hybrid_retriever.py — 混合检索

实现 RAG 的核心链路：
  1. BM25 检索（复用 semantic_memory 的 FTS5）
  2. 向量检索（VectorStore）
  3. RRF 融合
  4. 元数据过滤（regime / symbol）

面试话术：
  "我实现了 BM25 + 向量混合检索，用 RRF 融合。
   RRF 只看排名不看分数，避免了 BM25 分数和向量相似度量纲不同的问题。"
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .embedding_provider import EmbeddingProvider
from .vector_store import VectorStore, VectorHit

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    id: str
    content: str
    score: float
    source: str          # "bm25" / "vector" / "rrf"
    metadata: Dict[str, Any]


class HybridRetriever:
    """
    BM25 + 向量混合检索 + RRF 融合。
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedding_provider: EmbeddingProvider,
        bm25_search_fn=None,       # 可选：外部 BM25 函数 (query, k) -> List[Dict]
        rrf_k: int = 60,
    ):
        self._vs = vector_store
        self._emb = embedding_provider
        self._bm25_fn = bm25_search_fn
        self._rrf_k = rrf_k

    def add_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """索引一条文档：入向量库（BM25 由外部负责）"""
        emb = self._emb.embed_one(content)
        return self._vs.add(content, emb, metadata)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
        vector_top_k: int = 20,
        bm25_top_k: int = 20,
    ) -> List[RetrievalResult]:
        """
        混合检索：
          1. BM25 top-K
          2. 向量 top-K
          3. RRF 融合
          4. 取 top-k
        """
        bm25_results = self._bm25_search(query, bm25_top_k, metadata_filter)
        vector_results = self._vector_search(query, vector_top_k, metadata_filter)

        # RRF 融合
        fused = self._rrf_fuse([bm25_results, vector_results], k=k)
        return fused

    # ── 内部 ──────────────────────────────
    def _bm25_search(self, query, k, metadata_filter):
        if self._bm25_fn is None:
            return []
        try:
            return self._bm25_fn(query, k)
        except Exception as e:
            logger.warning("BM25 检索失败: %s", e)
            return []

    def _vector_search(self, query, k, metadata_filter):
        q_emb = self._emb.embed_one(query)
        hits = self._vs.search(q_emb, k=k, metadata_filter=metadata_filter)
        return [
            {"id": h.id, "content": h.content, "score": h.score, "metadata": h.metadata}
            for h in hits
        ]

    def _rrf_fuse(self, result_lists, k):
        """
        RRF 公式：
          score(d) = Σ_r 1 / (rrf_k + rank_r(d))
        """
        scores: Dict[str, float] = {}
        contents: Dict[str, Dict] = {}

        for results in result_lists:
            for rank, r in enumerate(results, start=1):
                doc_id = r["id"]
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self._rrf_k + rank)
                if doc_id not in contents:
                    contents[doc_id] = r

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [
            RetrievalResult(
                id=doc_id,
                content=contents[doc_id]["content"],
                score=score,
                source="rrf",
                metadata=contents[doc_id].get("metadata", {}),
            )
            for doc_id, score in ranked
        ]