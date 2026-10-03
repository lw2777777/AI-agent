"""
vector_store.py — 向量存储 + 检索

设计：
  - 复用 SQLite（不引入 Faiss/Milvus）
  - 存 (id, text, embedding, metadata)
  - 检索：cosine 相似度暴力搜索（< 1 万条足够快）
  - 支持元数据过滤（regime / symbol）
"""
from __future__ import annotations

import json
import logging
import sqlite3
import struct
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_DB = str(Path(__file__).parent.parent / "agent_memory.db")

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS vector_store (
    id              TEXT PRIMARY KEY,
    content         TEXT NOT NULL,
    embedding       BLOB NOT NULL,       -- float32 二进制
    metadata_json   TEXT,
    created_at      TEXT NOT NULL
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_vs_created ON vector_store(created_at DESC)",
]


@dataclass
class VectorHit:
    id: str
    content: str
    score: float          # cosine 相似度 [−1, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)


class VectorStore:
    """
    SQLite 向量库。
    适合 < 10 万条；超了应换 Faiss / Milvus。
    """

    def __init__(self, db_path: str = _DEFAULT_DB):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._ensure_schema()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._conn() as conn:
            conn.execute(_CREATE_TABLE)
            for idx in _CREATE_INDEXES:
                try:
                    conn.execute(idx)
                except sqlite3.OperationalError:
                    pass

    # ── 写 ────────────────────────────────
    def add(
        self,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        did = doc_id or uuid.uuid4().hex
        blob = _vec_to_blob(embedding)
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO vector_store "
                "(id, content, embedding, metadata_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (did, content, blob, json.dumps(metadata or {}),
                 datetime.now().isoformat()),
            )
        return did

    def add_batch(
        self,
        contents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[str]:
        metadatas = metadatas or [{} for _ in contents]
        ids = []
        with self._lock, self._conn() as conn:
            for c, e, m in zip(contents, embeddings, metadatas):
                did = uuid.uuid4().hex
                conn.execute(
                    "INSERT INTO vector_store "
                    "(id, content, embedding, metadata_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (did, c, _vec_to_blob(e), json.dumps(m),
                     datetime.now().isoformat()),
                )
                ids.append(did)
        return ids

    # ── 读 ────────────────────────────────
    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[VectorHit]:
        """
        暴力 cosine 搜索（< 10 万条 OK）
        """
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, content, embedding, metadata_json FROM vector_store"
            ).fetchall()

        q = np.asarray(query_embedding, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-8)

        hits: List[VectorHit] = []
        for row in rows:
            meta = json.loads(row["metadata_json"] or "{}")
            if metadata_filter and not _match_filter(meta, metadata_filter):
                continue
            emb = _blob_to_vec(row["embedding"])
            e_norm = emb / (np.linalg.norm(emb) + 1e-8)
            score = float(np.dot(q_norm, e_norm))
            hits.append(VectorHit(
                id=row["id"], content=row["content"],
                score=score, metadata=meta,
            ))

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM vector_store").fetchone()[0]

    def clear(self) -> None:
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM vector_store")


# ── 工具 ──
def _vec_to_blob(vec: List[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)

def _blob_to_vec(blob: bytes) -> np.ndarray:
    n = len(blob) // 4
    return np.asarray(struct.unpack(f"{n}f", blob), dtype=np.float32)

def _match_filter(meta: Dict[str, Any], flt: Dict[str, Any]) -> bool:
    for k, v in flt.items():
        if meta.get(k) != v:
            return False
    return True