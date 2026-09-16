"""
Archival Memory - 长期记忆存储 (Letta Tier 3)

功能:
1. 跨任务持久化记忆
2. FTS5全文搜索 (带BM25排序)
3. 用户隔离
4. 记忆重要性评分
5. 访问追踪
6. 记忆衰减

Schema:
    agent_memory_archival(
        id TEXT PRIMARY KEY,
        user_id TEXT,
        agent_id TEXT,           # 新增: Agent隔离
        type TEXT NOT NULL,      # fact, insight, preference, lesson
        content TEXT NOT NULL,
        metadata_json TEXT,
        importance REAL DEFAULT 1.0,
        access_count INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        last_accessed_at TEXT,
        expires_at TEXT          # 新增: 过期时间
    )
    + FTS5镜像: agent_memory_archival_fts
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

_DEFAULT_DB = str(Path(__file__).parent.parent.parent.parent / "agent_memory.db")

# 建表语句
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS agent_memory_archival (
    id               TEXT PRIMARY KEY,
    user_id          TEXT,
    agent_id         TEXT NOT NULL DEFAULT 'default',
    type             TEXT NOT NULL DEFAULT 'fact',
    content          TEXT NOT NULL,
    metadata_json    TEXT,
    importance       REAL DEFAULT 1.0,
    access_count     INTEGER DEFAULT 0,
    created_at       TEXT NOT NULL,
    last_accessed_at TEXT,
    expires_at       TEXT
)
"""

_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS agent_memory_archival_fts USING fts5(
    content,
    content='agent_memory_archival', content_rowid='rowid'
)
"""

_CREATE_TRIGGERS = [
    """
    CREATE TRIGGER IF NOT EXISTS agent_memory_archival_ai AFTER INSERT ON agent_memory_archival BEGIN
      INSERT INTO agent_memory_archival_fts(rowid, content) VALUES (new.rowid, new.content);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS agent_memory_archival_ad AFTER DELETE ON agent_memory_archival BEGIN
      INSERT INTO agent_memory_archival_fts(agent_memory_archival_fts, rowid, content)
      VALUES('delete', old.rowid, old.content);
    END;
    """,
    """
    CREATE TRIGGER IF NOT EXISTS agent_memory_archival_au AFTER UPDATE ON agent_memory_archival BEGIN
      INSERT INTO agent_memory_archival_fts(agent_memory_archival_fts, rowid, content)
      VALUES('delete', old.rowid, old.content);
      INSERT INTO agent_memory_archival_fts(rowid, content) VALUES (new.rowid, new.content);
    END;
    """,
]

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_memory_user ON agent_memory_archival(user_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_agent ON agent_memory_archival(agent_id)",
    "CREATE INDEX IF NOT EXISTS idx_memory_type ON agent_memory_archival(type)",
    "CREATE INDEX IF NOT EXISTS idx_memory_importance ON agent_memory_archival(importance DESC)",
    "CREATE INDEX IF NOT EXISTS idx_memory_expires ON agent_memory_archival(expires_at)",
]


class ArchivalMemoryStore:
    """长期记忆存储"""

    def __init__(self, db_path: str = _DEFAULT_DB):
        self.db_path = db_path
        self._has_fts = False
        self._ensure_schema()
        self._clean_expired()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        """确保数据库架构存在"""
        with self._conn() as conn:
            conn.execute(_CREATE_TABLE)
            for idx in _CREATE_INDEXES:
                try:
                    conn.execute(idx)
                except sqlite3.OperationalError:
                    pass

            # FTS5支持
            try:
                conn.execute(_CREATE_FTS)
                for t in _CREATE_TRIGGERS:
                    conn.execute(t)
                # 回填FTS
                conn.execute(
                    "INSERT OR IGNORE INTO agent_memory_archival_fts(rowid, content) "
                    "SELECT rowid, content FROM agent_memory_archival "
                    "WHERE rowid NOT IN (SELECT rowid FROM agent_memory_archival_fts)"
                )
                self._has_fts = True
            except sqlite3.OperationalError:
                self._has_fts = False
                logger.debug("FTS5 not available, using LIKE fallback")

    def _clean_expired(self) -> None:
        """清理过期记忆"""
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM agent_memory_archival WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (now,)
            )

    # ==================== CRUD ====================

    def save(
        self,
        content: str,
        user_id: Optional[str] = None,
        agent_id: str = "default",
        memory_type: str = "fact",
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 1.0,
        ttl_days: Optional[int] = None
    ) -> str:
        """
        保存记忆

        Args:
            content: 记忆内容
            user_id: 用户ID
            agent_id: Agent ID
            memory_type: 记忆类型
            metadata: 元数据
            importance: 重要性 (0-1)
            ttl_days: 过期天数 (None = 永不过期)

        Returns:
            记忆ID
        """
        mid = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        expires_at = (datetime.utcnow() + timedelta(days=ttl_days)).isoformat() if ttl_days else None

        with self._conn() as conn:
            conn.execute(
                """INSERT INTO agent_memory_archival 
                   (id, user_id, agent_id, type, content, metadata_json, 
                    importance, created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (mid, user_id, agent_id, memory_type, content,
                 json.dumps(metadata or {}), importance, now, expires_at)
            )

        logger.debug(f"Saved memory {mid}: {content[:50]}...")
        return mid

    def get(self, mid: str) -> Optional[Dict[str, Any]]:
        """获取单条记忆"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM agent_memory_archival WHERE id = ?", (mid,)
            ).fetchone()
        if row:
            # 更新访问计数
            self._update_access(mid)
            return self._row_to_dict(row)
        return None

    def update(
        self,
        mid: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None
    ) -> bool:
        """更新记忆"""
        updates = []
        params = []

        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if metadata is not None:
            updates.append("metadata_json = ?")
            params.append(json.dumps(metadata))
        if importance is not None:
            updates.append("importance = ?")
            params.append(importance)

        if not updates:
            return True

        params.append(mid)
        query = f"UPDATE agent_memory_archival SET {', '.join(updates)} WHERE id = ?"

        with self._conn() as conn:
            cursor = conn.execute(query, params)
            return cursor.rowcount > 0

    def delete(self, mid: str) -> None:
        """删除记忆"""
        with self._conn() as conn:
            conn.execute("DELETE FROM agent_memory_archival WHERE id = ?", (mid,))

    def _update_access(self, mid: str) -> None:
        """更新访问计数"""
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute(
                "UPDATE agent_memory_archival SET access_count = access_count + 1, "
                "last_accessed_at = ? WHERE id = ?",
                (now, mid)
            )

    # ==================== 检索 ====================

    def search(
        self,
        query: str,
        k: int = 5,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        搜索记忆

        Args:
            query: 搜索关键词
            k: 返回数量
            user_id: 用户过滤
            agent_id: Agent过滤
            memory_type: 类型过滤
            min_importance: 最小重要性

        Returns:
            记忆列表
        """
        query = (query or "").strip()
        if not query:
            return self.list(user_id, agent_id, memory_type, limit=k)

        if self._has_fts:
            cleaned = self._fts_sanitize(query)
            if cleaned:
                results = self._fts_search(cleaned, k, user_id, agent_id, memory_type, min_importance)
                if results:
                    return results

        return self._fallback_search(query, k, user_id, agent_id, memory_type, min_importance)

    def _fts_search(
        self,
        query: str,
        k: int,
        user_id: Optional[str],
        agent_id: Optional[str],
        memory_type: Optional[str],
        min_importance: float
    ) -> List[Dict[str, Any]]:
        """FTS5搜索"""
        sql = """
            SELECT m.* FROM agent_memory_archival_fts f
            JOIN agent_memory_archival m ON m.rowid = f.rowid
            WHERE agent_memory_archival_fts MATCH ?
        """
        params = [query]

        if user_id:
            sql += " AND m.user_id = ?"
            params.append(user_id)
        if agent_id:
            sql += " AND m.agent_id = ?"
            params.append(agent_id)
        if memory_type:
            sql += " AND m.type = ?"
            params.append(memory_type)
        if min_importance > 0:
            sql += " AND m.importance >= ?"
            params.append(min_importance)

        sql += " ORDER BY bm25(agent_memory_archival_fts) ASC, m.importance DESC LIMIT ?"
        params.append(k)

        try:
            with self._conn() as conn:
                rows = conn.execute(sql, params).fetchall()
            self._batch_update_access([r["id"] for r in rows])
            return [self._row_to_dict(r) for r in rows]
        except sqlite3.OperationalError as e:
            logger.debug(f"FTS search failed: {e}")
            return []

    def _fallback_search(
        self,
        query: str,
        k: int,
        user_id: Optional[str],
        agent_id: Optional[str],
        memory_type: Optional[str],
        min_importance: float
    ) -> List[Dict[str, Any]]:
        """LIKE模糊搜索 (FTS回退)"""
        like = f"%{query}%"
        sql = "SELECT * FROM agent_memory_archival WHERE content LIKE ?"
        params = [like]

        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        if agent_id:
            sql += " AND agent_id = ?"
            params.append(agent_id)
        if memory_type:
            sql += " AND type = ?"
            params.append(memory_type)
        if min_importance > 0:
            sql += " AND importance >= ?"
            params.append(min_importance)

        sql += " ORDER BY importance DESC, access_count DESC LIMIT ?"
        params.append(k)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        self._batch_update_access([r["id"] for r in rows])
        return [self._row_to_dict(r) for r in rows]

    def list(
        self,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """列出记忆"""
        sql = "SELECT * FROM agent_memory_archival WHERE 1=1"
        params = []

        if user_id:
            sql += " AND user_id = ?"
            params.append(user_id)
        if agent_id:
            sql += " AND agent_id = ?"
            params.append(agent_id)
        if memory_type:
            sql += " AND type = ?"
            params.append(memory_type)

        sql += " ORDER BY importance DESC, created_at DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ==================== 记忆管理 ====================

    def forget_old(self, days: int = 30, min_importance: float = 0.3) -> int:
        """遗忘旧的、低重要性的记忆"""
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM agent_memory_archival "
                "WHERE last_accessed_at IS NOT NULL "
                "AND last_accessed_at <= ? "
                "AND importance < ?",
                (cutoff, min_importance)
            )
            return cursor.rowcount

    def get_stats(self) -> Dict[str, Any]:
        """获取记忆统计"""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM agent_memory_archival").fetchone()[0]
            by_type = conn.execute(
                "SELECT type, COUNT(*) FROM agent_memory_archival GROUP BY type"
            ).fetchall()
            avg_importance = conn.execute(
                "SELECT AVG(importance) FROM agent_memory_archival"
            ).fetchone()[0]

        return {
            "total_memories": total,
            "by_type": {r[0]: r[1] for r in by_type},
            "avg_importance": avg_importance or 0,
            "has_fts": self._has_fts
        }

    def _batch_update_access(self, ids: List[str]) -> None:
        """批量更新访问计数"""
        if not ids:
            return
        now = datetime.utcnow().isoformat()
        placeholders = ",".join("?" * len(ids))
        with self._conn() as conn:
            conn.execute(
                f"UPDATE agent_memory_archival SET access_count = access_count + 1, "
                f"last_accessed_at = ? WHERE id IN ({placeholders})",
                [now] + ids
            )

    # ==================== 工具方法 ====================

    @staticmethod
    def _fts_sanitize(query: str) -> str:
        """FTS5查询净化"""
        # 保留字母数字、空格、连字符
        cleaned = "".join(c for c in query if c.isalnum() or c in (" ", "-", "_"))
        terms = [t for t in cleaned.split() if len(t) >= 2]
        return " OR ".join(terms) if terms else ""

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        d["metadata"] = json.loads(d.pop("metadata_json", "{}"))
        return d


# ==================== 便捷函数 ====================

def create_memory_store(db_path: str = _DEFAULT_DB) -> ArchivalMemoryStore:
    """创建记忆存储实例"""
    return ArchivalMemoryStore(db_path)


def get_global_memory_store() -> ArchivalMemoryStore:
    """获取全局记忆存储实例"""
    global _global_store
    if _global_store is None:
        _global_store = ArchivalMemoryStore()
    return _global_store


_global_store: Optional[ArchivalMemoryStore] = None