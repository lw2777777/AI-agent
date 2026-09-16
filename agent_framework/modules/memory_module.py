"""
Memory Module - 记忆管理

三层记忆架构:
1. 工作记忆 (Working Memory): 当前会话的短期记忆
2. 情景记忆 (Episodic Memory): 历史会话的长期记忆
3. 代理记忆 (Agentic Memory): Agent自主决定记住什么

记忆类型:
- fact: 事实性知识
- preference: 用户偏好
- experience: 经验教训
- reflection: 反思见解
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import sqlite3
import uuid
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """记忆条目"""
    id: str
    content: str
    memory_type: str  # 'fact', 'preference', 'experience', 'reflection'
    timestamp: datetime
    importance: float = 1.0  # 0-1
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'content': self.content,
            'type': self.memory_type,
            'timestamp': self.timestamp.isoformat(),
            'importance': self.importance,
            'access_count': self.access_count,
            'metadata': self.metadata
        }


class WorkingMemory:
    """
    工作记忆 - 短期记忆
    
    特点:
    - 容量有限 (默认20条)
    - 按重要性排序
    - 自动遗忘最不重要的
    """
    
    def __init__(self, capacity: int = 20):
        self.capacity = capacity
        self.items: List[MemoryItem] = []
    
    def add(self, content: str, memory_type: str = 'fact', 
            metadata: Dict[str, Any] = None, importance: float = 1.0) -> MemoryItem:
        """添加记忆"""
        item = MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            timestamp=datetime.now(),
            importance=min(1.0, max(0.0, importance)),
            metadata=metadata or {}
        )
        self.items.append(item)
        
        # 超容量时遗忘
        if len(self.items) > self.capacity:
            self.items.sort(key=lambda x: x.importance * (1 + x.access_count * 0.1))
            self.items = self.items[-self.capacity:]
        
        return item
    
    def recall(self, query: str, limit: int = 5) -> List[MemoryItem]:
        """回忆相关记忆 (关键词匹配)"""
        query_words = set(query.lower().split())
        scored = []
        
        for item in self.items:
            content_words = set(item.content.lower().split())
            overlap = len(query_words & content_words)
            if overlap > 0:
                score = (overlap / max(len(query_words), 1)) * item.importance
                scored.append((score, item))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [item for _, item in scored[:limit]]
        
        # 更新访问计数
        for item in results:
            item.access_count += 1
        
        return results
    
    def get_all(self) -> List[MemoryItem]:
        """获取所有记忆"""
        return self.items.copy()
    
    def clear(self) -> None:
        """清空记忆"""
        self.items = []
    
    def forget(self, item_id: str) -> bool:
        """遗忘特定记忆"""
        for i, item in enumerate(self.items):
            if item.id == item_id:
                del self.items[i]
                return True
        return False
    
    def to_dict(self) -> List[Dict[str, Any]]:
        return [item.to_dict() for item in self.items]


class EpisodicMemory:
    """
    情景记忆 - 长期记忆 (SQLite存储)
    
    特点:
    - 持久化存储
    - 支持全文搜索
    - 按重要性排序
    - 支持时间衰减
    """
    
    def __init__(self, db_path: str = "agent_memory.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """初始化数据库"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memory (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                importance REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                metadata TEXT,
                expires_at TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodic_type 
            ON episodic_memory(memory_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodic_importance 
            ON episodic_memory(importance DESC)
        """)
        conn.commit()
        conn.close()
    
    def add(self, content: str, memory_type: str = 'experience',
            metadata: Dict[str, Any] = None, importance: float = 1.0,
            ttl_days: Optional[int] = None) -> MemoryItem:
        """添加记忆"""
        item = MemoryItem(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            timestamp=datetime.now(),
            importance=importance,
            metadata=metadata or {}
        )
        
        expires_at = None
        if ttl_days:
            expires_at = (datetime.now() + timedelta(days=ttl_days)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO episodic_memory 
            (id, content, memory_type, timestamp, importance, metadata, expires_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (item.id, item.content, item.memory_type, 
              item.timestamp.isoformat(), item.importance,
              json.dumps(item.metadata), expires_at))
        conn.commit()
        conn.close()
        
        return item
    
    def recall(self, query: str, limit: int = 5, 
               memory_type: Optional[str] = None,
               min_importance: float = 0.0) -> List[MemoryItem]:
        """搜索记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        sql = "SELECT * FROM episodic_memory WHERE 1=1"
        params = []
        
        # 关键词搜索 (简单版)
        words = query.lower().split()
        if words:
            conditions = ' OR '.join([f"content LIKE ?" for _ in words[:3]])
            sql += f" AND ({conditions})"
            params.extend([f"%{w}%" for w in words[:3]])
        
        if memory_type:
            sql += " AND memory_type = ?"
            params.append(memory_type)
        
        if min_importance > 0:
            sql += " AND importance >= ?"
            params.append(min_importance)
        
        # 过滤过期
        sql += " AND (expires_at IS NULL OR expires_at > datetime('now'))"
        
        sql += " ORDER BY importance DESC, access_count DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        
        items = []
        for row in rows:
            item = MemoryItem(
                id=row[0],
                content=row[1],
                memory_type=row[2],
                timestamp=datetime.fromisoformat(row[3]),
                importance=row[4],
                access_count=row[5],
                metadata=json.loads(row[6]) if row[6] else {}
            )
            items.append(item)
            self._update_access(item.id)
        
        return items
    
    def _update_access(self, item_id: str):
        """更新访问计数"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE episodic_memory 
            SET access_count = access_count + 1
            WHERE id = ?
        """, (item_id,))
        conn.commit()
        conn.close()
    
    def delete(self, item_id: str) -> bool:
        """删除记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM episodic_memory WHERE id = ?", (item_id,))
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        return affected > 0
    
    def cleanup_expired(self) -> int:
        """清理过期记忆"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM episodic_memory WHERE expires_at IS NOT NULL AND expires_at <= datetime('now')"
        )
        count = cursor.rowcount
        conn.commit()
        conn.close()
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        total = cursor.execute("SELECT COUNT(*) FROM episodic_memory").fetchone()[0]
        by_type = cursor.execute(
            "SELECT memory_type, COUNT(*) FROM episodic_memory GROUP BY memory_type"
        ).fetchall()
        
        conn.close()
        
        return {
            'total': total,
            'by_type': {t[0]: t[1] for t in by_type}
        }


class AgenticMemoryModule:
    """
    代理记忆 - Agent自主决定记住什么
    
    特点:
    - Agent主动存储重要的信息
    - 自动检索相关记忆
    - 支持记忆更新和遗忘
    - 按重要性排序
    """
    
    MEMORY_TYPES = {
        'fact': '事实性信息',
        'preference': '用户偏好',
        'experience': '经验教训',
        'reflection': '反思见解',
        'alert': '提醒和警告',
        'analysis': '分析结果'
    }
    
    def __init__(
        self,
        user_id: str = 'default',
        agent_id: str = 'default',
        db_path: Optional[str] = None,
        max_memories: int = 1000,
        enable_auto_store: bool = True
    ):
        self.user_id = user_id
        self.agent_id = agent_id
        self.db_path = db_path or f"agent_memory_{agent_id}.db"
        self.max_memories = max_memories
        self.enable_auto_store = enable_auto_store
        
        self._episodic = EpisodicMemory(self.db_path)
        self._working = WorkingMemory(capacity=20)
        
        # 记忆重要性衰减
        self._decay_rate = 0.01
    
    def store(self, content: str, memory_type: str = 'fact',
              metadata: Dict[str, Any] = None, 
              importance: float = 1.0,
              ttl_days: Optional[int] = None) -> str:
        """存储记忆"""
        item = self._episodic.add(
            content=content,
            memory_type=memory_type,
            metadata={'user_id': self.user_id, 'agent_id': self.agent_id, **(metadata or {})},
            importance=importance,
            ttl_days=ttl_days
        )
        
        # 同时加入工作记忆
        self._working.add(content, memory_type, metadata, importance)
        
        return item.id
    
    def recall(self, query: str, limit: int = 5,
               memory_type: Optional[str] = None) -> List[MemoryItem]:
        """检索记忆"""
        # 先从工作记忆检索
        working_results = self._working.recall(query, limit)
        working_ids = {w.id for w in working_results}
        
        # 再从长期记忆检索 (补充)
        remaining = limit - len(working_results)
        if remaining > 0:
            episodic_results = self._episodic.recall(
                query, remaining, memory_type, min_importance=0.3
            )
            # 去重
            for item in episodic_results:
                if item.id not in working_ids:
                    working_results.append(item)
        
        return working_results[:limit]
    
    def get_context(self, query: str, limit: int = 3) -> str:
        """获取记忆上下文 (用于Prompt)"""
        memories = self.recall(query, limit)
        
        if not memories:
            return ""
        
        lines = ["## Relevant Memories"]
        for mem in memories:
            lines.append(f"- {mem.content} (importance: {mem.importance:.2f})")
        
        return "\n".join(lines)
    
    def store_financial_context(self, symbol: str, context_type: str,
                                data: Dict[str, Any]) -> str:
        """存储金融上下文"""
        content = f"{symbol}: {json.dumps(data)[:200]}"
        return self.store(
            content=content,
            memory_type='analysis',
            metadata={'symbol': symbol, 'context_type': context_type},
            importance=0.8
        )
    
    def get_symbol_memories(self, symbol: str, limit: int = 5) -> List[MemoryItem]:
        """获取特定标的的记忆"""
        return self.recall(symbol, limit, memory_type='analysis')
    
    def forget(self, memory_id: str) -> bool:
        """遗忘特定记忆"""
        return self._episodic.delete(memory_id)
    
    def update_importance(self, memory_id: str, importance: float) -> bool:
        """更新记忆重要性"""
        # 简化: 重新存储(实际应该更新)
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self._episodic.get_stats()
        stats['working_memory_count'] = len(self._working.items)
        return stats
    
    def to_agent_config(self) -> Dict[str, Any]:
        """转换为Agent配置"""
        return {
            'enable_agentic_memory': True,
            'enable_user_memories': True,
            'add_memories_to_context': True,
            'memory_types': list(self.MEMORY_TYPES.keys())
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'AgenticMemoryModule':
        """从配置创建"""
        return cls(
            user_id=config.get('user_id', 'default'),
            agent_id=config.get('agent_id', 'default'),
            db_path=config.get('db_path'),
            max_memories=config.get('max_memories', 1000),
            enable_auto_store=config.get('enable_auto_store', True)
        )


class MemoryModule:
    """
    统一记忆模块
    
    组合工作记忆、情景记忆和代理记忆
    """
    
    def __init__(
        self,
        user_id: str = 'default',
        agent_id: str = 'default',
        db_path: Optional[str] = None,
        enable_working_memory: bool = True,
        enable_episodic_memory: bool = True,
        enable_agentic_memory: bool = True
    ):
        self.user_id = user_id
        self.agent_id = agent_id
        self.db_path = db_path or f"memory_{agent_id}.db"
        
        self.working = WorkingMemory() if enable_working_memory else None
        self.episodic = EpisodicMemory(self.db_path) if enable_episodic_memory else None
        self.agentic = AgenticMemoryModule(
            user_id, agent_id, self.db_path
        ) if enable_agentic_memory else None
    
    def remember(self, content: str, memory_type: str = 'fact',
                 metadata: Dict[str, Any] = None,
                 importance: float = 1.0) -> str:
        """记住信息"""
        if self.agentic:
            return self.agentic.store(content, memory_type, metadata, importance)
        elif self.working:
            item = self.working.add(content, memory_type, metadata, importance)
            return item.id
        elif self.episodic:
            item = self.episodic.add(content, memory_type, metadata, importance)
            return item.id
        return ''
    
    def recall(self, query: str, limit: int = 5) -> List[MemoryItem]:
        """回忆信息"""
        if self.agentic:
            return self.agentic.recall(query, limit)
        elif self.working:
            return self.working.recall(query, limit)
        elif self.episodic:
            return self.episodic.recall(query, limit)
        return []
    
    def get_context(self, query: str) -> str:
        """获取记忆上下文"""
        if self.agentic:
            return self.agentic.get_context(query)
        return ""
    
    def to_agent_config(self) -> Dict[str, Any]:
        """转换为Agent配置"""
        config = {}
        if self.working:
            config['working_memory'] = True
        if self.episodic:
            config['episodic_memory'] = True
        if self.agentic:
            config.update(self.agentic.to_agent_config())
        return config