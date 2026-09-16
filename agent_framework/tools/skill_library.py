"""
Skill Library - 技能库 (Voyager风格)

技能是可复用的任务配方，包含:
1. 名称和描述
2. 步骤列表
3. 前置条件
4. 预期输出
5. 使用统计

技能来源:
1. 从成功任务中自动提炼
2. 手动创建
3. 从外部导入

设计理念 (Voyager):
- 技能是经验的压缩表示
- 通过检索相似技能来指导新任务
- 技能可以组合成更复杂的技能
"""

from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import json
import sqlite3
import uuid
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class SkillStep:
    """技能步骤"""
    name: str
    query: str
    tool: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'query': self.query,
            'tool': self.tool,
            'tool_args': self.tool_args
        }


@dataclass
class SkillRecipe:
    """技能配方"""
    version: int = 1
    steps: List[SkillStep] = field(default_factory=list)
    preconditions: Optional[str] = None
    expected_outputs: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'version': self.version,
            'steps': [s.to_dict() for s in self.steps],
            'preconditions': self.preconditions,
            'expected_outputs': self.expected_outputs
        }


@dataclass
class Skill:
    """技能完整定义"""
    id: str
    name: str
    description: str
    recipe: SkillRecipe
    success_count: int = 0
    last_used_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'recipe': self.recipe.to_dict(),
            'success_count': self.success_count,
            'last_used_at': self.last_used_at.isoformat() if self.last_used_at else None,
            'created_at': self.created_at.isoformat(),
            'tags': self.tags
        }


class SkillLibrary:
    """
    技能库 (SQLite存储)
    
    功能:
    1. 添加/获取/搜索技能
    2. 技能使用统计
    3. FTS5全文搜索
    4. 技能组合建议
    
    使用示例:
        library = SkillLibrary()
        
        # 添加技能
        skill_id = library.add(
            name="stock_analysis",
            description="分析股票并生成报告",
            recipe=SkillRecipe(steps=[
                SkillStep(name="fetch_data", query="获取股票数据"),
                SkillStep(name="analyze", query="分析数据"),
                SkillStep(name="report", query="生成报告")
            ])
        )
        
        # 搜索技能
        skills = library.search("分析股票")
        
        # 记录使用
        library.record_use(skill_id)
    """
    
    def __init__(self, db_path: str = "skill_library.db"):
        self.db_path = db_path
        self._has_fts = False
        self._ensure_schema()
    
    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _ensure_schema(self) -> None:
        """确保数据库架构存在"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with self._conn() as conn:
            # 主表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    recipe_json TEXT NOT NULL,
                    success_count INTEGER DEFAULT 0,
                    last_used_at TEXT,
                    created_at TEXT NOT NULL,
                    tags TEXT
                )
            """)
            
            # 索引
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_skills_name 
                ON skills(name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_skills_success 
                ON skills(success_count DESC)
            """)
            
            # FTS5
            try:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS skills_fts USING fts5(
                        name, description,
                        content='skills', content_rowid='rowid'
                    )
                """)
                
                # 触发器
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS skills_fts_ai 
                    AFTER INSERT ON skills BEGIN
                        INSERT INTO skills_fts(rowid, name, description)
                        VALUES (new.rowid, new.name, new.description);
                    END;
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS skills_fts_ad 
                    AFTER DELETE ON skills BEGIN
                        INSERT INTO skills_fts(skills_fts, rowid, name, description)
                        VALUES('delete', old.rowid, old.name, old.description);
                    END;
                """)
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS skills_fts_au 
                    AFTER UPDATE ON skills BEGIN
                        INSERT INTO skills_fts(skills_fts, rowid, name, description)
                        VALUES('delete', old.rowid, old.name, old.description);
                        INSERT INTO skills_fts(rowid, name, description)
                        VALUES (new.rowid, new.name, new.description);
                    END;
                """)
                
                self._has_fts = True
                
            except sqlite3.OperationalError:
                self._has_fts = False
                logger.debug("FTS5 not available, using LIKE fallback")
    
    def add(
        self,
        name: str,
        description: str,
        recipe: SkillRecipe,
        tags: List[str] = None
    ) -> str:
        """添加技能"""
        skill_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO skills 
                   (id, name, description, recipe_json, created_at, tags)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (skill_id, name.strip(), description.strip(),
                 json.dumps(recipe.to_dict()), now,
                 json.dumps(tags or []))
            )
        
        logger.info(f"Added skill: {name} ({skill_id})")
        return skill_id
    
    def get(self, skill_id: str) -> Optional[Skill]:
        """获取技能"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM skills WHERE id = ?", (skill_id,)
            ).fetchone()
        
        if not row:
            return None
        
        return self._row_to_skill(row)
    
    def get_by_name(self, name: str) -> Optional[Skill]:
        """按名称获取技能"""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM skills WHERE name = ?", (name,)
            ).fetchone()
        
        if not row:
            return None
        
        return self._row_to_skill(row)
    
    def search(self, query: str, k: int = 5) -> List[Skill]:
        """搜索技能"""
        query = (query or "").strip()
        if not query:
            return self.list(k)
        
        if self._has_fts:
            cleaned = self._fts_sanitize(query)
            if cleaned:
                skills = self._fts_search(cleaned, k)
                if skills:
                    return skills
        
        return self._fallback_search(query, k)
    
    def _fts_search(self, query: str, k: int) -> List[Skill]:
        """FTS5搜索"""
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    """SELECT s.* FROM skills_fts f
                       JOIN skills s ON s.rowid = f.rowid
                       WHERE skills_fts MATCH ?
                       ORDER BY bm25(skills_fts) ASC
                       LIMIT ?""",
                    (query, k)
                ).fetchall()
            
            return [self._row_to_skill(r) for r in rows]
            
        except sqlite3.OperationalError as e:
            logger.debug(f"FTS search failed: {e}")
            return []
    
    def _fallback_search(self, query: str, k: int) -> List[Skill]:
        """LIKE模糊搜索"""
        like = f"%{query}%"
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM skills
                   WHERE name LIKE ? OR description LIKE ?
                   ORDER BY success_count DESC
                   LIMIT ?""",
                (like, like, k)
            ).fetchall()
        
        return [self._row_to_skill(r) for r in rows]
    
    def list(self, limit: int = 50) -> List[Skill]:
        """列出技能 (按成功率排序)"""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT * FROM skills
                   ORDER BY success_count DESC, created_at DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
        
        return [self._row_to_skill(r) for r in rows]
    
    def record_use(self, skill_id: str) -> None:
        """记录技能使用 (增加成功率)"""
        now = datetime.now().isoformat()
        with self._conn() as conn:
            conn.execute(
                """UPDATE skills 
                   SET success_count = success_count + 1,
                       last_used_at = ?
                   WHERE id = ?""",
                (now, skill_id)
            )
    
    def delete(self, skill_id: str) -> bool:
        """删除技能"""
        with self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM skills WHERE id = ?", (skill_id,)
            )
            return cursor.rowcount > 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
            total_uses = conn.execute(
                "SELECT SUM(success_count) FROM skills"
            ).fetchone()[0] or 0
        
        return {
            'total_skills': total,
            'total_uses': total_uses,
            'avg_success': total_uses / total if total > 0 else 0,
            'has_fts': self._has_fts
        }
    
    def get_recommendations(self, query: str, k: int = 3) -> List[Skill]:
        """获取技能推荐 (基于相似度)"""
        # 搜索相关技能
        skills = self.search(query, k * 2)
        
        if not skills:
            return []
        
        # 按成功率排序
        skills.sort(key=lambda s: s.success_count, reverse=True)
        return skills[:k]
    
    @staticmethod
    def _fts_sanitize(query: str) -> str:
        """FTS5查询净化"""
        # 保留字母数字、空格、连字符
        cleaned = "".join(c for c in query if c.isalnum() or c in (" ", "-", "_"))
        terms = [t for t in cleaned.split() if len(t) >= 2]
        return " OR ".join(terms) if terms else ""
    
    @staticmethod
    def _row_to_skill(row: sqlite3.Row) -> Skill:
        """将数据库行转换为Skill对象"""
        recipe_dict = json.loads(row['recipe_json'])
        recipe = SkillRecipe(
            version=recipe_dict.get('version', 1),
            steps=[SkillStep(**s) for s in recipe_dict.get('steps', [])],
            preconditions=recipe_dict.get('preconditions'),
            expected_outputs=recipe_dict.get('expected_outputs')
        )
        
        return Skill(
            id=row['id'],
            name=row['name'],
            description=row['description'],
            recipe=recipe,
            success_count=row['success_count'],
            last_used_at=datetime.fromisoformat(row['last_used_at']) if row['last_used_at'] else None,
            created_at=datetime.fromisoformat(row['created_at']),
            tags=json.loads(row['tags']) if row['tags'] else []
        )


# ==================== 便捷函数 ====================

_global_library: Optional[SkillLibrary] = None


def get_global_library() -> SkillLibrary:
    """获取全局技能库"""
    global _global_library
    if _global_library is None:
        _global_library = SkillLibrary()
    return _global_library


def create_skill_library(db_path: str = "skill_library.db") -> SkillLibrary:
    """创建技能库"""
    return SkillLibrary(db_path)


def add_skill(
    name: str,
    description: str,
    steps: List[Dict[str, Any]],
    tags: List[str] = None
) -> str:
    """快捷添加技能"""
    library = get_global_library()
    recipe = SkillRecipe(
        steps=[SkillStep(**s) for s in steps]
    )
    return library.add(name, description, recipe, tags)


def get_skill(skill_id: str) -> Optional[Dict[str, Any]]:
    """获取技能"""
    library = get_global_library()
    skill = library.get(skill_id)
    return skill.to_dict() if skill else None


def search_skills(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """搜索技能"""
    library = get_global_library()
    return [s.to_dict() for s in library.search(query, k)]