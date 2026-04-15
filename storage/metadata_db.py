"""
SQLite 元数据管理 - 分类/权限/文档信息
"""
import sqlite3
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MetadataDB:
    """SQLite 元数据数据库"""
    
    def __init__(self, db_path: str = "storage/metadata.db"):
        """
        初始化元数据数据库
        :param db_path: SQLite 数据库路径
        """
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """连接数据库"""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()
        logger.info(f"Connected to metadata DB at {self.db_path}")
    
    def close(self):
        """关闭连接"""
        if self.conn:
            self.conn.close()
    
    def _create_tables(self):
        """创建表结构"""
        cursor = self.conn.cursor()
        
        # 文档表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            file_path TEXT,
            category TEXT NOT NULL,
            file_size INTEGER,
            file_hash TEXT,
            status TEXT DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_by TEXT,
            vector_count INTEGER DEFAULT 0,
            entity_count INTEGER DEFAULT 0
        )
        """)
        
        # 分类表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            parent_category TEXT,
            description TEXT,
            visible_roles TEXT,  -- JSON 数组
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # 用户表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            role TEXT DEFAULT 'user',
            query_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # 查询日志表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            query_text TEXT NOT NULL,
            category_filter TEXT,
            confidence_score REAL,
            confidence_level TEXT,
            response_time_ms INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # 置信度评估记录表
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS confidence_records (
            audit_id TEXT PRIMARY KEY,
            query_id INTEGER,
            confidence_score REAL,
            confidence_level TEXT,
            vector_signal REAL,
            kg_signal REAL,
            faithfulness_signal REAL,
            explanation TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        self.conn.commit()
        logger.info("Created metadata tables")
    
    # 文档操作
    def insert_document(self, doc: Dict[str, Any]) -> str:
        """插入文档记录"""
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO documents 
        (doc_id, file_name, file_path, category, file_size, file_hash, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            doc["doc_id"],
            doc["file_name"],
            doc.get("file_path"),
            doc.get("category", "general"),
            doc.get("file_size"),
            doc.get("file_hash"),
            doc.get("status", "pending")
        ))
        self.conn.commit()
        return doc["doc_id"]
    
    def update_document_status(self, doc_id: str, status: str, vector_count: int = 0, entity_count: int = 0):
        """更新文档状态"""
        cursor = self.conn.cursor()
        cursor.execute("""
        UPDATE documents 
        SET status = ?, vector_count = ?, entity_count = ?, updated_at = CURRENT_TIMESTAMP
        WHERE doc_id = ?
        """, (status, vector_count, entity_count, doc_id))
        self.conn.commit()
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """获取文档信息"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    
    def list_documents(
        self, 
        category: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """列出文档"""
        cursor = self.conn.cursor()
        
        query = "SELECT * FROM documents WHERE 1=1"
        params = []
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
    
    def delete_document(self, doc_id: str):
        """删除文档记录"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
        self.conn.commit()
    
    # 分类操作
    def insert_category(self, category: Dict[str, Any]) -> str:
        """插入分类"""
        cursor = self.conn.cursor()
        import json
        cursor.execute("""
        INSERT OR REPLACE INTO categories 
        (category_id, name, parent_category, description, visible_roles)
        VALUES (?, ?, ?, ?, ?)
        """, (
            category["category_id"],
            category["name"],
            category.get("parent_category"),
            category.get("description"),
            json.dumps(category.get("visible_roles", ["all"]))
        ))
        self.conn.commit()
        return category["category_id"]
    
    def get_categories(self) -> List[Dict]:
        """获取所有分类"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM categories ORDER BY name")
        return [dict(row) for row in cursor.fetchall()]
    
    def get_category_tree(self, parent: Optional[str] = None) -> List[Dict]:
        """获取分类树（支持递归）"""
        cursor = self.conn.cursor()
        
        if parent is None:
            cursor.execute("SELECT * FROM categories WHERE parent_category IS NULL")
        else:
            cursor.execute("SELECT * FROM categories WHERE parent_category = ?", (parent,))
        
        return [dict(row) for row in cursor.fetchall()]
    
    # 用户操作
    def get_or_create_user(self, user_id: str, username: str, role: str = "user") -> Dict:
        """获取或创建用户"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        
        if not row:
            cursor.execute("""
            INSERT INTO users (user_id, username, role)
            VALUES (?, ?, ?)
            """, (user_id, username, role))
            self.conn.commit()
            return {"user_id": user_id, "username": username, "role": role}
        
        return dict(row)
    
    def increment_user_query_count(self, user_id: str):
        """增加用户查询次数"""
        cursor = self.conn.cursor()
        cursor.execute("""
        UPDATE users SET query_count = query_count + 1 WHERE user_id = ?
        """, (user_id,))
        self.conn.commit()
    
    def get_user_stats(self, user_id: str) -> Dict:
        """获取用户统计"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else {}
    
    # 查询日志
    def log_query(
        self,
        user_id: str,
        query_text: str,
        category_filter: Optional[str],
        confidence_score: Optional[float],
        confidence_level: Optional[str],
        response_time_ms: int
    ) -> int:
        """记录查询日志"""
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT INTO query_logs 
        (user_id, query_text, category_filter, confidence_score, confidence_level, response_time_ms)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, query_text, category_filter, confidence_score, confidence_level, response_time_ms))
        self.conn.commit()
        return cursor.lastrowid
    
    # 置信度记录
    def save_confidence_record(self, record: Dict[str, Any]):
        """保存置信度评估记录"""
        cursor = self.conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO confidence_records 
        (audit_id, query_id, confidence_score, confidence_level, 
         vector_signal, kg_signal, faithfulness_signal, explanation)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record["audit_id"],
            record.get("query_id"),
            record["confidence_score"],
            record["confidence_level"],
            record.get("vector_signal"),
            record.get("kg_signal"),
            record.get("faithfulness_signal"),
            record.get("explanation")
        ))
        self.conn.commit()
