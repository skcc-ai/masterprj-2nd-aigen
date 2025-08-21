"""
SQLite storage for Code Analytica 
"""

import sqlite3
import os
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SQLiteStore:
    """SQLite 데이터베이스 스토어"""

    def __init__(self, db_path: str = "./data/code.db"):
        self.db_path = db_path
        self.ensure_data_dir()
        self.init_database()

    def ensure_data_dir(self):
        """데이터 디렉토리 생성"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def init_database(self):
        """데이터베이스 초기화 및 테이블 생성"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")

            # files 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,
                    language TEXT NOT NULL
                )
            """)

            # symbols 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    type TEXT NOT NULL,
                    language TEXT NOT NULL,
                    signature TEXT,
                    embedding_id INTEGER,
                    start_line INTEGER,
                    end_line INTEGER,
                    FOREIGN KEY (file_id) REFERENCES files(id)
                )
            """)

            # chunks 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol_id INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    tokens INTEGER,
                    embedding_id INTEGER,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (symbol_id) REFERENCES symbols(id)
                )
            """)

            # calls 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    caller_id INTEGER NOT NULL,
                    callee_id INTEGER NOT NULL,
                    call_type TEXT,
                    metadata JSON,
                    FOREIGN KEY (caller_id) REFERENCES symbols(id),
                    FOREIGN KEY (callee_id) REFERENCES symbols(id)
                )
            """)

            # embeddings 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol_id INTEGER,
                    chunk_id INTEGER,
                    faiss_index_id INTEGER NOT NULL,
                    dimension INTEGER NOT NULL,
                    metadata JSON,
                    FOREIGN KEY (symbol_id) REFERENCES symbols(id),
                    FOREIGN KEY (chunk_id) REFERENCES chunks(id)
                )
            """)

            # runs 테이블
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    finished_at TIMESTAMP,
                    input_summary TEXT,
                    output_summary TEXT,
                    metadata JSON
                )
            """)

            # 인덱스
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_path ON files(path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_file_id ON symbols(file_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_symbol_id ON chunks(symbol_id)")

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    # -----------------------
    # INSERT 메서드
    # -----------------------
    def insert_file(self, path: str, language: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT OR IGNORE INTO files (path, language)
                VALUES (?, ?)
            """, (path, language))
            conn.commit()
            return cursor.lastrowid

    def insert_symbol(self, file_id: int, name: str, type_: str, language: str,
                      signature: Optional[str], start_line: int, end_line: int,
                      embedding_id: Optional[int] = None) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO symbols (file_id, name, type, language, signature, start_line, end_line, embedding_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (file_id, name, type_, language, signature, start_line, end_line, embedding_id))
            conn.commit()
            return cursor.lastrowid

    def insert_chunk(self, symbol_id: int, content: str, tokens: int,
                     embedding_id: Optional[int] = None) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO chunks (symbol_id, content, tokens, embedding_id)
                VALUES (?, ?, ?, ?)
            """, (symbol_id, content, tokens, embedding_id))
            conn.commit()
            return cursor.lastrowid

    def insert_call(self, caller_id: int, callee_id: int, call_type: str = "direct",
                    metadata: Optional[str] = None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO calls (caller_id, callee_id, call_type, metadata)
                VALUES (?, ?, ?, ?)
            """, (caller_id, callee_id, call_type, metadata))
            conn.commit()

    def insert_embedding(self, symbol_id: Optional[int], chunk_id: Optional[int],
                         faiss_index_id: int, dimension: int, metadata: Optional[str] = None) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO embeddings (symbol_id, chunk_id, faiss_index_id, dimension, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (symbol_id, chunk_id, faiss_index_id, dimension, metadata))
            conn.commit()
            return cursor.lastrowid

    def insert_run(self, agent_name: str, input_summary: str,
                   output_summary: str, metadata: Optional[str] = None) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO runs (agent_name, input_summary, output_summary, metadata)
                VALUES (?, ?, ?, ?)
            """, (agent_name, input_summary, output_summary, metadata))
            conn.commit()
            return cursor.lastrowid

    # -----------------------
    # 조회 메서드
    # -----------------------
    def get_file_by_path(self, path: str) -> Optional[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM files WHERE path = ?", (path,))
            row = cursor.fetchone()
            if row:
                return dict(zip([col[0] for col in cursor.description], row))
            return None

    def get_symbols_by_file(self, file_id: int) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM symbols WHERE file_id = ? ORDER BY start_line
            """, (file_id,))
            return [dict(zip([col[0] for col in cursor.description], row))
                    for row in cursor.fetchall()]

    def get_chunks_by_symbol(self, symbol_id: int) -> List[Dict[str, Any]]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM chunks WHERE symbol_id = ? ORDER BY id
            """, (symbol_id,))
            return [dict(zip([col[0] for col in cursor.description], row))
                    for row in cursor.fetchall()]

    def get_stats(self) -> Dict[str, Any]:
        """데이터베이스 통계 정보 반환"""
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            
            # 파일 수
            cursor = conn.execute("SELECT COUNT(*) FROM files")
            stats['total_files'] = cursor.fetchone()[0]
            
            # 심볼 수
            cursor = conn.execute("SELECT COUNT(*) FROM symbols")
            stats['total_symbols'] = cursor.fetchone()[0]
            
            # 청크 수
            cursor = conn.execute("SELECT COUNT(*) FROM chunks")
            stats['total_chunks'] = cursor.fetchone()[0]
            
            # 호출 관계 수
            cursor = conn.execute("SELECT COUNT(*) FROM calls")
            stats['total_calls'] = cursor.fetchone()[0]
            
            # 임베딩 수
            cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
            stats['total_embeddings'] = cursor.fetchone()[0]
            
            # 언어별 파일 수
            cursor = conn.execute("SELECT language, COUNT(*) FROM files GROUP BY language")
            stats['files_by_language'] = dict(cursor.fetchall())
            
            # 심볼 타입별 수
            cursor = conn.execute("SELECT type, COUNT(*) FROM symbols GROUP BY type")
            stats['symbols_by_type'] = dict(cursor.fetchall())
            
            return stats
