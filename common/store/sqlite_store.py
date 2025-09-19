"""
SQLite storage for Code Analytica 
AST 구조 기반 코드 분석 결과 저장
"""

import sqlite3
import os
import json
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class SQLiteStore:
    """SQLite 데이터베이스 스토어 - AST 구조 기반"""

    def __init__(self, db_path: str = "./data/code.db"):
        self.db_path = db_path
        self.ensure_data_dir()
        self.init_database()

    def ensure_data_dir(self):
        """데이터 디렉토리 생성"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def init_database(self):
        """데이터베이스 초기화 및 테이블 생성 - 사용자 요청 스키마"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")

            # 소스 파일 단위 메타데이터
            conn.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT NOT NULL UNIQUE,   -- 파일 경로
                    language TEXT NOT NULL,      -- 언어 (python, java, cpp...)
                    llm_summary TEXT,            -- 파일 전체 요약 (LLM 분석 결과)
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 코드 심볼 기본 정보 (클래스, 함수, 메서드, 변수 등)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS symbols (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER NOT NULL,    -- files.id 참조
                    name TEXT NOT NULL,          -- 심볼명 (클래스, 함수 등)
                    type TEXT NOT NULL,          -- function, class, method, variable 등
                    language TEXT NOT NULL,      -- 소스 언어
                    signature TEXT,              -- 시그니처 (def foo(x:int) -> str)
                    start_line INTEGER,          -- 시작 라인
                    end_line INTEGER,            -- 끝 라인

                    -- LLM 분석 정보
                    llm_summary TEXT,            -- 요약 (1~2문장)
                    responsibility TEXT,         -- 책임/역할
                    design_notes TEXT,           -- 설계 특징/비고
                    collaboration TEXT,          -- 협력 모듈/심볼

                    embedding_id INTEGER,        -- embeddings.id
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (file_id) REFERENCES files(id)
                )
            """)

            # 심볼 내부 청크 단위 저장 (코드 조각 or 요약 조각)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol_id INTEGER NOT NULL,  -- symbols.id 참조
                    chunk_type TEXT NOT NULL,    -- code | summary | docstring | comment
                    content TEXT NOT NULL,       -- 코드 스니펫 또는 요약 텍스트
                    tokens INTEGER,              -- 토큰 수 (길이 관리용)
                    embedding_id INTEGER,        -- embeddings.id
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (symbol_id) REFERENCES symbols(id)
                )
            """)

            # 호출 관계 (호출 그래프의 엣지)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    caller_id INTEGER NOT NULL,  -- 호출하는 함수 (symbols.id)
                    callee_id INTEGER NOT NULL,  -- 호출되는 함수 (symbols.id)
                    call_type TEXT,              -- direct, indirect, async 등
                    confidence REAL,             -- (옵션) LLM 추출 신뢰도
                    metadata TEXT,               -- 추가 메타데이터 (JSON)
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (caller_id) REFERENCES symbols(id),
                    FOREIGN KEY (callee_id) REFERENCES symbols(id)
                )
            """)

            # 벡터 임베딩 테이블 (FAISS 인덱스와 매핑)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    object_type TEXT NOT NULL,   -- file | symbol | chunk
                    object_id INTEGER NOT NULL,  -- 해당 object의 id
                    vector BLOB NOT NULL,        -- 임베딩 벡터 (float array 직렬화)
                    dimension INTEGER NOT NULL,  -- 벡터 차원
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 에이전트 실행 로그
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_name TEXT NOT NULL,    -- StructSynth, InsightGen 등
                    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    finished_at TIMESTAMP,
                    input_summary TEXT,
                    output_summary TEXT,
                    metadata TEXT,               -- JSON 형태의 추가 메타데이터
                    status TEXT DEFAULT 'running' -- running, completed, failed
                )
            """)

            # 인덱스 생성
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_path ON files(path)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_files_language ON files(language)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_file_id ON symbols(file_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_type ON symbols(type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_symbols_language ON symbols(language)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_symbol_id ON chunks(symbol_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_type ON chunks(chunk_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_calls_caller ON calls(caller_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_calls_callee ON calls(callee_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_object ON embeddings(object_type, object_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_agent ON runs(agent_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status)")

            conn.commit()
            logger.info(f"Database initialized with new schema at {self.db_path}")

    # -----------------------
    # INSERT 메서드
    # -----------------------
    def insert_file(self, path: str, language: str, llm_summary: Optional[str] = None) -> int:
        """파일 정보 저장"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT OR REPLACE INTO files (path, language, llm_summary)
                VALUES (?, ?, ?)
            """, (path, language, llm_summary))
            conn.commit()
            return cursor.lastrowid

    def insert_symbol(self, file_id: int, name: str, type_: str, language: str,
                      signature: Optional[str], start_line: int, end_line: int,
                      llm_summary: Optional[str] = None, responsibility: Optional[str] = None,
                      design_notes: Optional[str] = None, collaboration: Optional[str] = None,
                      embedding_id: Optional[int] = None) -> int:
        """심볼 정보 저장"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO symbols (file_id, name, type, language, signature, start_line, end_line,
                                   llm_summary, responsibility, design_notes, collaboration, embedding_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (file_id, name, type_, language, signature, start_line, end_line,
                  llm_summary, responsibility, design_notes, collaboration, embedding_id))
            conn.commit()
            return cursor.lastrowid

    def insert_chunk(self, symbol_id: int, chunk_type: str, content: str, tokens: Optional[int] = None,
                     embedding_id: Optional[int] = None) -> int:
        """청크 정보 저장"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO chunks (symbol_id, chunk_type, content, tokens, embedding_id)
                VALUES (?, ?, ?, ?, ?)
            """, (symbol_id, chunk_type, content, tokens, embedding_id))
            conn.commit()
            return cursor.lastrowid

    def insert_call(self, caller_id: int, callee_id: int, call_type: str = "direct",
                    confidence: Optional[float] = None, metadata: Optional[Dict] = None):
        """호출 관계 저장"""
        metadata_json = json.dumps(metadata) if metadata else None
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO calls (caller_id, callee_id, call_type, confidence, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (caller_id, callee_id, call_type, confidence, metadata_json))
            conn.commit()

    def insert_embedding(self, object_type: str, object_id: int, vector: bytes, dimension: int) -> int:
        """임베딩 벡터 저장"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO embeddings (object_type, object_id, vector, dimension)
                VALUES (?, ?, ?, ?)
            """, (object_type, object_id, vector, dimension))
            conn.commit()
            return cursor.lastrowid

    def insert_run(self, agent_name: str, input_summary: Optional[str] = None,
                   output_summary: Optional[str] = None, metadata: Optional[Dict] = None) -> int:
        """에이전트 실행 로그 저장"""
        metadata_json = json.dumps(metadata) if metadata else None
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO runs (agent_name, input_summary, output_summary, metadata)
                VALUES (?, ?, ?, ?)
            """, (agent_name, input_summary, output_summary, metadata_json))
            conn.commit()
            return cursor.lastrowid

    def update_run_status(self, run_id: int, status: str, finished_at: Optional[str] = None,
                         output_summary: Optional[str] = None):
        """실행 상태 업데이트"""
        with sqlite3.connect(self.db_path) as conn:
            if finished_at and output_summary:
                conn.execute("""
                    UPDATE runs SET status = ?, finished_at = ?, output_summary = ?
                    WHERE id = ?
                """, (status, finished_at, output_summary, run_id))
            else:
                conn.execute("""
                    UPDATE runs SET status = ? WHERE id = ?
                """, (status, run_id))
            conn.commit()

    # -----------------------
    # 조회 메서드
    # -----------------------
    def get_file_by_path(self, path: str) -> Optional[Dict[str, Any]]:
        """경로로 파일 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT * FROM files WHERE path = ?", (path,))
            row = cursor.fetchone()
            if row:
                return dict(zip([col[0] for col in cursor.description], row))
            return None

    def get_symbols_by_file(self, file_id: int) -> List[Dict[str, Any]]:
        """파일의 모든 심볼 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM symbols WHERE file_id = ? ORDER BY start_line
            """, (file_id,))
            return [dict(zip([col[0] for col in cursor.description], row))
                    for row in cursor.fetchall()]

    def get_symbols_by_type(self, symbol_type: str) -> List[Dict[str, Any]]:
        """타입별 심볼 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM symbols WHERE type = ? ORDER BY name
            """, (symbol_type,))
            return [dict(zip([col[0] for col in cursor.description], row))
                    for row in cursor.fetchall()]

    def get_chunks_by_symbol(self, symbol_id: int) -> List[Dict[str, Any]]:
        """심볼의 모든 청크 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM chunks WHERE symbol_id = ? ORDER BY id
            """, (symbol_id,))
            return [dict(zip([col[0] for col in cursor.description], row))
                    for row in cursor.fetchall()]

    def get_calls_by_caller(self, caller_id: int) -> List[Dict[str, Any]]:
        """호출하는 함수의 모든 호출 관계 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM calls WHERE caller_id = ? ORDER BY id
            """, (caller_id,))
            return [dict(zip([col[0] for col in cursor.description], row))
                    for row in cursor.fetchall()]

    def get_calls_by_callee(self, callee_id: int) -> List[Dict[str, Any]]:
        """호출되는 함수의 모든 호출 관계 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM calls WHERE callee_id = ? ORDER BY id
            """, (callee_id,))
            return [dict(zip([col[0] for col in cursor.description], row))
                    for row in cursor.fetchall()]

    def get_embedding(self, object_type: str, object_id: int) -> Optional[Dict[str, Any]]:
        """임베딩 벡터 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM embeddings WHERE object_type = ? AND object_id = ?
            """, (object_type, object_id))
            row = cursor.fetchone()
            if row:
                return dict(zip([col[0] for col in cursor.description], row))
            return None

    def get_latest_run(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """최신 실행 로그 조회"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM runs WHERE agent_name = ? ORDER BY started_at DESC LIMIT 1
            """, (agent_name,))
            row = cursor.fetchone()
            if row:
                return dict(zip([col[0] for col in cursor.description], row))
            return None

    # -----------------------
    # AST 데이터 저장 메서드
    # -----------------------
    def save_ast_data(self, ast_data: Dict[str, Any]) -> Dict[str, int]:
        """AST 데이터를 DB에 저장하고 ID 매핑 반환"""
        saved_ids = {}
        
        try:
            # 1. 파일 정보 저장
            file_info = ast_data.get("file", {})
            file_path = file_info.get("path", "")
            language = file_info.get("language", "unknown")
            llm_summary = file_info.get("llm_summary")
            
            file_id = self.insert_file(file_path, language, llm_summary)
            saved_ids["file_id"] = file_id
            
            # 2. 심볼들 저장
            symbols = ast_data.get("symbols", [])
            for symbol in symbols:
                symbol_id = self.insert_symbol(
                    file_id=file_id,
                    name=symbol.get("name", ""),
                    type_=symbol.get("type", ""),
                    language=language,
                    signature=symbol.get("signature", ""),
                    start_line=symbol.get("location", {}).get("start_line", 0),
                    end_line=symbol.get("location", {}).get("end_line", 0),
                    llm_summary=symbol.get("llm_summary"),
                    responsibility=symbol.get("responsibility"),
                    design_notes=symbol.get("design_notes"),
                    collaboration=symbol.get("collaboration")
                )
                
                # 심볼 ID 저장
                symbol_name = symbol.get("name", "")
                saved_ids[f"symbol_{symbol_name}"] = symbol_id
                
                # 3. 청크들 저장
                chunks = symbol.get("chunks", [])
                for chunk in chunks:
                    chunk_id = self.insert_chunk(
                        symbol_id=symbol_id,
                        chunk_type=chunk.get("type", "code"),
                        content=chunk.get("content", ""),
                        tokens=chunk.get("tokens")
                    )
                    saved_ids[f"chunk_{symbol_name}_{chunk.get('type', 'code')}"] = chunk_id
                
                # 4. 호출 관계 저장
                calls = symbol.get("calls", [])
                for call in calls:
                    try:
                        # 새로운 call 구조에 맞춰 처리
                        call_type = call.get("type", "direct")
                        confidence = call.get("confidence", 1.0)
                        
                        # 호출되는 대상 이름 추출
                        callee_name = None
                        if call_type == "function_call":
                            callee_name = call.get("name")
                        elif call_type == "method_call":
                            callee_name = call.get("method") or call.get("name")
                        elif call_type == "constructor_call":
                            callee_name = call.get("class_name") or call.get("name")
                        elif call_type == "chained_call":
                            callee_name = call.get("method") or call.get("name")
                        elif call_type == "nested_method_call":
                            callee_name = call.get("method") or call.get("name")
                        elif call_type == "attribute_call":
                            callee_name = call.get("method") or call.get("name")
                        elif call_type == "import_call":
                            callee_name = call.get("name")
                        elif call_type == "import_from_call":
                            callee_name = call.get("name")
                        elif call_type == "complex_call":
                            callee_name = call.get("name", "unknown_call")
                        
                        if callee_name:
                            # 같은 파일 내 호출 관계인지 확인
                            callee_symbol = next((s for s in symbols if s.get("name") == callee_name), None)
                            
                            if callee_symbol:
                                # 같은 파일 내 호출 관계 - 심볼 ID로 저장
                                callee_id = saved_ids.get(f"symbol_{callee_name}")
                                if callee_id:
                                    self.insert_call(
                                        caller_id=symbol_id,
                                        callee_id=callee_id,
                                        call_type=call_type,
                                        confidence=confidence,
                                        metadata=json.dumps(call)
                                    )
                                    logger.debug(f"같은 파일 내 호출 관계 저장: {symbol.get('name')} -> {callee_name}")
                                else:
                                    logger.warning(f"심볼 ID를 찾을 수 없음: {callee_name}")
                            else:
                                # 외부 모듈/라이브러리 호출 - 외부 호출로 저장
                                # 외부 호출을 위한 임시 심볼 생성
                                external_callee_id = self.insert_symbol(
                                    file_id=saved_ids["file_id"],
                                    name=callee_name,
                                    type_="external_call",
                                    language="python",
                                    signature=f"external:{callee_name}",
                                    start_line=0,
                                    end_line=0
                                )
                                
                                if external_callee_id:
                                    self.insert_call(
                                        caller_id=symbol_id,
                                        callee_id=external_callee_id,
                                        call_type=f"external_{call_type}",
                                        confidence=confidence,
                                        metadata=json.dumps(call)
                                    )
                                    logger.debug(f"외부 호출 관계 저장: {symbol.get('name')} -> {callee_name}")
                        
                    except Exception as e:
                        logger.warning(f"호출 관계 저장 실패: {e}")
                        continue
            
            logger.info(f"AST 데이터 저장 완료: 파일 {file_path}, 심볼 {len(symbols)}개")
            return saved_ids
            
        except Exception as e:
            logger.error(f"AST 데이터 저장 실패: {e}")
            raise

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
            
            # 청크 타입별 수
            cursor = conn.execute("SELECT chunk_type, COUNT(*) FROM chunks GROUP BY chunk_type")
            stats['chunks_by_type'] = dict(cursor.fetchall())
            
            # 최근 실행 상태
            cursor = conn.execute("""
                SELECT agent_name, status, started_at FROM runs 
                ORDER BY started_at DESC LIMIT 5
            """)
            stats['recent_runs'] = [dict(zip(['agent_name', 'status', 'started_at'], row))
                                    for row in cursor.fetchall()]
            
            return stats

    def search_symbols_fts(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """FTS를 사용한 심볼 검색"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # FTS 테이블이 없으면 contentless 버전으로 생성
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts 
                    USING fts5(
                        name, type, language, signature, llm_summary, responsibility, design_notes, collaboration,
                        content=''
                    )
                """)

                # 기존 심볼 데이터를 FTS 테이블에 인덱싱 (rowid 지정)
                conn.execute("""
                    INSERT OR REPLACE INTO symbols_fts(
                        rowid, name, type, language, signature, llm_summary, responsibility, design_notes, collaboration
                    )
                    SELECT id, name, type, language, signature, llm_summary, responsibility, design_notes, collaboration
                    FROM symbols
                """)

                # FTS 검색 실행 - 파라미터 바인딩 문제 해결
                search_query = f'"{query}"'  # 쿼리를 따옴표로 감싸기
                cursor = conn.execute("""
                    SELECT s.id, s.name, s.type, s.language, s.signature, s.llm_summary,
                           s.responsibility, s.design_notes, s.collaboration,
                           f.path as file_path,
                           bm25(symbols_fts) as rank
                    FROM symbols_fts
                    JOIN symbols s ON symbols_fts.rowid = s.id
                    JOIN files f ON s.file_id = f.id
                    WHERE symbols_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                """, (search_query, top_k))

                results = []
                for row in cursor.fetchall():
                    result = {
                        "id": row[0],
                        "name": row[1],
                        "type": row[2],
                        "language": row[3],
                        "signature": row[4],
                        "llm_summary": row[5],
                        "responsibility": row[6],
                        "design_notes": row[7],
                        "collaboration": row[8],
                        "file_path": row[9],
                        "rank": row[10]
                    }
                    results.append(result)

                logger.info(f"FTS 검색 완료: '{query}' -> {len(results)}개 결과")
                return results

        except Exception as e:
            logger.error(f"FTS 검색 실패: {e}")
            return []

    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """모든 임베딩 데이터 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, object_type, object_id, vector, dimension, created_at
                    FROM embeddings
                    ORDER BY id
                """)
                
                results = []
                for row in cursor.fetchall():
                    result = {
                        "id": row[0],
                        "object_type": row[1],
                        "object_id": row[2],
                        "vector": row[3],  # bytes
                        "dimension": row[4],
                        "created_at": row[5]
                    }
                    results.append(result)
                
                logger.info(f"임베딩 데이터 조회 완료: {len(results)}개")
                return results
                
        except Exception as e:
            logger.error(f"임베딩 데이터 조회 실패: {e}")
            return []
    
    def get_chunk_info(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """청크 정보 조회 (파일 경로 포함)"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # 먼저 chunks 테이블에서 symbol_id 확인
                cursor = conn.execute("SELECT symbol_id FROM chunks WHERE id = ?", (chunk_id,))
                chunk_row = cursor.fetchone()
                
                if not chunk_row:
                    logger.warning(f"청크 {chunk_id}를 찾을 수 없습니다")
                    return None
                
                symbol_id = chunk_row[0]
                
                # symbol_id가 문자열인 경우 (예: "1_SimpleClass")
                if isinstance(symbol_id, str) and '_' in symbol_id:
                    # 문자열에서 실제 심볼 이름 추출
                    actual_symbol_name = symbol_id.split('_', 1)[1]
                    
                    # 심볼 이름으로 검색
                    cursor = conn.execute("""
                        SELECT s.id, s.name, s.type, s.start_line, s.end_line,
                               f.path as file_path
                        FROM symbols s
                        JOIN files f ON s.file_id = f.id
                        WHERE s.name = ?
                    """, (actual_symbol_name,))
                    
                    symbol_row = cursor.fetchone()
                    if symbol_row:
                        # 청크 내용 조회
                        cursor = conn.execute("""
                            SELECT chunk_type, content, tokens, embedding_id
                            FROM chunks WHERE id = ?
                        """, (chunk_id,))
                        chunk_info = cursor.fetchone()
                        
                        if chunk_info:
                            result = {
                                "id": chunk_id,
                                "chunk_type": chunk_info[0],
                                "content": chunk_info[1],
                                "tokens": chunk_info[2],
                                "embedding_id": chunk_info[3],
                                "symbol_name": symbol_row[1],
                                "symbol_type": symbol_row[2],
                                "start_line": symbol_row[3],
                                "end_line": symbol_row[4],
                                "file_path": symbol_row[5]
                            }
                            return result
                
                # 기존 방식 (symbol_id가 숫자인 경우)
                cursor = conn.execute("""
                    SELECT c.id, c.chunk_type, c.content, c.tokens, c.embedding_id,
                           s.name as symbol_name, s.type as symbol_type,
                           s.start_line, s.end_line,
                           f.path as file_path
                    FROM chunks c
                    JOIN symbols s ON c.symbol_id = s.id
                    JOIN files f ON s.file_id = f.id
                    WHERE c.id = ?
                """, (chunk_id,))
                
                row = cursor.fetchone()
                if row:
                    result = {
                        "id": row[0],
                        "chunk_type": row[1],
                        "content": row[2],
                        "tokens": row[3],
                        "embedding_id": row[4],
                        "symbol_name": row[5],
                        "symbol_type": row[6],
                        "start_line": row[7],
                        "end_line": row[8],
                        "file_path": row[9]
                    }
                    return result
                else:
                    logger.warning(f"청크 {chunk_id} 정보를 찾을 수 없습니다")
                    return None
                    
        except Exception as e:
            logger.error(f"청크 정보 조회 실패: {e}")
            return None
    
    def get_database_stats(self) -> Dict[str, Any]:
        """데이터베이스 통계 정보 반환"""
        try:
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
                
                # 임베딩 수
                cursor = conn.execute("SELECT COUNT(*) FROM embeddings")
                stats['total_embeddings'] = cursor.fetchone()[0]
                
                # 언어별 분포
                cursor = conn.execute("SELECT language, COUNT(*) FROM files GROUP BY language")
                stats['languages'] = dict(cursor.fetchall())
                
                # 심볼 타입별 분포
                cursor = conn.execute("SELECT type, COUNT(*) FROM symbols GROUP BY type")
                stats['symbol_types'] = dict(cursor.fetchall())
                
                return stats
                
        except Exception as e:
            logger.error(f"데이터베이스 통계 조회 실패: {e}")
            return {}
    
    def get_all_files(self) -> List[Dict[str, Any]]:
        """모든 파일 정보 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, path, language, llm_summary, created_at
                    FROM files 
                    ORDER BY path
                """)
                
                files = []
                for row in cursor.fetchall():
                    files.append({
                        'id': row[0],
                        'path': row[1],
                        'language': row[2],
                        'llm_summary': row[3],
                        'created_at': row[4]
                    })
                
                return files
                
        except Exception as e:
            logger.error(f"전체 파일 조회 실패: {e}")
            return []
    
    def get_all_symbols(self) -> List[Dict[str, Any]]:
        """모든 심볼 정보 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT id, name, type, language, llm_summary, responsibility, 
                           design_notes, collaboration, start_line, end_line
                    FROM symbols 
                    ORDER BY name
                """)
                
                symbols = []
                for row in cursor.fetchall():
                    symbols.append({
                        'id': row[0],
                        'name': row[1],
                        'type': row[2],
                        'language': row[3],
                        'llm_summary': row[4],
                        'responsibility': row[5],
                        'design_notes': row[6],
                        'collaboration': row[7],
                        'start_line': row[8],
                        'end_line': row[9]
                    })
                
                return symbols
                
        except Exception as e:
            logger.error(f"전체 심볼 조회 실패: {e}")
            return []
    
    def get_symbol_by_id(self, symbol_id: int) -> Optional[Dict[str, Any]]:
        """ID로 특정 심볼 정보 조회"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT s.id, s.name, s.type, s.language, s.llm_summary, 
                           s.responsibility, s.design_notes, s.collaboration, 
                           s.start_line, s.end_line, s.signature,
                           f.path as file_path
                    FROM symbols s
                    JOIN files f ON s.file_id = f.id
                    WHERE s.id = ?
                """, (symbol_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'id': row[0],
                        'name': row[1],
                        'type': row[2],
                        'language': row[3],
                        'llm_summary': row[4],
                        'responsibility': row[5],
                        'design_notes': row[6],
                        'collaboration': row[7],
                        'start_line': row[8],
                        'end_line': row[9],
                        'signature': row[10],
                        'file_path': row[11]
                    }
                return None
                
        except Exception as e:
            logger.error(f"심볼 {symbol_id} 조회 실패: {e}")
            return None
    
    def update_chunk_embedding(self, chunk_id: int, embedding_id: int):
        """청크의 embedding_id 업데이트"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE chunks 
                    SET embedding_id = ? 
                    WHERE id = ?
                """, (embedding_id, chunk_id))
                conn.commit()
                logger.info(f"청크 {chunk_id}의 embedding_id 업데이트 완료: {embedding_id}")
                return True
        except Exception as e:
            logger.error(f"청크 {chunk_id}의 embedding_id 업데이트 실패: {e}")
            return False
    
    def search_symbols_by_name(self, symbol_name: str) -> List[Dict[str, Any]]:
        """심볼 이름으로 검색"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT s.id, s.file_id, s.name, s.type, s.language, s.signature,
                           s.start_line, s.end_line, f.path as file_path
                    FROM symbols s
                    JOIN files f ON s.file_id = f.id
                    WHERE s.name = ?
                    ORDER BY s.name, f.path
                """, (symbol_name,))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'id': row[0],
                        'file_id': row[1],
                        'name': row[2],
                        'type': row[3],
                        'language': row[4],
                        'signature': row[5],
                        'start_line': row[6],
                        'end_line': row[7],
                        'file_path': row[8]
                    })
                
                logger.info(f"심볼 '{symbol_name}' 검색 완료: {len(results)}개 발견")
                return results
                
        except Exception as e:
            logger.error(f"심볼 이름 검색 실패: {e}")
            return []
    
    def search_symbols_by_name_and_file(self, symbol_name: str, file_path: str) -> List[Dict[str, Any]]:
        """심볼 이름과 파일 경로로 검색"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT s.id, s.file_id, s.name, s.type, s.language, s.signature,
                           s.start_line, s.end_line, f.path as file_path
                    FROM symbols s
                    JOIN files f ON s.file_id = f.id
                    WHERE s.name = ? AND f.path LIKE ?
                    ORDER BY s.name, f.path
                """, (symbol_name, f"%{file_path}%"))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'id': row[0],
                        'file_id': row[1],
                        'name': row[2],
                        'type': row[3],
                        'language': row[4],
                        'signature': row[5],
                        'start_line': row[6],
                        'end_line': row[7],
                        'file_path': row[8]
                    })
                
                logger.info(f"심볼 '{symbol_name}' (파일: {file_path}) 검색 완료: {len(results)}개 발견")
                return results
                
        except Exception as e:
            logger.error(f"심볼 이름 및 파일 검색 실패: {e}")
            return []