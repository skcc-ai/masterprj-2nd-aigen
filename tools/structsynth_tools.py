from typing import Any, Dict, List, Optional

from ._compat import tool

from common.store.sqlite_store import SQLiteStore


def _get_agent(repo_path: str = ".", artifacts_dir: str = "./artifacts", data_dir: str = "./data"):
    # Lazy import to avoid FAISS/NumPy import if not needed
    from agents.structsynth.agent import StructSynthAgent  # type: ignore
    return StructSynthAgent(repo_path=repo_path, artifacts_dir=artifacts_dir, data_dir=data_dir)


def _get_store(data_dir: str = "./data") -> SQLiteStore:
    return SQLiteStore(db_path=f"{data_dir}/structsynth_code.db")


@tool("search_symbols_semantic", return_direct=False)
def search_symbols_semantic(query: str, top_k: int = 10, repo_path: str = ".", data_dir: str = "./data") -> List[Dict[str, Any]]:
    """심볼을 의미 기반으로 검색합니다. FAISS/임베딩 미사용 시 FTS로 폴백합니다."""
    try:
        agent = _get_agent(repo_path=repo_path, data_dir=data_dir)
        return agent.search_symbols(query=query, top_k=top_k)
    except Exception:
        store = _get_store(data_dir=data_dir)
        return store.search_symbols_fts(query=query, top_k=top_k)


@tool("search_symbols_fts", return_direct=False)
def search_symbols_fts(query: str, top_k: int = 10, data_dir: str = "./data") -> List[Dict[str, Any]]:
    """BM25 FTS로 심볼을 텍스트 검색합니다."""
    store = _get_store(data_dir=data_dir)
    return store.search_symbols_fts(query=query, top_k=top_k)


@tool("get_symbol", return_direct=False)
def get_symbol(symbol_id: int, data_dir: str = "./data") -> Optional[Dict[str, Any]]:
    """심볼 ID로 단일 심볼 메타데이터를 조회합니다."""
    store = _get_store(data_dir=data_dir)
    return store.get_symbol_by_id(symbol_id)


@tool("get_symbol_by_name", return_direct=False)
def get_symbol_by_name(symbol_name: str, data_dir: str = "./data") -> Optional[Dict[str, Any]]:
    """이름으로 심볼 정보를 조회합니다."""
    import sqlite3
    store = _get_store(data_dir=data_dir)
    
    with sqlite3.connect(store.db_path) as conn:
        cursor = conn.execute("""
            SELECT * FROM symbols WHERE name = ? LIMIT 1
        """, (symbol_name,))
        row = cursor.fetchone()
        if row:
            columns = [description[0] for description in cursor.description]
            return dict(zip(columns, row))
    return None


@tool("get_chunks_by_symbol", return_direct=False)
def get_chunks_by_symbol(symbol_id: int, data_dir: str = "./data") -> List[Dict[str, Any]]:
    """심볼 ID로 청크 목록을 조회합니다."""
    store = _get_store(data_dir=data_dir)
    return store.get_chunks_by_symbol(symbol_id)


@tool("get_chunk", return_direct=False)
def get_chunk(chunk_id: int, data_dir: str = "./data") -> Optional[Dict[str, Any]]:
    """청크 ID로 청크 상세 정보를 조회합니다."""
    store = _get_store(data_dir=data_dir)
    return store.get_chunk_info(chunk_id)


@tool("get_calls_from", return_direct=False)
def get_calls_from(caller_id: int, data_dir: str = "./data") -> List[Dict[str, Any]]:
    """caller 심볼에서 나가는 호출 관계를 조회합니다."""
    store = _get_store(data_dir=data_dir)
    return store.get_calls_by_caller(caller_id)


@tool("get_calls_to", return_direct=False)
def get_calls_to(callee_id: int, data_dir: str = "./data") -> List[Dict[str, Any]]:
    """callee 심볼로 들어오는 호출 관계를 조회합니다."""
    store = _get_store(data_dir=data_dir)
    return store.get_calls_by_callee(callee_id)


@tool("get_call_graph", return_direct=False)
def get_call_graph(symbol_id: int, depth: int = 2, data_dir: str = "./data") -> Dict[str, Any]:
    """심볼의 호출 그래프를 깊이 지정하여 조회합니다."""
    store = _get_store(data_dir=data_dir)
    
    def build_graph(current_id: int, current_depth: int, visited: set) -> Dict[str, Any]:
        if current_depth <= 0 or current_id in visited:
            return {"symbol_id": current_id, "calls": []}
        
        visited.add(current_id)
        symbol_info = store.get_symbol_by_id(current_id)
        calls_out = store.get_calls_by_caller(current_id)
        calls_in = store.get_calls_by_callee(current_id)
        
        graph_node = {
            "symbol_id": current_id,
            "symbol_info": symbol_info,
            "calls_out": [],
            "calls_in": calls_in,
            "depth": current_depth
        }
        
        # 나가는 호출들을 재귀적으로 탐색
        for call in calls_out:
            callee_id = call.get("callee_id")
            if callee_id and callee_id not in visited:
                sub_graph = build_graph(callee_id, current_depth - 1, visited.copy())
                graph_node["calls_out"].append({
                    "call_info": call,
                    "target_graph": sub_graph
                })
        
        return graph_node
    
    return build_graph(symbol_id, depth, set())


@tool("analyze_call_patterns", return_direct=False)
def analyze_call_patterns(data_dir: str = "./data") -> Dict[str, Any]:
    """전체 코드베이스의 호출 패턴을 분석합니다."""
    import sqlite3
    store = _get_store(data_dir=data_dir)
    
    with sqlite3.connect(store.db_path) as conn:
        # 호출 타입별 통계
        cursor = conn.execute("""
            SELECT call_type, COUNT(*) as count
            FROM calls
            GROUP BY call_type
            ORDER BY count DESC
        """)
        call_type_stats = [{"call_type": row[0], "count": row[1]} for row in cursor.fetchall()]
        
        # 가장 많이 호출되는 함수들 (인기도)
        cursor = conn.execute("""
            SELECT s.name, s.type, COUNT(c.id) as call_count
            FROM symbols s
            LEFT JOIN calls c ON s.id = c.callee_id
            GROUP BY s.id, s.name, s.type
            HAVING call_count > 0
            ORDER BY call_count DESC
            LIMIT 10
        """)
        most_called = [{"name": row[0], "type": row[1], "call_count": row[2]} for row in cursor.fetchall()]
        
        # 가장 많이 호출하는 함수들 (복잡도)
        cursor = conn.execute("""
            SELECT s.name, s.type, COUNT(c.id) as calls_made
            FROM symbols s
            LEFT JOIN calls c ON s.id = c.caller_id
            GROUP BY s.id, s.name, s.type
            HAVING calls_made > 0
            ORDER BY calls_made DESC
            LIMIT 10
        """)
        most_calling = [{"name": row[0], "type": row[1], "calls_made": row[2]} for row in cursor.fetchall()]
        
        # 외부 호출 통계
        cursor = conn.execute("""
            SELECT call_type, COUNT(*) as count
            FROM calls
            WHERE call_type LIKE 'external_%' OR call_type LIKE 'import_%'
            GROUP BY call_type
            ORDER BY count DESC
        """)
        external_calls = [{"call_type": row[0], "count": row[1]} for row in cursor.fetchall()]
    
    return {
        "call_type_statistics": call_type_stats,
        "most_called_symbols": most_called,
        "most_calling_symbols": most_calling,
        "external_call_patterns": external_calls,
        "total_calls": sum(stat["count"] for stat in call_type_stats)
    }


@tool("list_files", return_direct=False)
def list_files(data_dir: str = "./data") -> List[Dict[str, Any]]:
    """모든 파일 목록을 조회합니다."""
    store = _get_store(data_dir=data_dir)
    return store.get_all_files()


@tool("list_symbols", return_direct=False)
def list_symbols(type: Optional[str] = None, language: Optional[str] = None, limit: int = 200, data_dir: str = "./data") -> List[Dict[str, Any]]:
    """모든 심볼을 조회하고 선택적으로 타입/언어로 필터링합니다."""
    store = _get_store(data_dir=data_dir)
    symbols = store.get_all_symbols()
    if type:
        symbols = [s for s in symbols if s.get("type") == type]
    if language:
        symbols = [s for s in symbols if s.get("language") == language]
    return symbols[: max(0, limit)]


@tool("get_database_stats", return_direct=False)
def get_database_stats(data_dir: str = "./data") -> Dict[str, Any]:
    """데이터베이스 통계를 반환합니다."""
    store = _get_store(data_dir=data_dir)
    return store.get_database_stats()


@tool("get_vector_store_stats", return_direct=False)
def get_vector_store_stats(repo_path: str = ".", data_dir: str = "./data") -> Dict[str, Any]:
    """FAISS 벡터 스토어 통계를 반환합니다. 미설치 시 기본 정보를 반환합니다."""
    try:
        agent = _get_agent(repo_path=repo_path, data_dir=data_dir)
        return agent.get_vector_store_stats()
    except Exception as e:
        return {"error": "vector_store_unavailable", "detail": str(e), "total_vectors": 0}


@tool("get_analysis_summary", return_direct=False)
def get_analysis_summary(repo_path: str = ".", data_dir: str = "./data") -> Dict[str, Any]:
    """최근 수행된 분석 요약을 반환합니다. 에이전트 생성 실패 시 DB 통계를 반환합니다."""
    try:
        agent = _get_agent(repo_path=repo_path, data_dir=data_dir)
        return agent.get_analysis_summary()
    except Exception as e:
        store = _get_store(data_dir=data_dir)
        stats = store.get_database_stats()
        return {"status": "agent_unavailable", "detail": str(e), "db_stats": stats}


@tool("run_repository_analysis", return_direct=False)
def run_repository_analysis(repo_path: str, artifacts_dir: str = "./artifacts", data_dir: str = "./data") -> Dict[str, Any]:
    """저장소 전체 분석을 실행합니다. 벡터 스토어/LLM 미설치 시 에러를 반환합니다."""
    try:
        agent = _get_agent(repo_path=repo_path, artifacts_dir=artifacts_dir, data_dir=data_dir)
        return agent.analyze_repository()
    except Exception as e:
        return {"error": "analysis_unavailable", "detail": str(e), "repo_path": repo_path}


@tool("get_latest_run", return_direct=False)
def get_latest_run(agent_name: str = "StructSynth", data_dir: str = "./data") -> Optional[Dict[str, Any]]:
    """지정된 에이전트의 최신 실행 정보를 반환합니다."""
    store = _get_store(data_dir=data_dir)
    return store.get_latest_run(agent_name)


