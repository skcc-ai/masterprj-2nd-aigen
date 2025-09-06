import os
from typing import Any, Dict, Optional

from ._compat import tool

from agents.structsynth.llm_analyzer import LLMAnalyzer
from common.store.sqlite_store import SQLiteStore


def _ensure_env():
    required = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
    ]
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise RuntimeError(f"Azure OpenAI 환경변수 누락: {missing}")


def _get_analyzer() -> LLMAnalyzer:
    _ensure_env()
    return LLMAnalyzer()


@tool("analyze_file_llm", return_direct=False)
def analyze_file_llm(file_path: str, data_dir: str = "./data") -> Dict[str, Any]:
    """파일 단위 LLM 분석을 수행합니다."""
    analyzer = _get_analyzer()
    # 파일 컨텍스트 로딩은 analyzer 내부에서 프롬프트에 포함되므로 빈 컨텍스트 전달 가능
    from agents.structsynth.parser import CodeParser

    parser = CodeParser()
    ast_data = parser.parse_file(file_path)
    if not ast_data:
        return {"error": "파일 파싱 실패", "file_path": file_path}
    # 파일 전체 컨텍스트를 위해 원본 파일을 읽음
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            file_context = f.read()
    except Exception:
        file_context = ""
    return analyzer.analyze_file(ast_data, file_context)


@tool("analyze_symbol_llm", return_direct=False)
def analyze_symbol_llm(symbol_id: int, data_dir: str = "./data") -> Dict[str, Any]:
    """심볼 단위 LLM 분석을 수행합니다(함수/클래스 자동 구분)."""
    analyzer = _get_analyzer()
    store = SQLiteStore(db_path=f"{data_dir}/structsynth_code.db")
    symbol = store.get_symbol_by_id(symbol_id)
    if not symbol:
        return {"error": "심볼을 찾을 수 없음", "symbol_id": symbol_id}
    file_path = symbol.get("file_path", "")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            file_context = f.read()
    except Exception:
        file_context = ""
    # 간단한 데이터 구조로 매핑
    symbol_data = {
        "type": symbol.get("type"),
        "name": symbol.get("name"),
        "signature": symbol.get("signature"),
        "location": {"start_line": symbol.get("start_line"), "end_line": symbol.get("end_line")},
        "metadata": {},
        "body": {},
    }
    if symbol.get("type") == "class":
        return analyzer.analyze_class(symbol_data, file_context)
    else:
        return analyzer.analyze_function(symbol_data, file_context)


@tool("analyze_chunk_llm", return_direct=False)
def analyze_chunk_llm(chunk_id: Optional[int] = None, symbol_id: Optional[int] = None, data_dir: str = "./data") -> Dict[str, Any]:
    """청크 단위 LLM 요약을 수행합니다.
    - chunk_id가 없으면 symbol_id의 첫 번째 청크를 사용합니다.
    - 둘 다 없거나 비어 있으면 DB에서 첫 가용 청크를 자동 선택합니다.
    """
    analyzer = _get_analyzer()
    store = SQLiteStore(db_path=f"{data_dir}/structsynth_code.db")

    resolved_chunk_id: Optional[int] = chunk_id
    try:
        # 1) chunk_id 직접 사용 가능
        if resolved_chunk_id is None:
            # 2) symbol_id 기반으로 유도
            if symbol_id is not None:
                chunks = store.get_chunks_by_symbol(int(symbol_id))
                if isinstance(chunks, list) and chunks:
                    resolved_chunk_id = int(chunks[0].get("id"))
            # 3) DB에서 아무 청크나 선택
            if resolved_chunk_id is None:
                symbols = store.get_all_symbols()
                for s in symbols:
                    sid = s.get("id")
                    if sid is None:
                        continue
                    chunks = store.get_chunks_by_symbol(int(sid))
                    if isinstance(chunks, list) and chunks:
                        resolved_chunk_id = int(chunks[0].get("id"))
                        break
    except Exception:
        pass

    if resolved_chunk_id is None:
        return {"error": "no_chunk_available", "detail": "chunk_id와 symbol_id로도 청크를 찾지 못했습니다."}

    chunk = store.get_chunk_info(resolved_chunk_id)
    if not chunk:
        return {"error": "청크를 찾을 수 없음", "chunk_id": resolved_chunk_id}
    # LLMAnalyzer.analyze_chunk에 필요한 구조로 맞춤
    chunk_payload = {
        "symbol_type": chunk.get("symbol_type", "unknown"),
        "symbol_name": chunk.get("symbol_name", "unknown"),
        "content": chunk.get("content", ""),
    }
    return analyzer.analyze_chunk(chunk_payload)


