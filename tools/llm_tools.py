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
def analyze_chunk_llm(chunk_id: int, data_dir: str = "./data") -> Dict[str, Any]:
    """청크 단위 LLM 요약을 수행합니다."""
    analyzer = _get_analyzer()
    store = SQLiteStore(db_path=f"{data_dir}/structsynth_code.db")
    chunk = store.get_chunk_info(chunk_id)
    if not chunk:
        return {"error": "청크를 찾을 수 없음", "chunk_id": chunk_id}
    # LLMAnalyzer.analyze_chunk에 필요한 구조로 맞춤
    chunk_payload = {
        "symbol_type": chunk.get("symbol_type", "unknown"),
        "symbol_name": chunk.get("symbol_name", "unknown"),
        "content": chunk.get("content", ""),
    }
    return analyzer.analyze_chunk(chunk_payload)


