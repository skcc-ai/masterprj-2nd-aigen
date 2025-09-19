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


@tool("get_source_code", return_direct=False)
def get_source_code(file_path: str, start_line: int = None, end_line: int = None, data_dir: str = "./data") -> Dict[str, Any]:
    """파일의 실제 소스코드를 가져옵니다.
    
    Args:
        file_path: 소스코드 파일 경로
        start_line: 시작 라인 번호 (1부터 시작, None이면 전체 파일)
        end_line: 끝 라인 번호 (None이면 파일 끝까지)
        data_dir: 데이터 디렉토리 경로
    
    Returns:
        실제 소스코드 내용과 메타데이터
    """
    try:
        # 절대 경로 처리
        if not os.path.isabs(file_path):
            # 상대 경로인 경우 프로젝트 루트를 기준으로 처리
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path = os.path.join(project_root, file_path)
        else:
            full_path = file_path
        
        # 파일 존재 확인
        if not os.path.exists(full_path):
            return {
                "error": "파일을 찾을 수 없습니다",
                "file_path": file_path,
                "full_path": full_path
            }
        
        # 파일 읽기
        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 라인 범위 처리
        total_lines = len(lines)
        actual_start = max(1, start_line or 1)
        actual_end = min(total_lines, end_line or total_lines)
        
        # 라인 번호 검증
        if actual_start > total_lines:
            return {
                "error": "시작 라인이 파일 총 라인 수를 초과합니다",
                "file_path": file_path,
                "total_lines": total_lines,
                "requested_start": start_line
            }
        
        # 지정된 라인 범위의 코드 추출
        selected_lines = lines[actual_start-1:actual_end]
        source_code = ''.join(selected_lines)
        
        # 라인 번호와 함께 코드 생성
        numbered_lines = []
        for i, line in enumerate(selected_lines, start=actual_start):
            numbered_lines.append(f"{i:4d}| {line.rstrip()}")
        
        numbered_source = '\n'.join(numbered_lines)
        
        return {
            "success": True,
            "file_path": file_path,
            "full_path": full_path,
            "start_line": actual_start,
            "end_line": actual_end,
            "total_lines": total_lines,
            "source_code": source_code,
            "numbered_source": numbered_source,
            "line_count": len(selected_lines)
        }
        
    except Exception as e:
        return {
            "error": f"파일 읽기 실패: {str(e)}",
            "file_path": file_path,
            "exception": str(e)
        }


@tool("analyze_source_code_with_llm", return_direct=False) 
def analyze_source_code_with_llm(file_path: str, start_line: int = None, end_line: int = None, 
                                question: str = "", data_dir: str = "./data") -> Dict[str, Any]:
    """실제 소스코드를 가져와서 LLM으로 분석합니다.
    
    Args:
        file_path: 소스코드 파일 경로
        start_line: 시작 라인 번호 (1부터 시작)
        end_line: 끝 라인 번호
        question: 소스코드에 대한 구체적인 질문
        data_dir: 데이터 디렉토리 경로
    
    Returns:
        LLM이 분석한 소스코드 설명
    """
    try:
        # 먼저 소스코드 가져오기
        source_result = get_source_code(file_path, start_line, end_line, data_dir)
        
        if "error" in source_result:
            return source_result
        
        # LLM 분석 수행
        analyzer = _get_analyzer()
        
        # 분석용 프롬프트 구성
        source_code = source_result["source_code"]
        numbered_source = source_result["numbered_source"]
        
        analysis_prompt = f"""
다음 소스코드를 분석해주세요:

파일 경로: {file_path}
라인 범위: {source_result['start_line']}-{source_result['end_line']} (총 {source_result['line_count']}줄)

소스코드:
```
{numbered_source}
```

{'사용자 질문: ' + question if question else ''}

분석 요청사항:
1. 코드의 주요 기능과 목적
2. 중요한 변수와 함수들의 역할
3. 코드의 실행 흐름
4. 잠재적인 문제점이나 개선점
5. 다른 코드와의 연관성 (import, 호출 관계 등)

한국어로 상세하고 구체적으로 분석해주세요.
"""
        
        # LLM 호출
        from openai import AzureOpenAI
        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        )
        
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o"),
            messages=[
                {
                    "role": "system", 
                    "content": "당신은 코드 분석 전문가입니다. 제공된 소스코드를 정확하고 상세하게 분석하여 설명해주세요."
                },
                {
                    "role": "user", 
                    "content": analysis_prompt
                }
            ],
            temperature=0.1,
            max_tokens=2000
        )
        
        analysis = response.choices[0].message.content
        
        return {
            "success": True,
            "file_path": file_path,
            "line_range": f"{source_result['start_line']}-{source_result['end_line']}",
            "source_code": source_code,
            "numbered_source": numbered_source,
            "analysis": analysis,
            "question": question,
            "metadata": {
                "total_lines": source_result['total_lines'],
                "analyzed_lines": source_result['line_count']
            }
        }
        
    except Exception as e:
        return {
            "error": f"소스코드 분석 실패: {str(e)}",
            "file_path": file_path,
            "exception": str(e)
        }


@tool("get_function_source", return_direct=False)
def get_function_source(symbol_name: str, file_path: str = None, data_dir: str = "./data") -> Dict[str, Any]:
    """데이터베이스에서 함수/클래스 정보를 찾아 실제 소스코드를 가져옵니다.
    
    Args:
        symbol_name: 함수나 클래스 이름
        file_path: 특정 파일 경로 (None이면 모든 파일에서 검색)
        data_dir: 데이터 디렉토리 경로
    
    Returns:
        함수/클래스의 실제 소스코드와 메타데이터
    """
    try:
        # SQLite에서 심볼 정보 검색
        store = SQLiteStore(db_path=f"{data_dir}/structsynth_code.db")
        
        # 심볼 검색
        if file_path:
            symbols = store.search_symbols_by_name_and_file(symbol_name, file_path)
        else:
            symbols = store.search_symbols_by_name(symbol_name)
        
        if not symbols:
            return {
                "error": "심볼을 찾을 수 없습니다",
                "symbol_name": symbol_name,
                "file_path": file_path
            }
        
        results = []
        for symbol in symbols:
            symbol_file_path = symbol.get("file_path")
            start_line = symbol.get("start_line")
            end_line = symbol.get("end_line")
            symbol_type = symbol.get("type")
            signature = symbol.get("signature")
            
            # 실제 소스코드 가져오기
            source_result = get_source_code(symbol_file_path, start_line, end_line, data_dir)
            
            if "error" not in source_result:
                results.append({
                    "symbol_id": symbol.get("id"),
                    "symbol_name": symbol_name,
                    "symbol_type": symbol_type,
                    "signature": signature,
                    "file_path": symbol_file_path,
                    "line_range": f"{start_line}-{end_line}",
                    "source_code": source_result["source_code"],
                    "numbered_source": source_result["numbered_source"],
                    "metadata": symbol
                })
            else:
                results.append({
                    "symbol_id": symbol.get("id"),
                    "symbol_name": symbol_name,
                    "error": source_result["error"],
                    "file_path": symbol_file_path
                })
        
        return {
            "success": True,
            "symbol_name": symbol_name,
            "found_count": len(results),
            "results": results
        }
        
    except Exception as e:
        return {
            "error": f"함수 소스코드 검색 실패: {str(e)}",
            "symbol_name": symbol_name,
            "exception": str(e)
        }


