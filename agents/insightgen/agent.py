import json
from typing import Any, Dict, List

from tools import get_tools
from agents.insightgen.planner import LLMPlanner
import os
import shutil


class InsightGenAgent:
    """
    간단한 InsightGen 에이전트: DB 규모/구조를 바탕으로 산출물 3가지를 선택하고,
    각 산출물별 사용할 tool과 근거를 결정한다. 실제 데이터는 tools를 호출해 요약을 생성.
    """

    def __init__(self, data_dir: str = "./data", repo_path: str = "."):
        self.data_dir = data_dir
        self.repo_path = repo_path
        # Tools 중 repo_path를 필요로 하는 호출에서 사용할 실제 대상 경로
        # (StructSynth 최신 실행의 repo_path가 있으면 그것을 우선 사용)
        self._effective_repo_path: str | None = None
        self.tools = {t.name: t for t in get_tools()}
        self._llm_planner = None

    def _invoke(self, name: str, payload: Dict[str, Any]) -> Any:
        tool = self.tools.get(name)
        if not tool:
            return {"error": f"tool_not_found:{name}"}
        try:
            return tool.invoke(payload)
        except Exception as e:
            return {"error": str(e), "tool": name}

    def _normalize_params(
        self,
        tool_name: str,
        params: Dict[str, Any],
        symbols_snapshot: list | None = None,
    ) -> Dict[str, Any]:
        """LLM이 제안한 파라미터 키/타입을 각 tool 시그니처에 맞게 정규화한다.
        - get_calls_from: caller_id 필요
        - get_calls_to: callee_id 필요
        - analyze_chunk_llm: chunk_id 필요 (symbol_id만 있으면 첫 청크를 조회하여 사용)
        - 공통: *_id 값은 int 캐스팅 시도
        """

        def _as_int(value: Any) -> Any:
            try:
                return int(value)
            except Exception:
                return value

        normalized = dict(params or {})

        # get_calls_from: map symbol_id/id/caller -> caller_id
        if tool_name == "get_calls_from":
            if "caller_id" not in normalized:
                for alt in ("symbol_id", "id", "caller"):
                    if alt in normalized:
                        normalized["caller_id"] = normalized.pop(alt)
                        break
            if "caller_id" in normalized:
                normalized["caller_id"] = _as_int(normalized["caller_id"])

        # get_calls_to: map symbol_id/id/callee -> callee_id
        if tool_name == "get_calls_to":
            if "callee_id" not in normalized:
                for alt in ("symbol_id", "id", "callee"):
                    if alt in normalized:
                        normalized["callee_id"] = normalized.pop(alt)
                        break
            if "callee_id" in normalized:
                normalized["callee_id"] = _as_int(normalized["callee_id"])

        # analyze_chunk_llm: need chunk_id; allow id/chunk/chunkId, or derive from symbol_id
        if tool_name == "analyze_chunk_llm":
            # rename to chunk_id if needed
            if "chunk_id" not in normalized:
                for alt in ("id", "chunk", "chunkId"):
                    if alt in normalized:
                        normalized["chunk_id"] = normalized.pop(alt)
                        break
            # derive from symbol_id if still missing
            if "chunk_id" not in normalized and "symbol_id" in normalized:
                symbol_id_val = _as_int(normalized.get("symbol_id"))
                # 첫 번째 청크를 조회하여 사용
                chunks = self._invoke("get_chunks_by_symbol", {"symbol_id": symbol_id_val, "data_dir": self.data_dir})
                if isinstance(chunks, list) and chunks:
                    normalized["chunk_id"] = chunks[0].get("id")
                # 사용 후 불필요 키 제거
                normalized.pop("symbol_id", None)
            if "chunk_id" in normalized:
                normalized["chunk_id"] = _as_int(normalized["chunk_id"])

        # analyze_file_llm: ensure file_path; if missing, derive from first file
        if tool_name == "analyze_file_llm":
            if "file_path" not in normalized or not normalized.get("file_path"):
                try:
                    files = self._invoke("list_files", {"data_dir": self.data_dir})
                    if isinstance(files, list) and files:
                        # 우선순위: .py > .js > .java > 기타
                        def _rank(f):
                            p = (f or {}).get("path") or (f or {}).get("file", {}).get("path")
                            if not p:
                                return 99
                            if str(p).endswith(".py"):
                                return 0
                            if str(p).endswith(".js") or str(p).endswith(".ts"):
                                return 1
                            if str(p).endswith(".java"):
                                return 2
                            return 3
                        files_sorted = sorted(files, key=_rank)
                        picked = files_sorted[0]
                        normalized["file_path"] = picked.get("path") or picked.get("file", {}).get("path")
                except Exception:
                    pass

        # get_symbol/get_chunks_by_symbol: ensure int id
        if tool_name in ("get_symbol", "get_chunks_by_symbol"):
            if "symbol_id" in normalized:
                normalized["symbol_id"] = _as_int(normalized["symbol_id"])

        # list_symbols: rename limit/top_k to limit
        if tool_name == "list_symbols":
            if "limit" not in normalized:
                if "top_k" in normalized:
                    normalized["limit"] = normalized.pop("top_k")
            if "limit" in normalized:
                normalized["limit"] = _as_int(normalized["limit"])

        return normalized

    def _safe_invoke_with_fallback(
        self,
        tool_name: str,
        params: Dict[str, Any],
        symbols: list | None,
        tool_call_log: list,
    ) -> Any:
        """Invoke tool with normalized params and apply pragmatic fallbacks if result is empty.
        Adds detailed entries to tool_call_log for debugging.
        """
        norm_params = self._normalize_params(tool_name, params, symbols)

        # 허용 파라미터 화이트리스트 구성
        allowed_map = {
            "search_symbols_semantic": {"query", "top_k", "repo_path", "data_dir"},
            "search_symbols_fts": {"query", "top_k", "data_dir"},
            "get_symbol": {"symbol_id", "data_dir"},
            "get_chunks_by_symbol": {"symbol_id", "data_dir"},
            "get_chunk": {"chunk_id", "data_dir"},
            "get_calls_from": {"caller_id", "data_dir"},
            "get_calls_to": {"callee_id", "data_dir"},
            "list_files": {"data_dir"},
            "list_symbols": {"type", "language", "limit", "data_dir"},
            "get_database_stats": {"data_dir"},
            "get_vector_store_stats": {"repo_path", "data_dir"},
            "get_analysis_summary": {"repo_path", "data_dir"},
            "run_repository_analysis": {"repo_path", "artifacts_dir", "data_dir"},
            "get_latest_run": {"agent_name", "data_dir"},
            "analyze_file_llm": {"file_path", "data_dir"},
            "analyze_symbol_llm": {"symbol_id", "data_dir"},
            "analyze_chunk_llm": {"chunk_id", "data_dir"},
        }

        # repo_path는 지원 도구에만 유지
        tools_need_repo_path = {
            "search_symbols_semantic",
            "get_vector_store_stats",
            "get_analysis_summary",
            "run_repository_analysis",
        }

        # 불필요 키 제거 및 기본 키 추가(data_dir)
        filtered = {k: v for k, v in (norm_params or {}).items() if k in allowed_map.get(tool_name, set())}
        filtered["data_dir"] = self.data_dir
        if tool_name in tools_need_repo_path:
            filtered["repo_path"] = self._effective_repo_path or self.repo_path

        base_payload = filtered
        result = self._invoke(tool_name, base_payload)
        tool_call_log.append({
            "tool": tool_name,
            "params": base_payload,
            "result_preview": (result if isinstance(result, dict) else (result[:3] if isinstance(result, list) else result)),
        })

        # If result is empty and we can fallback, try once
        def _is_empty(res: Any) -> bool:
            if res is None:
                return True
            if isinstance(res, list):
                return len(res) == 0
            if isinstance(res, dict) and res.get("error"):
                return False  # surface error instead of fallback
            return False

        # Treat certain tool errors as fallback triggers
        error_like = isinstance(result, dict) and bool(result.get("error"))
        if not _is_empty(result) and not (
            error_like and tool_name in ("analyze_file_llm", "analyze_chunk_llm")
        ):
            return result

        # Fallbacks for specific tools
        if tool_name in ("get_calls_from", "get_calls_to") and isinstance(symbols, list):
            # scan up to first 50 symbols to find one with calls
            limit_scan = min(50, len(symbols))
            if tool_name == "get_calls_from":
                for s in symbols[:limit_scan]:
                    sid = s.get("id")
                    if sid is None:
                        continue
                    res = self._invoke("get_calls_from", {"caller_id": int(sid), "data_dir": self.data_dir})
                    if isinstance(res, list) and res:
                        tool_call_log.append({
                            "tool": tool_name,
                            "fallback_applied": True,
                            "fallback_params": {"caller_id": int(sid)},
                            "result_preview": res[:3],
                        })
                        return res
            else:
                for s in symbols[:limit_scan]:
                    sid = s.get("id")
                    if sid is None:
                        continue
                    res = self._invoke("get_calls_to", {"callee_id": int(sid), "data_dir": self.data_dir})
                    if isinstance(res, list) and res:
                        tool_call_log.append({
                            "tool": tool_name,
                            "fallback_applied": True,
                            "fallback_params": {"callee_id": int(sid)},
                            "result_preview": res[:3],
                        })
                        return res

        if tool_name == "analyze_chunk_llm":
            # If chunk_id invalid or missing, try to derive from first symbol having chunks
            # Attempt scan of symbols to get a chunk
            if isinstance(symbols, list):
                limit_scan = min(50, len(symbols))
                for s in symbols[:limit_scan]:
                    sid = s.get("id")
                    if sid is None:
                        continue
                    chunks = self._invoke("get_chunks_by_symbol", {"symbol_id": int(sid), "data_dir": self.data_dir})
                    if isinstance(chunks, list) and chunks:
                        res = self._invoke("analyze_chunk_llm", {"chunk_id": int(chunks[0].get("id")), "data_dir": self.data_dir})
                        tool_call_log.append({
                            "tool": tool_name,
                            "fallback_applied": True,
                            "fallback_params": {"chunk_id": int(chunks[0].get("id"))},
                            "result_preview": res if isinstance(res, dict) else res,
                        })
                        return res

        if tool_name == "analyze_file_llm":
            # If file_path missing/invalid, try first available code file
            try:
                files = self._invoke("list_files", {"data_dir": self.data_dir})
                if isinstance(files, list) and files:
                    def _rank(f):
                        p = (f or {}).get("path") or (f or {}).get("file", {}).get("path")
                        if not p:
                            return 99
                        if str(p).endswith(".py"):
                            return 0
                        if str(p).endswith(".js") or str(p).endswith(".ts"):
                            return 1
                        if str(p).endswith(".java"):
                            return 2
                        return 3
                    files_sorted = sorted(files, key=_rank)
                    picked_path = files_sorted[0].get("path") or files_sorted[0].get("file", {}).get("path")
                    if picked_path:
                        res = self._invoke("analyze_file_llm", {"file_path": picked_path, "data_dir": self.data_dir})
                        tool_call_log.append({
                            "tool": tool_name,
                            "fallback_applied": True,
                            "fallback_params": {"file_path": picked_path},
                            "result_preview": res if isinstance(res, dict) else res,
                        })
                        return res
            except Exception:
                pass

        return result

    def analyze(self) -> Dict[str, Any]:
        # 0) 가능한 경우 StructSynth 최신 실행으로부터 대상 repo_path를 가져와 반영
        try:
            latest_run = self._invoke("get_latest_run", {"agent_name": "StructSynth", "data_dir": self.data_dir})
            if isinstance(latest_run, dict):
                # 다양한 저장 형태를 고려하여 repo_path 후보를 탐색
                candidate = (
                    latest_run.get("repo_path")
                    or (latest_run.get("params", {}) or {}).get("repo_path")
                    or (latest_run.get("config", {}) or {}).get("repo_path")
                )
                if isinstance(candidate, str) and candidate.strip():
                    self._effective_repo_path = candidate.strip()
        except Exception:
            pass

        # 1) 규모/구조 파악
        stats = self._invoke("get_database_stats", {"data_dir": self.data_dir})
        files = self._invoke("list_files", {"data_dir": self.data_dir})
        symbols = self._invoke("list_symbols", {"limit": 5000, "data_dir": self.data_dir})

        total_files = stats.get("total_files", 0) if isinstance(stats, dict) else 0
        total_symbols = stats.get("total_symbols", 0) if isinstance(stats, dict) else 0
        total_functions = 0
        if isinstance(stats, dict):
            st = stats.get("symbol_types", {}) or {}
            if isinstance(st, dict):
                total_functions = st.get("function", 0) or 0

        # 2) 산출물 3가지 선택
        # LLM이 결정하도록 시도 (환경변수/모델 설정 필요). 실패 시 사용자에게 실패를 표기하고 중단
        outputs = []
        try:
            tool_specs = [
                {"name": n, "description": (getattr(t, "description", "") or "").strip()}
                for n, t in self.tools.items()
            ]
            if self._llm_planner is None:
                self._llm_planner = LLMPlanner()
            outputs = self._llm_planner.plan(stats if isinstance(stats, dict) else {}, tool_specs)
        except Exception as e:
            notice = (
                "LLM 기반 계획 수립(planner.py) 단계에서 실패했습니다. 환경변수/모델 설정을 확인하고 다시 시도해 주세요. "
                "임의의 고정 답변은 제공하지 않습니다."
            )
            return {
                "status": "planner_failed",
                "notice": notice,
                "error": str(e),
                "inputs": {"stats": stats},
                "selected_outputs": [],
                "drafts": {},
            }

        # 3) 각 산출물 초안 생성(도구 호출 샘플)
        # 핵심 심볼 인덱스: 검색어 몇 개로 샘플 인덱스 생성
        idx_samples: List[Any] = []
        for q in ["init", "data", "process"]:
            r = self._invoke("search_symbols_fts", {"query": q, "top_k": 5, "data_dir": self.data_dir})
            idx_samples.append({"query": q, "results": r})

        # 호출 그래프 개요: 첫 번째 함수형 심볼 기준으로 in/out 확인
        first_func = next((s for s in symbols if s.get("type") == "function"), None) if isinstance(symbols, list) else None
        calls_overview = {}
        if first_func and first_func.get("id"):
            fid = int(first_func["id"])  # type: ignore
            calls_overview = {
                "symbol": first_func,
                "outgoing": self._invoke("get_calls_from", {"caller_id": fid, "data_dir": self.data_dir}),
                "incoming": self._invoke("get_calls_to", {"callee_id": fid, "data_dir": self.data_dir}),
            }

        # 파일별 요약 카탈로그: 상위 2개 파일만 샘플 구성
        catalog: List[Dict[str, Any]] = []
        if isinstance(files, list):
            for f in files[:2]:
                file_symbols = [s for s in symbols if s.get("language") == f.get("language")] if isinstance(symbols, list) else []
                catalog.append({
                    "file": f,
                    "symbol_count_same_lang": len(file_symbols),
                })

        # 4) 산출물별 프롬프트/툴 호출 계획을 LLM이 설계 (chunk_id 필요 도구 금지)
        try:
            tool_specs = [
                {"name": n, "description": (getattr(t, "description", "") or "").strip()}
                for n, t in self.tools.items()
            ]
            plans = self._llm_planner.plan_artifact_prompts(outputs, stats if isinstance(stats, dict) else {}, tool_specs)
        except Exception as e:
            return {
                "status": "artifact_planner_failed",
                "notice": "산출물 프롬프트/툴 계획 수립 실패. 환경설정 또는 응답 포맷을 확인하세요.",
                "error": str(e),
                "selected_outputs": outputs,
            }

        # 첫 번째 산출물은 "분석 대상 소스"에 대한 개요가 되도록 고정 계획을 적용
        overview_plan = {
            "name": "분석 대상 소스 개요",
            "file_ext": "md",
            "tool_invocations": [
                {"name": "get_database_stats", "params": {}, "alias": "db_stats", "purpose": "분석 대상 DB 규모/구조 요약"},
                {"name": "list_files", "params": {}, "alias": "files", "purpose": "파일 목록/언어/디렉토리 분석"},
                {"name": "list_symbols", "params": {"limit": 100}, "alias": "symbols", "purpose": "심볼 분포 샘플"},
                {"name": "search_symbols_fts", "params": {"query": "service", "top_k": 20}, "alias": "sym_service", "purpose": "핵심 기능 단서(서비스 계층)"},
                {"name": "search_symbols_fts", "params": {"query": "controller", "top_k": 20}, "alias": "sym_controller", "purpose": "핵심 기능 단서(컨트롤러/엔드포인트)"},
                {"name": "search_symbols_fts", "params": {"query": "handler", "top_k": 20}, "alias": "sym_handler", "purpose": "핵심 기능 단서(이벤트/요청 처리)"},
                {"name": "search_symbols_fts", "params": {"query": "model", "top_k": 20}, "alias": "sym_model", "purpose": "핵심 기능 단서(데이터/도메인 모델)"},
                {"name": "search_symbols_fts", "params": {"query": "api", "top_k": 20}, "alias": "sym_api", "purpose": "핵심 기능 단서(API/라우팅)"},
                {"name": "get_calls_from", "params": {}, "alias": "sample_out_calls", "purpose": "대표 함수의 외부 호출 예시"},
                {"name": "get_calls_to", "params": {}, "alias": "sample_in_calls", "purpose": "대표 함수로의 내부 호출 예시"},
                {"name": "get_vector_store_stats", "params": {}, "alias": "vector_stats", "purpose": "(선택) 벡터 인덱스 상태"},
                {"name": "get_analysis_summary", "params": {}, "alias": "analysis_summary", "purpose": "(선택) 최근 분석 요약"},
            ],
            "prompt_template": (
                "[목적] 분석 대상 소스 코드베이스의 구조와 특성을 빠르게 파악할 수 있도록 요약합니다.\n"
                "[대상] 이 코드베이스를 처음 접하는 개발자/리뷰어.\n"
                "[출력 형식] 마크다운. 한국어. ToolResults 근거 기반으로 간결하고 정확하게 작성.\n"
                "[구성] \n"
                "# 분석 대상 소스 개요\n"
                "## 전체 요약\n"
                "- 전체 파일 수, 심볼 수, 언어 분포 등 핵심 지표를 표로 정리 (db_stats, symbols 활용)\n"
                "## 디렉토리/모듈 구조\n"
                "- 상위 디렉토리 구조와 역할을 요약 (files 기반, 디렉토리별 파일 수/대표 파일 예시 포함)\n"
                "## 핵심 기능 요약\n"
                "- sym_service/sym_controller/sym_handler/sym_model/sym_api 및 symbols, files의 명명 규칙을 근거로\n"
                "  주요 기능(예: 인증/권한, 데이터 처리, API 제공, 배치/잡, UI/뷰 등)을 항목별로 정리\n"
                "- 각 항목마다 근거가 된 파일/심볼 예시를 코드 블록 또는 인라인 코드로 제시\n"
                "## 핵심 엔트리포인트/모듈 추정\n"
                "- 파일/경로/함수명을 근거로 엔트리포인트 및 핵심 모듈 후보를 제시 (근거를 함께 표기)\n"
                "## 코드 의존관계 개요\n"
                "- sample_out_calls, sample_in_calls 결과를 간단히 서술 (어떤 함수가 무엇을 호출/호출받는지)\n"
                "## 최근 분석/인덱스 상태 (있을 경우)\n"
                "- vector_stats, analysis_summary가 존재하면 간략히 상태를 요약\n"
                "## 다음 탐색 가이드\n"
                "- 코드 이해를 위해 바로 확인하면 좋은 포인트 3-5가지를 제안\n"
                "[작성 지침]\n"
                "- 모든 수치/주장은 반드시 ToolResults 근거를 바탕으로 작성. 근거 없으면 생략하고 추정/환상 금지.\n"
                "- 이 분석 도구(우리 프로젝트)의 구성/포트/실행방법 등은 포함하지 말 것.\n"
                "- 경로/파일명/언어/숫자는 코드 포맷으로 명확히 표기.\n"
            ),
        }

        # 총 4개 산출물: 1) 고정 개요 + 2~4) LLM 결정
        plans = [overview_plan] + (plans or [])

        # 선택 산출물 메타도 첫 항목을 "분석 대상 소스 개요"로 정렬
        try:
            overview_meta = {
                "name": "분석 대상 소스 개요",
                "tools": [
                    "get_database_stats",
                    "list_files",
                    "list_symbols",
                    "search_symbols_fts",
                    "get_calls_from",
                    "get_calls_to",
                    "get_vector_store_stats",
                    "get_analysis_summary",
                ],
                "reason": "분석 대상 코드베이스의 구조/지표/의존관계를 요약하여 빠른 이해를 지원",
                "evaluation_criteria": [
                    "사실성", "근거 제시", "구성 명료성", "경로/수치의 정확도"
                ],
            }
            outputs = [overview_meta] + (outputs or [])
        except Exception:
            pass

        # 5) 계획에 따라 selected_outputs를 실제 산출물로 생성/저장 (chunk 기반 도구는 사용하지 않음)
        artifacts_dir = os.path.join(self.data_dir, "insightgen")
        # 기존 산출물 폴더 제거 후 재생성
        try:
            if os.path.isdir(artifacts_dir):
                shutil.rmtree(artifacts_dir, ignore_errors=True)
        except Exception:
            # 삭제 실패 시에도 이후 생성 시도
            pass
        os.makedirs(artifacts_dir, exist_ok=True)

        artifact_paths: List[str] = []
        tool_call_log: List[Dict[str, Any]] = []

        # LLM 요약 보강 헬퍼
        def _llm_summarize_json(title: str, payload: Any) -> str:
            try:
                # 간단히 search_symbols_fts를 재활용한지 여부와 무관하게 텍스트 생성
                from agents.structsynth.llm_analyzer import LLMAnalyzer  # type: ignore
                analyzer = LLMAnalyzer()
                # analyzer는 텍스트 기반이므로 JSON을 문자열로 변환하여 일반 청크 분석을 활용
                text = f"{title}\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)
                return analyzer._get_llm_response(text)  # type: ignore
            except Exception:
                return ""

        # 거대 ToolResults로 인한 응답 절단 방지용 축약기
        def _shrink_tool_results(obj: Any, max_items_per_list: int = 50, max_str_len: int = 2000, max_depth: int = 6, _depth: int = 0) -> Any:
            try:
                if _depth > max_depth:
                    return "<truncated>"
                if isinstance(obj, dict):
                    trimmed = {}
                    for k, v in obj.items():
                        # content나 code 같은 필드는 더 과감히 축약
                        if isinstance(v, str):
                            val = v[:max_str_len]
                        else:
                            val = _shrink_tool_results(v, max_items_per_list, max_str_len, max_depth, _depth + 1)
                        trimmed[k] = val
                    return trimmed
                if isinstance(obj, list):
                    sliced = obj[:max_items_per_list]
                    return [
                        _shrink_tool_results(v, max_items_per_list, max_str_len, max_depth, _depth + 1)
                        for v in sliced
                    ]
                if isinstance(obj, str):
                    return obj[:max_str_len]
                return obj
            except Exception:
                return obj

        # 긴 산출물 생성을 위한 페이징 생성기
        def _generate_paged_response(analyzer, base_prompt: str, per_call_max_tokens: int = 900, max_pages: int = 5) -> str:
            combined = ""
            instruction = (
                "\n\n[응답 지침]\n"
                "- 가능한 한 완전하게 작성하되, 답변 끝에 상태 마커를 붙이세요.\n"
                "- 완료 시: <<<END>>>\n"
                "- 분량이 남으면: <<<MORE>>> (추가 분량은 이후 요청에서 이어서 작성)\n"
                "- 표/리스트는 간결하게. 중복 금지.\n"
            )
            prompt = base_prompt + instruction
            last_len = 0
            for page in range(max_pages):
                part = analyzer._get_llm_response(prompt, max_tokens=per_call_max_tokens)  # type: ignore
                if not part:
                    break
                combined += ("\n" if combined else "") + part
                # 종료 조건: 마커 존재 또는 진행 없음
                if "<<<END>>>" in part:
                    break
                if len(combined) - last_len < 50:  # 거의 진행 없음
                    break
                last_len = len(combined)
                # 다음 페이지 프롬프트: 이어서 작성
                prompt = (
                    "이전 응답의 이어서 작성:\n" + part[-1200:] +
                    "\n\n[지침] 바로 이어서 새 내용만 작성하세요. 반복 금지. 마지막에 <<<END>>> 또는 <<<MORE>>> 표시."
                )
            return combined.replace("<<<MORE>>>", "").replace("<<<END>>>", "").strip()

        # 계획된 각 산출물에 대해: 지정된 tool_invocations 실행 → ToolResults를 사용해 LLM 프롬프트 템플릿으로 생성
        for i, plan in enumerate(plans):
            name = plan.get("name", f"artifact_{i+1}")
            ext = plan.get("file_ext", "md")
            invocations = plan.get("tool_invocations", []) or []
            prompt_template = plan.get("prompt_template", "")

            tool_results = {}
            for inv in invocations:
                tname = inv.get("name")
                params = inv.get("params", {})
                alias = inv.get("alias", tname)
                if not tname:
                    continue
                result = self._safe_invoke_with_fallback(
                    tname,
                    params,
                    symbols if isinstance(symbols, list) else None,
                    tool_call_log,
                )
                tool_results[alias] = result

            # 산출물 생성용 프롬프트를 LLM이 준 템플릿과 ToolResults로 구성하여 요약
            try:
                from agents.structsynth.llm_analyzer import LLMAnalyzer  # type: ignore
                analyzer = LLMAnalyzer()
                safe_tool_results = _shrink_tool_results(tool_results, max_items_per_list=30, max_str_len=1200, max_depth=4)
                full_prompt = (
                    f"산출물: {name}\n\n[요구 템플릿]\n{prompt_template}\n\n[ToolResults](요약본)\n"
                    + json.dumps(safe_tool_results, ensure_ascii=False, indent=2)
                )
                # 최종 산출물은 페이징으로 수집하여 응답 절단 방지
                output_text = _generate_paged_response(analyzer, full_prompt, per_call_max_tokens=1200, max_pages=12)
            except Exception:
                # LLM 불가 시: 고정 응답을 생성하지 않고, 템플릿/도구결과 원본만 남김
                output_text = json.dumps({"template": prompt_template, "ToolResults": tool_results}, ensure_ascii=False, indent=2)

            filename = f"{i+1:02d}-{name}".replace(" ", "-")
            outpath = os.path.join(artifacts_dir, f"{filename}.{ext}")
            with open(outpath, "w", encoding="utf-8") as f:
                f.write(output_text)
            artifact_paths.append(outpath)

        return {
            "inputs": {"stats": stats},
            "selected_outputs": outputs,
            "drafts": {
                "index_samples": idx_samples,
                "calls_overview": calls_overview,
                "file_catalog": catalog,
            },
            "artifacts": artifact_paths,
            "plans": plans,
            "tool_call_log": tool_call_log,
        }


