"""
LLM Analyzer - 함수/메서드 내부 정보 분석
AST에서 추출된 함수 정보를 LLM에 전달하여 상세 분석 수행
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional
import openai

logger = logging.getLogger(__name__)


class LLMAnalyzer:
    """LLM을 통한 함수/메서드 분석기"""
    
    def __init__(self):
        # 환경변수에서 Azure OpenAI 설정 가져오기
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
        self.embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        
        # API 키 검증
        if not self.api_key:
            raise RuntimeError("AZURE_OPENAI_API_KEY 환경변수가 설정되지 않았습니다")
        if not self.endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT 환경변수가 설정되지 않았습니다")
        
        # OpenAI 클라이언트 초기화
        self.client = openai.AzureOpenAI(
            api_key=self.api_key,
            azure_endpoint=self.endpoint,
            api_version=self.api_version
        )
        
        logger.info(f"LLMAnalyzer initialized with Azure OpenAI deployment: {self.deployment}")

    def analyze_function(self, function_data: Dict[str, Any], file_context: str = "") -> Dict[str, Any]:
        """함수/메서드 분석"""
        try:
            # LLM 분석을 위한 프롬프트 구성
            prompt = self._build_function_analysis_prompt(function_data, file_context)
            
            # LLM 호출
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            # 응답 파싱
            analysis = self._parse_llm_response(response.choices[0].message.content)
            
            # 원본 데이터와 분석 결과 병합
            enriched_function = function_data.copy()
            enriched_function["llm_analysis"] = {
                "summary": analysis.get("summary", "분석 완료"),
                "purpose": analysis.get("purpose", "목적 분석"),
                "logic_overview": analysis.get("logic_overview", "로직 분석"),
                "execution_flow": analysis.get("execution_flow", "실행 흐름 분석"),
                "sequence_steps": analysis.get("sequence_steps", [])
            }
            
            logger.info(f"함수 분석 완료: {function_data.get('name', 'unknown')}")
            return enriched_function
            
        except Exception as e:
            logger.error(f"함수 분석 실패: {e}")
            # 분석 실패 시 원본 데이터 반환
            function_data["llm_analysis"] = {
                "summary": "분석 실패",
                "purpose": "분석 실패",
                "logic_overview": "분석 실패",
                "execution_flow": "분석 실패",
                "sequence_steps": []
            }
            return function_data

    def analyze_class(self, class_data: Dict[str, Any], file_context: str = "") -> Dict[str, Any]:
        """클래스 분석"""
        try:
            # LLM 분석을 위한 프롬프트 구성
            prompt = self._build_class_analysis_prompt(class_data, file_context)
            
            # LLM 호출
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1200
            )
            
            # 응답 파싱
            analysis = self._parse_llm_response(response.choices[0].message.content)
            
            # 원본 데이터와 분석 결과 병합
            enriched_class = class_data.copy()
            enriched_class["llm_analysis"] = {
                "summary": analysis.get("summary", "분석 완료"),
                "responsibility": analysis.get("responsibility", "책임 분석"),
                "collaboration": analysis.get("collaboration", []),
                "design_notes": analysis.get("design_notes", "설계 분석")
            }
            
            logger.info(f"클래스 분석 완료: {class_data.get('name', 'unknown')}")
            return enriched_class
            
        except Exception as e:
            logger.error(f"클래스 분석 실패: {e}")
            # 분석 실패 시 원본 데이터 반환
            class_data["llm_analysis"] = {
                "summary": "분석 실패",
                "responsibility": "분석 실패",
                "collaboration": [],
                "design_notes": "분석 실패"
            }
            return class_data

    def analyze_file(self, ast_data: Dict[str, Any], file_context: str) -> Dict[str, Any]:
        """파일 전체에 대한 LLM 분석"""
        try:
            symbols = ast_data.get("symbols", [])
            file_info = ast_data.get("file", {})
            
            # 파일 요약 정보
            summary_prompt = f"""
다음 Python 파일을 분석하여 간결하게 요약해주세요:

{file_context}

파일의 주요 기능과 역할을 1-2문장으로 요약해주세요.
"""
            
            summary = self._get_llm_response(summary_prompt)
            
            # 책임 분석
            responsibility_prompt = f"""
다음 Python 파일의 주요 책임을 분석해주세요:

{file_context}

이 파일이 담당하는 주요 책임을 1-2문장으로 설명해주세요.
"""
            
            responsibility = self._get_llm_response(responsibility_prompt)
            
            # 협력 관계 분석
            collaboration_prompt = f"""
다음 Python 파일의 협력 관계를 분석해주세요:

{file_context}

이 파일이 어떤 객체/모듈과 협력하는지 리스트로 정리해주세요.
"""
            
            collaboration = self._get_llm_response(collaboration_prompt)
            
            # 설계 노트
            design_prompt = f"""
다음 Python 파일의 설계 특징을 분석해주세요:

{file_context}

이 파일의 설계 패턴, 원칙, 장점 등을 1-2문장으로 설명해주세요.
"""
            
            design_notes = self._get_llm_response(design_prompt)
            
            return {
                "summary": summary,
                "responsibility": responsibility,
                "collaboration": [collab.strip() for collab in collaboration.split('\n') if collab.strip()],
                "design_notes": design_notes
            }
            
        except Exception as e:
            logger.error(f"파일 분석 실패: {e}")
            return {
                "summary": "분석 실패",
                "responsibility": "분석 실패",
                "collaboration": [],
                "design_notes": "분석 실패"
            }

    def _get_llm_response(self, prompt: str) -> str:
        """LLM에 프롬프트를 전송하고 응답 받기"""
        try:
            response = self.client.chat.completions.create(
                model=self.deployment,
                messages=[
                    {"role": "system", "content": "당신은 코드 분석 전문가입니다. 간결하고 정확한 분석을 제공해주세요."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM 응답 실패: {e}")
            return "분석 실패"

    def _build_function_analysis_prompt(self, function_data: Dict[str, Any], file_context: str) -> str:
        """함수 분석을 위한 프롬프트 구성"""
        prompt = f"""
다음 함수를 분석해주세요:

**함수 정보:**
- 이름: {function_data.get('name', 'N/A')}
- 타입: {function_data.get('type', 'N/A')}
- 시그니처: {function_data.get('signature', 'N/A')}
- 위치: {function_data.get('location', 'N/A')}
- 매개변수: {function_data.get('metadata', {}).get('parameters', [])}
- 반환 타입: {function_data.get('metadata', {}).get('return_type', 'N/A')}
- 데코레이터: {function_data.get('metadata', {}).get('decorators', [])}

**함수 본문:**
{json.dumps(function_data.get('body', {}), indent=2, ensure_ascii=False)}

**파일 컨텍스트:**
{file_context[:1000] if file_context else 'N/A'}

다음 JSON 형식으로 분석 결과를 반환해주세요:
{{
    "summary": "함수의 주요 기능 요약 (한 문장)",
    "purpose": "함수의 주요 목적과 역할 (1-2문장)",
    "logic_overview": "주요 로직과 동작 방식 요약 (1-2문장)",
    "execution_flow": "실행 흐름 설명 (1-2문장)",
    "sequence_steps": ["단계별 실행 순서 리스트"]
}}
"""
        return prompt

    def _build_class_analysis_prompt(self, class_data: Dict[str, Any], file_context: str) -> str:
        """클래스 분석을 위한 프롬프트 구성"""
        prompt = f"""
다음 클래스를 분석해주세요:

**클래스 정보:**
- 이름: {class_data.get('name', 'N/A')}
- 시그니처: {class_data.get('signature', 'N/A')}
- 위치: {class_data.get('location', 'N/A')}
- 상속: {class_data.get('metadata', {}).get('bases', [])}
- 데코레이터: {class_data.get('metadata', {}).get('decorators', [])}
- 메서드: {class_data.get('metadata', {}).get('methods', [])}
- 필드: {class_data.get('metadata', {}).get('fields', [])}

**클래스 본문:**
{json.dumps(class_data.get('body', {}), indent=2, ensure_ascii=False)}

**파일 컨텍스트:**
{file_context[:1000] if file_context else 'N/A'}

다음 JSON 형식으로 분석 결과를 반환해주세요:
{{
    "summary": "클래스의 주요 기능 요약 (한 문장)",
    "responsibility": "단일 책임 원칙 관점에서의 책임 (1-2문장)",
    "collaboration": ["협력하는 주요 객체/클래스들"],
    "design_notes": "사용된 디자인 패턴과 원칙 (1-2문장)"
}}
"""
        return prompt

    def _get_system_prompt(self) -> str:
        """시스템 프롬프트"""
        return """당신은 코드 분석 전문가입니다. 
주어진 함수나 클래스를 분석하여 목적, 복잡도, 의존성 등을 파악하고,
JSON 형식으로 구조화된 분석 결과를 제공해야 합니다.
분석은 객관적이고 정확해야 하며, 코드의 가독성과 유지보수성 관점에서 평가해야 합니다."""

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """LLM 응답을 파싱하여 JSON으로 변환"""
        try:
            # JSON 부분 추출
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # JSON이 없는 경우 기본 구조 반환
                return {
                    "purpose": "응답 파싱 실패",
                    "summary": response[:200] if response else "빈 응답"
                }
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 파싱 실패: {e}")
            return {
                "purpose": "JSON 파싱 실패",
                "summary": response[:200] if response else "빈 응답"
            }
    
    def analyze_repository(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """저장소 전체에 대한 LLM 분석"""
        try:
            files = parsed_data.get("files", [])
            symbols = parsed_data.get("symbols", [])
            
            # 저장소 요약 정보
            summary_prompt = f"""
다음 코드베이스를 분석하여 간결하게 요약해주세요:

총 파일 수: {len(files)}
총 심볼 수: {len(symbols)}

파일 목록:
{chr(10).join([f"- {f.get('file', {}).get('path', 'unknown')}" for f in files[:10]])}

이 코드베이스의 주요 기능과 역할을 2-3문장으로 요약해주세요.
"""
            
            summary = self._get_llm_response(summary_prompt)
            
            # 전체 구조 분석
            structure_prompt = f"""
다음 코드베이스의 구조적 특징을 분석해주세요:

파일 수: {len(files)}
심볼 수: {len(symbols)}

주요 언어: {', '.join(set([f.get('file', {}).get('language', 'unknown') for f in files]))}

이 코드베이스의 아키텍처와 설계 패턴을 2-3문장으로 설명해주세요.
"""
            
            structure = self._get_llm_response(structure_prompt)
            
            return {
                "summary": summary,
                "structure": structure,
                "total_files": len(files),
                "total_symbols": len(symbols),
                "languages": list(set([f.get('file', {}).get('language', 'unknown') for f in files]))
            }
            
        except Exception as e:
            logger.error(f"저장소 분석 실패: {e}")
            return {
                "summary": "분석 실패",
                "structure": "분석 실패",
                "total_files": 0,
                "total_symbols": 0,
                "languages": []
            }
    
    def perform_repository_analysis(self, parsed_data: Dict[str, Any]) -> Dict[str, Any]:
        """저장소 전체에 대한 종합 LLM 분석 수행"""
        try:
            analysis_results = {}
            
            # 1. 파일별 개별 분석
            for file_data in parsed_data.get("files", []):
                file_path = file_data.get("file_path", "")
                if file_path:
                    try:
                        file_summary = self.analyze_file(file_data, "")
                        analysis_results[file_path] = file_summary
                        logger.info(f"파일 분석 완료: {file_path}")
                    except Exception as e:
                        logger.warning(f"파일 분석 실패 {file_path}: {e}")
                        analysis_results[file_path] = {"error": str(e)}
            
            # 2. 저장소 전체 분석
            try:
                repo_analysis = self.analyze_repository(parsed_data)
                analysis_results["repository"] = repo_analysis
                logger.info("저장소 전체 분석 완료")
            except Exception as e:
                logger.warning(f"저장소 전체 분석 실패: {e}")
                analysis_results["repository"] = {"error": str(e)}
            
            # 3. 분석 통계
            analysis_results["statistics"] = {
                "total_files": len(parsed_data.get("files", [])),
                "total_symbols": len(parsed_data.get("symbols", [])),
                "files_analyzed": len([k for k, v in analysis_results.items() if k != "repository" and k != "statistics" and "error" not in v]),
                "analysis_timestamp": self._get_timestamp()
            }
            
            logger.info("저장소 LLM 분석 완료")
            return analysis_results
            
        except Exception as e:
            logger.error(f"저장소 LLM 분석 실패: {e}")
            return {
                "error": str(e),
                "statistics": {
                    "total_files": 0,
                    "total_symbols": 0,
                    "files_analyzed": 0,
                    "analysis_timestamp": self._get_timestamp()
                }
            }
    
    def _get_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def analyze_chunk(self, chunk_data: Dict[str, Any]) -> Dict[str, Any]:
        """청킹 단위 LLM 분석"""
        try:
            chunk_type = chunk_data.get("symbol_type", "unknown")
            chunk_name = chunk_data.get("symbol_name", "unknown")
            content = chunk_data.get("content", "")
            
            # 청크 타입별 분석 프롬프트 구성
            if chunk_type == "function":
                prompt = self._build_function_chunk_prompt(chunk_name, content)
            elif chunk_type == "class":
                prompt = self._build_class_chunk_prompt(chunk_name, content)
            else:
                prompt = self._build_general_chunk_prompt(chunk_name, chunk_type, content)
            
            # LLM 호출
            response = self._get_llm_response(prompt)
            
            # 분석 결과 구성
            analysis = {
                "summary": response,
                "chunk_type": chunk_type,
                "chunk_name": chunk_name,
                "analysis_timestamp": self._get_timestamp()
            }
            
            logger.info(f"청크 분석 완료: {chunk_name} ({chunk_type})")
            return analysis
            
        except Exception as e:
            logger.error(f"청크 분석 실패: {e}")
            return {
                "summary": "분석 실패",
                "chunk_type": chunk_data.get("symbol_type", "unknown"),
                "chunk_name": chunk_data.get("symbol_name", "unknown"),
                "error": str(e),
                "analysis_timestamp": self._get_timestamp()
            }
    
    def _build_function_chunk_prompt(self, func_name: str, content: str) -> str:
        """함수 청크 분석 프롬프트"""
        return f"""
다음 Python 함수를 분석해주세요:

**함수명**: {func_name}
**내용**:
{content}

이 함수의 주요 기능, 목적, 동작 방식을 1-2문장으로 요약해주세요.
"""
    
    def _build_class_chunk_prompt(self, class_name: str, content: str) -> str:
        """클래스 청크 분석 프롬프트"""
        return f"""
다음 Python 클래스를 분석해주세요:

**클래스명**: {class_name}
**내용**:
{content}

이 클래스의 주요 책임, 설계 패턴, 역할을 1-2문장으로 요약해주세요.
"""
    
    def _build_general_chunk_prompt(self, chunk_name: str, chunk_type: str, content: str) -> str:
        """일반 청크 분석 프롬프트"""
        return f"""
다음 코드 청크를 분석해주세요:

**이름**: {chunk_name}
**타입**: {chunk_type}
**내용**:
{content}

이 코드의 주요 기능과 역할을 1-2문장으로 요약해주세요.
""" 