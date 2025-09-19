"""
CodeChat Agent - 하이브리드 검색(FTS+FAISS) + RAG 기반 코드 채팅
코드베이스에 대한 질문에 답변하고 관련 코드를 찾아서 제시
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np

from openai import AzureOpenAI
from common.store.sqlite_store import SQLiteStore
from .simple_context_manager import SimpleContextManager, ConversationContext
from .simple_evaluator import SimpleSelfEvaluator, SimpleEvaluation

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    symbol_name: str
    symbol_type: str
    file_path: str
    start_line: int
    end_line: int
    content: str
    source: str
    similarity_score: float
    input_types: Optional[str] = None
    output_types: Optional[str] = None
    dependencies: Optional[List[str]] = None
    usage_examples: Optional[str] = None

@dataclass
class ChatResponse:
    """채팅 응답 데이터 클래스"""
    answer: str
    evidence: List[SearchResult]
    confidence: float

class CodeChatAgent:
    """코드 채팅 에이전트 - 하이브리드 검색 + RAG"""
    
    def __init__(self, repo_path: str = ".", artifacts_dir: str = "./artifacts", data_dir: str = "./data"):
        """
        CodeChatAgent 초기화
        
        Args:
            repo_path: 코드베이스 경로
            artifacts_dir: 벡터 스토어가 저장된 디렉토리
            data_dir: SQLite 데이터베이스 디렉토리
        """
        self.repo_path = Path(repo_path)
        self.artifacts_dir = Path(artifacts_dir)
        self.data_dir = Path(data_dir)
        
        # Azure OpenAI 클라이언트 초기화
        self._init_openai_client()
        
        # 데이터베이스 및 벡터 스토어 초기화
        self._init_data_sources()
        
        # AI 컨텍스트 관리 초기화
        self._init_ai_context_manager()
        
        # AI 자율 평가 시스템 초기화
        self._init_self_evaluator()
        
        logger.info(f"CodeChatAgent 초기화 완료: {self.repo_path}")
    
    def _init_openai_client(self):
        """Azure OpenAI 클라이언트 초기화"""
        try:
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
            
            # 채팅용 모델 (gpt-4o)
            deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
            
            # embeddings용 모델
            embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
            
            if not api_key or not endpoint:
                raise ValueError("Azure OpenAI API 키와 엔드포인트가 필요합니다")
            
            self.openai_client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version
            )
            self.deployment_name = deployment_name
            self.embedding_deployment = embedding_deployment
            
            logger.info(f"Azure OpenAI 클라이언트 초기화 완료 (모델: {deployment_name}, 임베딩: {embedding_deployment})")
            
        except Exception as e:
            logger.error(f"Azure OpenAI 클라이언트 초기화 실패: {e}")
            raise
    
    def _init_data_sources(self):
        """데이터 소스 초기화"""
        try:
            # SQLite 데이터베이스 초기화
            self.sqlite_store = SQLiteStore(self.data_dir / "structsynth_code.db")
            
            # StructSynthAgent 초기화 (개선된 검색 기능 사용)
            try:
                from agents.structsynth.agent import StructSynthAgent
                self.structsynth_agent = StructSynthAgent(
                    repo_path=str(self.repo_path),
                    artifacts_dir=str(self.artifacts_dir),
                    data_dir=str(self.data_dir)
                )
                logger.info("StructSynthAgent 초기화 완료")
            except Exception as e:
                logger.warning(f"StructSynthAgent 초기화 실패: {e}")
                self.structsynth_agent = None
            
            # 벡터 스토어 로드 (기존 방식 유지)
            self._load_vector_store()
            
            logger.info("데이터 소스 초기화 완료")
            
        except Exception as e:
            logger.error(f"데이터 소스 초기화 실패: {e}")
            raise
    
    def _init_ai_context_manager(self):
        """단순 대화 컨텍스트 관리자 초기화"""
        try:
            # 단순 대화 컨텍스트 관리자 초기화
            self.context_manager = SimpleContextManager(max_context_size=10)
            
            logger.info("대화 컨텍스트 관리자 초기화 완료")
            
        except Exception as e:
            logger.error(f"대화 컨텍스트 관리자 초기화 실패: {e}")
            # 컨텍스트 관리자 실패 시에도 기본 기능은 동작하도록 함
            self.context_manager = None
    
    def _init_self_evaluator(self):
        """AI 자율 평가 시스템 초기화"""
        try:
            # 간단한 AI 자율 평가 시스템 초기화
            self.self_evaluator = SimpleSelfEvaluator(
                self.openai_client,
                self.deployment_name
            )
            
            logger.info("AI 자율 평가 시스템 초기화 완료")
            
        except Exception as e:
            logger.error(f"AI 자율 평가 시스템 초기화 실패: {e}")
            # 평가 시스템 실패 시에도 기본 기능은 동작하도록 함
            self.self_evaluator = None
    
    def _load_vector_store(self):
        """벡터 스토어를 SQLite와 FAISS에서 로드"""
        try:
            # 1. SQLite embeddings 테이블에서 벡터 데이터 로드
            embeddings_data = self.sqlite_store.get_all_embeddings()
            
            if embeddings_data:
                # 벡터와 메타데이터 구성
                self.vectors = []
                self.metadata = []
                
                for emb in embeddings_data:
                    try:
                        # bytes -> numpy array 변환
                        vector = np.frombuffer(emb['vector'], dtype=np.float32)
                        self.vectors.append(vector)
                        
                        # chunks 테이블에서 메타데이터 조회
                        chunk_info = self.sqlite_store.get_chunk_info(emb['object_id'])
                        if chunk_info:
                            # SearchResult 형식에 맞게 메타데이터 구성
                            metadata = {
                                "symbol_name": chunk_info.get('symbol_name', ''),
                                "symbol_type": chunk_info.get('symbol_type', ''),
                                "file_path": chunk_info.get('file_path', ''),
                                "start_line": chunk_info.get('start_line', 0),
                                "end_line": chunk_info.get('end_line', 0),
                                "content": chunk_info.get('content', ''),
                                "chunk_id": emb['object_id']
                            }
                            self.metadata.append(metadata)
                        else:
                            logger.warning(f"청크 {emb['object_id']} 정보를 찾을 수 없습니다")
                            
                    except Exception as e:
                        logger.warning(f"임베딩 {emb['id']} 처리 실패: {e}")
                        continue
                
                logger.info(f"SQLite에서 벡터 로드 완료: {len(self.vectors)}개")
                
                # 2. FAISS 인덱스 로드 시도
                self._load_faiss_index()
                
            else:
                logger.warning("embeddings 테이블에 데이터가 없습니다")
                self.vectors = None
                self.metadata = None
                
        except Exception as e:
            logger.error(f"벡터 스토어 로드 실패: {e}")
            self.vectors = None
            self.metadata = None
    
    def _load_faiss_index(self):
        """FAISS 인덱스 로드"""
        try:
            # FAISS 인덱스 파일 경로 확인
            faiss_index_path = self.artifacts_dir / "faiss.index"
            
            if faiss_index_path.exists():
                # FAISSStore 사용하여 인덱스 로드
                from common.store.faiss_store import FAISSStore
                
                self.faiss_store = FAISSStore(
                    index_path=str(faiss_index_path),
                    dimension=3072  # text-embedding-3-large 차원
                )
                
                # 기존 인덱스 로드
                if hasattr(self.faiss_store, 'load_or_create_index'):
                    self.faiss_store.load_or_create_index()
                    logger.info("FAISS 인덱스 로드 완료")
                else:
                    logger.info("FAISS 인덱스 초기화 완료")
                    
            else:
                logger.warning(f"FAISS 인덱스 파일이 없습니다: {faiss_index_path}")
                self.faiss_store = None
                
        except Exception as e:
            logger.error(f"FAISS 인덱스 로드 실패: {e}")
            self.faiss_store = None
    
    def chat(self, query: str, top_k: int = 10, session_id: str = "default", 
             user_id: Optional[str] = None) -> ChatResponse:
        """개선된 채팅 응답 생성 - AI 기반 도구 선택"""
        try:
            # query 검증 및 상세 로깅
            logger.info(f"채팅 요청 수신 - query 타입: {type(query)}, 값: '{query}'")
            
            if query is None:
                logger.warning("채팅 요청 무효: query가 None")
                return ChatResponse(
                    answer="질문을 입력해주세요.",
                    evidence=[],
                    confidence=0.0
                )
            
            if not isinstance(query, str):
                logger.warning(f"채팅 요청 무효: query가 문자열이 아님 (타입: {type(query)})")
                return ChatResponse(
                    answer="질문이 올바른 형식이 아닙니다.",
                    evidence=[],
                    confidence=0.0
                )
            
            if not query.strip():
                logger.warning("채팅 요청 무효: query가 비어 있음 (공백만 포함)")
                return ChatResponse(
                    answer="질문을 입력해주세요.",
                    evidence=[],
                    confidence=0.0
                )
            
            # 공백 제거된 query 사용
            query = query.strip()
            logger.info(f"채팅 요청 처리 시작: '{query}' (세션: {session_id})")
            
            # 1. 대화 컨텍스트 정보 가져오기
            context = {}
            if self.context_manager:
                try:
                    context = self.context_manager.get_context_for_query(session_id, query)
                    logger.info(f"💬 대화 컨텍스트 로드: {context.get('total_conversations', 0)}개 대화 기록")
                    if context.get('mentioned_symbols'):
                        logger.info(f"   - 언급된 심볼들: {context['mentioned_symbols'][:3]}")
                    if context.get('mentioned_files'):
                        logger.info(f"   - 언급된 파일들: {context['mentioned_files'][:3]}")
                except Exception as e:
                    logger.warning(f"대화 컨텍스트 로드 실패: {e}")
            
            # 2. AI 기반 질문 분석 및 도구 선택 (컨텍스트 활용)
            logger.info("AI 기반 질문 분석 시작")
            tool_selection = self._analyze_query_and_select_tools(query, context)
            
            # 도구 선택 결과 로깅
            logger.info(f"🔍 질문 분석 결과:")
            logger.info(f"   - 질문 유형: {tool_selection['query_type']}")
            logger.info(f"   - 선택된 도구: {tool_selection['selected_tools']}")
            logger.info(f"   - 선택 이유: {tool_selection['reasoning']}")
            if tool_selection.get('artifact_name'):
                logger.info(f"   - 산출물 파일: {tool_selection['artifact_name']}")
            
            # 3. 질문 유형에 따른 처리
            if tool_selection["query_type"] == "overview":
                # 개요/전반적 질문: get_artifact 사용
                logger.info("📄 개요 질문 감지 - 산출물 기반 답변 생성")
                response = self._handle_overview_query(query, tool_selection)
            else:
                # 특정 코드 질문: 하이브리드 검색 + LLM 분석
                logger.info("🔍 특정 코드 질문 감지 - 하이브리드 검색 수행")
                response = self._handle_specific_query(query, tool_selection, top_k)
            
            # 4. AI 자율 평가 및 개선 (3번 시도 후 최고 점수 선택)
            if self.self_evaluator:
                try:
                    logger.info("🤖 AI 자율 평가 및 다중 시도 개선 시작")
                    
                    # 최대 3번 시도하여 가장 좋은 답변 선택
                    best_answer = response.answer
                    best_evaluation = None
                    best_score = 0.0
                    attempts = []
                    
                    for attempt in range(3):
                        logger.info(f"🔄 개선 시도 {attempt + 1}/3")
                        
                        try:
                            # 각 시도마다 답변 개선
                            current_answer = response.answer if attempt == 0 else best_answer
                            improved_answer, evaluation = self.self_evaluator.evaluate_and_improve(
                                question=query,
                                answer=current_answer,
                                context=context,
                                codechat_agent=self,
                                attempt_number=attempt + 1
                            )
                            
                            # 점수 기록
                            current_score = evaluation.score if evaluation else 0.0
                            attempts.append({
                                "attempt": attempt + 1,
                                "answer": improved_answer,
                                "evaluation": evaluation,
                                "score": current_score
                            })
                            
                            logger.info(f"   시도 {attempt + 1} 점수: {current_score:.1f}/100")
                            
                            # 가장 높은 점수인 경우 업데이트
                            if current_score > best_score:
                                best_score = current_score
                                best_answer = improved_answer
                                best_evaluation = evaluation
                                logger.info(f"   ✨ 새로운 최고 점수! {current_score:.1f}/100")
                            
                            # 점수가 95점 이상이면 더 이상 시도하지 않음
                            if current_score >= 95.0:
                                logger.info(f"   🎯 우수한 점수 달성 ({current_score:.1f}), 추가 시도 중단")
                                break
                                
                        except Exception as e:
                            logger.warning(f"   ❌ 시도 {attempt + 1} 실패: {e}")
                            attempts.append({
                                "attempt": attempt + 1,
                                "answer": response.answer,
                                "evaluation": None,
                                "score": 0.0,
                                "error": str(e)
                            })
                            continue
                    
                    # 최고 점수 답변으로 업데이트
                    response.answer = best_answer
                    
                    # 최종 평가 결과 로깅
                    logger.info(f"🏆 다중 시도 완료 - 최종 선택:")
                    logger.info(f"   - 최고 점수: {best_score:.1f}/100")
                    if best_evaluation:
                        logger.info(f"   - 충분함: {best_evaluation.is_sufficient}")
                        if best_evaluation.missing_parts:
                            logger.info(f"   - 부족한 부분: {', '.join(best_evaluation.missing_parts)}")
                        if best_evaluation.feedback:
                            logger.info(f"   - 피드백: {best_evaluation.feedback}")
                    
                    # 모든 시도 요약
                    logger.info("📋 전체 시도 요약:")
                    for att in attempts:
                        if "error" in att:
                            logger.info(f"   시도 {att['attempt']}: 실패 ({att['error'][:50]}...)")
                        else:
                            logger.info(f"   시도 {att['attempt']}: {att['score']:.1f}점")
                    
                except Exception as e:
                    logger.error(f"❌ AI 자율 평가 전체 실패: {e}")
                    logger.info("🔄 원본 답변 유지")
            
            # 5. 대화 컨텍스트 업데이트
            if self.context_manager:
                try:
                    # SearchResult를 딕셔너리로 변환
                    search_results = []
                    for evidence in response.evidence:
                        search_results.append({
                            "symbol_name": evidence.symbol_name,
                            "file_path": evidence.file_path,
                            "symbol_type": evidence.symbol_type,
                            "content": evidence.content[:200] + "..." if len(evidence.content) > 200 else evidence.content
                        })
                    
                    self.context_manager.add_conversation(
                        session_id=session_id,
                        query=query,
                        answer=response.answer,
                        search_results=search_results,
                        query_type=tool_selection.get("query_type", "specific"),
                        tools_used=tool_selection.get("selected_tools", [])
                    )
                    logger.info("💬 대화 컨텍스트 업데이트 완료")
                except Exception as e:
                    logger.warning(f"대화 컨텍스트 업데이트 실패: {e}")
            
            return response
            
        except Exception as e:
            logger.error(f"채팅 응답 생성 실패: {e}")
            return ChatResponse(
                answer=f"죄송합니다. 오류가 발생했습니다: {str(e)}",
                evidence=[],
                confidence=0.0
            )
    
    def _detect_general_query(self, query: str) -> bool:
        """일반적인 질문인지 감지"""
        if not query or not isinstance(query, str):
            return False
            
        general_keywords = [
            "전체", "전반", "overview", "summary", "구조", "아키텍처",
            "분석해줘", "설명해줘", "요약해줘", "개요", "전체적으로", "전반적으로"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in general_keywords)
    
    def _handle_general_analysis(self, query: str) -> str:
        """전체 코드 분석 처리"""
        try:
            logger.info("전체 코드 분석 시작")
            
            # 1. 데이터베이스 통계 수집
            stats = self.sqlite_store.get_database_stats()
            
            # 2. 주요 파일 및 심볼 정보 수집
            files = self.sqlite_store.get_all_files()
            symbols = self.sqlite_store.get_all_symbols()
            
            # 3. 전체 구조 분석을 위한 컨텍스트 구성
            context = self._build_overview_context(stats, files, symbols)
            
            # 4. LLM을 통한 전체 분석
            return self._generate_overview_analysis(query, context)
            
        except Exception as e:
            logger.error(f"전체 분석 실패: {e}")
            return "전체 코드 분석 중 오류가 발생했습니다."
    
    def _build_overview_context(self, stats: Dict[str, Any], files: List[Dict], symbols: List[Dict]) -> str:
        """전체 분석을 위한 컨텍스트 구성"""
        context_parts = []
        
        # 통계 정보
        context_parts.append(f"""
        [프로젝트 개요]
        총 파일 수: {stats.get('total_files', 0)}
        총 심볼 수: {stats.get('total_symbols', 0)}
        총 청크 수: {stats.get('total_chunks', 0)}
        """)
        
        # 파일 구조
        if files:
            context_parts.append("\n[파일 구조]")
            for file_info in files[:10]:  # 상위 10개 파일만
                context_parts.append(f"- {file_info.get('path', 'unknown')} ({file_info.get('language', 'unknown')})")
                if file_info.get('llm_summary'):
                    context_parts.append(f"  요약: {file_info.get('llm_summary')}")
        
        # 심볼 분포
        if symbols:
            symbol_types = {}
            for symbol in symbols:
                symbol_type = symbol.get('type', 'unknown')
                symbol_types[symbol_type] = symbol_types.get(symbol_type, 0) + 1
            
            context_parts.append("\n[심볼 분포]")
            for symbol_type, count in symbol_types.items():
                context_parts.append(f"- {symbol_type}: {count}개")
        
        return "\n".join(context_parts)
    
    def _generate_overview_analysis(self, query: str, context: str) -> str:
        """전체 분석을 위한 LLM 응답 생성"""
        try:
            prompt = f"""
            다음 코드베이스에 대한 전체적인 분석을 요청받았습니다:
            
            질문: {query}
            
            코드베이스 컨텍스트:
            {context}
            
            요구사항:
            1. 코드베이스의 전체적인 구조와 특징을 설명해주세요
            2. 주요 파일들의 역할과 책임을 요약해주세요
            3. 심볼 분포를 바탕으로 코드의 복잡도와 패턴을 분석해주세요
            4. 전반적인 아키텍처 특징을 설명해주세요
            5. 한국어로 답변하세요
            """
            
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "당신은 코드베이스 분석 전문가입니다. 전체적인 관점에서 코드를 분석하고 요약해주세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            logger.info("전체 분석 완료")
            return answer
            
        except Exception as e:
            logger.error(f"전체 분석 LLM 응답 생성 실패: {e}")
            return f"전체 분석 중 오류가 발생했습니다: {str(e)}"
    
    def _hybrid_search(self, query: str, top_k: int) -> List[SearchResult]:
        """하이브리드 검색 (StructSynthAgent 우선 + 기존 방식 fallback)"""
        try:
            # query 검증 추가
            if not query or not isinstance(query, str):
                logger.warning("하이브리드 검색 무효: query가 비어 있음")
                return []
            
            logger.info(f"=== _hybrid_search 시작: '{query}', top_k={top_k} ===")
            logger.info(f"structsynth_agent 존재 여부: {hasattr(self, 'structsynth_agent')}")
            logger.info(f"structsynth_agent 값: {self.structsynth_agent}")
        
            # FTS 검색 (SQLite)
            fts_results = self._fts_search(query, top_k)
            
            # FAISS 검색 (벡터 유사도)
            faiss_results = self._faiss_search(query, top_k)
            
            # None 체크 및 안전 처리
            if faiss_results is None:
                logger.warning("FAISS 검색 결과가 None입니다. 빈 리스트로 초기화합니다.")
                faiss_results = []
            
            # 결과 병합 및 중복 제거
            merged_results = self._merge_search_results(fts_results, faiss_results, top_k)
            
            logger.info(f"하이브리드 검색 완료: {len(merged_results)}개 결과")
            return merged_results
            
        except Exception as e:
            logger.error(f"하이브리드 검색 실패: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            return []
    
    def _fts_search(self, query: str, top_k: int) -> List[SearchResult]:
        """SQLite FTS 검색"""
        try:
            # query 검증 추가
            if not query or not isinstance(query, str):
                logger.warning("FTS 검색 무효: query가 비어 있음")
                return []
            
            # SQLite에서 키워드 기반 검색
            results = self.sqlite_store.search_symbols_fts(query, top_k)
            
            fts_results = []
            for result in results:
                search_result = SearchResult(
                    symbol_name=result.get("name", ""),
                    symbol_type=result.get("type", ""),
                    file_path=result.get("file_path", ""),
                    start_line=result.get("start_line", 0),
                    end_line=result.get("end_line", 0),
                    content=result.get("content", ""),
                    source="fts",
                    similarity_score=result.get("relevance_score", 0.0),
                    input_types=result.get("input_types"),
                    output_types=result.get("output_types"),
                    dependencies=result.get("dependencies"),
                    usage_examples=result.get("usage_examples")
                )
                fts_results.append(search_result)
            
            logger.info(f"FTS 검색 완료: {len(fts_results)}개 결과")
            return fts_results
            
        except Exception as e:
            logger.error(f"FTS 검색 실패: {e}")
            return []
    
    def _faiss_search(self, query: str, top_k: int) -> List[SearchResult]:
        """FAISS 벡터 유사도 검색"""
        try:
            # query 검증 추가
            if not query or not isinstance(query, str):
                logger.warning("FAISS 검색 무효: query가 비어 있음")
                return []
            
            logger.info(f"=== FAISS 검색 시작: '{query}', top_k={top_k} ===")
            
            # FAISS 인덱스 상태 로깅
            if hasattr(self, 'faiss_store'):
                logger.info(f"FAISS 인덱스 존재: {self.faiss_store is not None}")
                if self.faiss_store is not None:
                    logger.info(f"FAISS 인덱스 타입: {type(self.faiss_store)}")
                    logger.info(f"FAISS 인덱스 메서드: {[method for method in dir(self.faiss_store) if not method.startswith('_')]}")
            else:
                logger.info("FAISS 인덱스 속성이 없음")
            
            # 로컬 벡터 상태 로깅
            logger.info(f"로컬 벡터 존재: {self.vectors is not None}")
            if self.vectors is not None:
                logger.info(f"로컬 벡터 개수: {len(self.vectors)}")
                logger.info(f"로컬 벡터 차원: {self.vectors[0].shape if len(self.vectors) > 0 else 'N/A'}")
            
            logger.info(f"로컬 메타데이터 존재: {self.metadata is not None}")
            if self.metadata is not None:
                logger.info(f"로컬 메타데이터 개수: {len(self.metadata)}")
            
            # 1. FAISS 인덱스가 있으면 우선 사용
            if hasattr(self, 'faiss_store') and self.faiss_store is not None:
                logger.info("FAISS 인덱스를 사용한 검색 시도")
                return self._faiss_index_search(query, top_k)
            
            # 2. fallback: 로컬 벡터 배열 사용
            if self.vectors is not None and self.metadata is not None:
                logger.info("로컬 벡터 배열을 사용한 검색 시도")
                return self._local_vector_search(query, top_k)
            
            logger.warning("벡터 스토어가 로드되지 않았습니다")
            return []
            
        except Exception as e:
            logger.error(f"FAISS 검색 실패: {e}")
            return []
    
    def _faiss_index_search(self, query: str, top_k: int) -> List[SearchResult]:
        """FAISS 인덱스를 사용한 벡터 검색"""
        try:
            # query 검증 추가
            if not query or not isinstance(query, str):
                logger.warning("FAISS 인덱스 검색 무효: query가 비어 있음")
                return []
            
            # 쿼리 임베딩 생성
            query_embedding = self._create_embedding(query)
            if query_embedding is None:
                logger.warning("쿼리 임베딩 생성 실패")
                return []
            
            # FAISS 인덱스에서 검색
            logger.info(f"FAISS 인덱스 검색 시도 - 임베딩 차원: {query_embedding.shape}")
            
            if hasattr(self.faiss_store, 'search_vector'):
                logger.info("search_vector 메서드 사용")
                # FAISSStore의 search_vector 메서드 사용 (numpy 배열 지원)
                search_results = self.faiss_store.search_vector(query_embedding.tolist(), top_k)
                logger.info(f"search_vector 결과: {len(search_results) if search_results else 0}개")
                
                # search_vector 결과를 SearchResult로 변환
                if search_results:
                    faiss_results = []
                    for result in search_results:
                        # result에서 doc_id와 similarity 추출
                        doc_id = result.get('doc_id', 0)
                        similarity = result.get('similarity', 0.0)
                        
                        # chunks 테이블에서 메타데이터 조회
                        chunk_info = self.sqlite_store.get_chunk_info(doc_id)
                        if chunk_info:
                            search_result = SearchResult(
                                symbol_name=chunk_info.get('symbol_name', ''),
                                symbol_type=chunk_info.get('symbol_type', ''),
                                file_path=chunk_info.get('file_path', ''),
                                start_line=chunk_info.get('start_line', 0),
                                end_line=chunk_info.get('end_line', 0),
                                content=chunk_info.get('content', ''),
                                source="faiss_index",
                                similarity_score=float(similarity),
                                input_types=chunk_info.get('input_types'),
                                output_types=chunk_info.get('output_types'),
                                dependencies=chunk_info.get('dependencies'),
                                usage_examples=chunk_info.get('usage_examples')
                            )
                            faiss_results.append(search_result)
                    
                    logger.info(f"search_vector 변환 완료: {len(faiss_results)}개 결과")
                    return faiss_results
                else:
                    logger.warning("search_vector 결과가 비어있습니다")
                    return []
                    
            elif hasattr(self.faiss_store, 'search'):
                logger.info("search 메서드 사용 (fallback)")
                # fallback: search 메서드 사용
                search_results = self.faiss_store.search(query_embedding.tolist(), top_k)
                logger.info(f"search 결과: {len(search_results) if search_results else 0}개")
                
                # SearchResult로 변환
                faiss_results = []
                for result in search_results:
                    # result에서 doc_id와 similarity 추출
                    doc_id = result.get('doc_id', 0)
                    similarity = result.get('similarity', 0.0)
                    
                    # chunks 테이블에서 메타데이터 조회
                    chunk_info = self.sqlite_store.get_chunk_info(doc_id)
                    if chunk_info:
                        search_result = SearchResult(
                            symbol_name=chunk_info.get('symbol_name', ''),
                            symbol_type=chunk_info.get('symbol_type', ''),
                            file_path=chunk_info.get('file_path', ''),
                            start_line=chunk_info.get('start_line', 0),
                            end_line=chunk_info.get('end_line', 0),
                            content=chunk_info.get('content', ''),
                            source="faiss_index",
                            similarity_score=float(similarity),
                            input_types=chunk_info.get('input_types'),
                            output_types=chunk_info.get('output_types'),
                            dependencies=chunk_info.get('dependencies'),
                            usage_examples=chunk_info.get('usage_examples')
                        )
                        faiss_results.append(search_result)
                
                logger.info(f"FAISS 인덱스 검색 완료: {len(faiss_results)}개 결과")
                return faiss_results
            else:
                logger.warning("FAISS 인덱스에 search 메서드가 없습니다")
                return []
                
        except Exception as e:
            logger.error(f"FAISS 인덱스 검색 실패: {e}")
            return []
    
    def _local_vector_search(self, query: str, top_k: int) -> List[SearchResult]:
        """로컬 벡터 배열을 사용한 검색 (fallback)"""
        try:
            # query 검증 추가
            if not query or not isinstance(query, str):
                logger.warning("로컬 벡터 검색 무효: query가 비어 있음")
                return []
            
            # numpy 배열 비교 문제 방지를 위해 명시적 체크
            if len(self.vectors) == 0 or len(self.metadata) == 0:
                logger.warning("벡터 스토어가 비어있습니다")
                return []
            
            # 쿼리 임베딩 생성
            query_embedding = self._create_embedding(query)
            if query_embedding is None:
                logger.warning("쿼리 임베딩 생성 실패")
                return []
            
            # 벡터 유사도 계산
            logger.info(f"로컬 벡터 유사도 계산 시작 - 총 {len(self.vectors)}개 벡터")
            similarities = []
            for i, vector in enumerate(self.vectors):
                try:
                    # numpy 배열을 리스트로 변환
                    if isinstance(vector, np.ndarray):
                        vector = vector.tolist()
                    similarity = self._cosine_similarity(query_embedding, vector)
                    similarities.append((i, similarity))
                    
                    # 진행 상황 로깅 (100개마다)
                    if (i + 1) % 100 == 0:
                        logger.info(f"벡터 유사도 계산 진행: {i + 1}/{len(self.vectors)}")
                        
                except Exception as e:
                    logger.warning(f"벡터 {i} 유사도 계산 실패: {e}")
                    continue
            
            if not similarities:
                logger.warning("유사도 계산 결과가 없습니다")
                return []
            
            # 유사도 순으로 정렬
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # 상위 결과 추출
            faiss_results = []
            for i, (idx, similarity) in enumerate(similarities[:top_k]):
                if idx < len(self.metadata):
                    metadata = self.metadata[idx]
                    search_result = SearchResult(
                        symbol_name=metadata.get("symbol_name", ""),
                        symbol_type=metadata.get("symbol_type", ""),
                        file_path=metadata.get("file_path", ""),
                        start_line=metadata.get("start_line", 0),
                        end_line=metadata.get("end_line", 0),
                        content=metadata.get("content", ""),
                        source="faiss_local",
                        similarity_score=float(similarity),
                        input_types=metadata.get("input_types"),
                        output_types=metadata.get("output_types"),
                        dependencies=metadata.get("dependencies"),
                        usage_examples=metadata.get("usage_examples")
                    )
                    faiss_results.append(search_result)
            
            logger.info(f"로컬 벡터 검색 완료: {len(faiss_results)}개 결과")
            return faiss_results
            
        except Exception as e:
            logger.error(f"로컬 벡터 검색 실패: {e}")
            return []
    
    def _merge_search_results(self, fts_results: List[SearchResult], 
                            faiss_results: List[SearchResult], top_k: int) -> List[SearchResult]:
        """검색 결과 병합 및 중복 제거"""
        try:
            # 결과 병합
            all_results = fts_results + faiss_results
            
            # 중복 제거 (파일 경로와 심볼 이름 기준)
            seen = set()
            unique_results = []
            
            for result in all_results:
                key = (result.file_path, result.symbol_name)
                if key not in seen:
                    seen.add(key)
                    unique_results.append(result)
            
            # 유사도 점수로 정렬
            unique_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # 상위 결과 반환
            return unique_results[:top_k]
            
        except Exception as e:
            logger.error(f"검색 결과 병합 실패: {e}")
            return fts_results[:top_k] if fts_results else []
    
    def _generate_rag_answer(self, query: str, search_results: List[SearchResult]) -> str:
        """RAG 기반 답변 생성"""
        try:
            if not search_results:
                return "관련 코드를 찾을 수 없습니다. 더 구체적인 질문을 해주세요."
            
            # OpenAI 클라이언트가 없으면 기본 답변 생성
            if not hasattr(self, 'openai_client') or self.openai_client is None:
                return self._generate_basic_answer(query, search_results)
            
            # 컨텍스트 구성
            context = self._build_context(search_results)
            
            # 프롬프트 구성
            prompt = self._build_prompt(query, context)
            
            # LLM 호출
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "당신은 코드 분석 전문가입니다. 제공된 코드 컨텍스트를 바탕으로 확신을 가지고 명확한 답변을 제공하세요. 추측성 표현('아마도', '추정됩니다', '생각됩니다', '것 같습니다')을 사용하지 말고, 단정적이고 확신 있는 표현을 사용하세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1  # 더 확정적인 답변을 위해 낮춤
            )
            
            answer = response.choices[0].message.content
            logger.info("RAG 답변 생성 완료")
            return answer
            
        except Exception as e:
            logger.error(f"RAG 답변 생성 실패: {e}")
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def _build_context(self, search_results: List[SearchResult]) -> str:
        """검색 결과로부터 컨텍스트 구성 (위치, 사용법, 활용 정보 포함)"""
        context_parts = []
        
        logger.info(f"컨텍스트 구성 시작: {len(search_results)}개 검색 결과")
        
        for i, result in enumerate(search_results, 1):
            logger.info(f"결과 {i} 디버깅: file_path={result.file_path}, symbol_name={result.symbol_name}, content_length={len(result.content) if result.content else 0}")
            
            context_part = f"""
            [참조 {i}]
            위치: {result.file_path}의 {result.start_line}-{result.end_line} 라인
            심볼: {result.symbol_name} ({result.symbol_type})
            
            코드 내용:
            {result.content}
            """
            
            # 추가 메타데이터가 있으면 포함
            if result.input_types or result.output_types or result.dependencies or result.usage_examples:
                context_part += "\n추가 정보:\n"
                
                if result.input_types:
                    context_part += f"입력: {result.input_types}\n"
                if result.output_types:
                    context_part += f"출력: {result.output_types}\n"
                if result.dependencies:
                    deps = ", ".join(result.dependencies) if isinstance(result.dependencies, list) else str(result.dependencies)
                    context_part += f"의존성: {deps}\n"
                if result.usage_examples:
                    context_part += f"사용 예시: {result.usage_examples}\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """프롬프트 구성 (자유로운 대화와 깊이 있는 분석 지원)"""
        return f"""
        사용자 질문: {query}
        
        다음 코드 컨텍스트를 참조하여 답변해주세요:
        {context}
        
        답변 요구사항:
        1. 제공된 코드 컨텍스트를 바탕으로 확신을 가지고 답변하세요
        2. 추측성 표현('아마도', '추정됩니다', '생각됩니다', '것 같습니다', '일 수 있습니다')을 절대 사용하지 마세요
        3. 단정적이고 명확한 표현을 사용하세요 ('이 코드는 ~합니다', '이 함수는 ~를 수행합니다', '이것은 ~입니다')
        4. 코드의 위치, 사용법, 기능을 구체적으로 설명하세요
        5. 필요한 경우 추가적인 세부사항이나 예시를 포함하세요
        6. 전문가로서 확신에 찬 답변을 제공하세요
        7. 한국어로 답변하세요
        """
    
    def _create_embedding(self, text: str) -> Optional[np.ndarray]:
        """텍스트 임베딩 생성"""
        try:
            # 환경변수에서 임베딩 모델 가져오기
            embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
            
            # Azure OpenAI 클라이언트 사용
            response = self.openai_client.embeddings.create(
                model=embedding_model,
                input=text
            )
            
            embedding_array = np.array(response.data[0].embedding)
            logger.info(f"임베딩 생성 성공: 차원={embedding_array.shape}")
            
            return embedding_array
            
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            logger.info("임베딩 실패로 인해 벡터 검색을 건너뜁니다.")
            # 임베딩 실패 시 None 반환하여 벡터 검색 건너뛰기
            return None
    
    def _cosine_similarity(self, vec1, vec2) -> float:
        """코사인 유사도 계산 (numpy 배열 또는 리스트 지원)"""
        try:
            # numpy 배열 또는 리스트를 numpy 배열로 변환
            vec1 = np.asarray(vec1, dtype=np.float64)
            vec2 = np.asarray(vec2, dtype=np.float64)
            
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            # numpy 배열 비교 대신 float 비교 사용
            if float(norm1) == 0.0 or float(norm2) == 0.0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except Exception as e:
            logger.error(f"코사인 유사도 계산 실패: {e}")
            return 0.0
    
    def _calculate_confidence(self, search_results: List[SearchResult]) -> float:
        """응답 신뢰도 계산"""
        if not search_results:
            return 0.0
        
        # 검색 결과의 평균 유사도 점수로 신뢰도 계산
        avg_similarity = sum(result.similarity_score for result in search_results) / len(search_results)
        return min(avg_similarity, 1.0)
    
    def get_search_stats(self) -> Dict[str, Any]:
        """검색 통계 정보 반환"""
        try:
            stats = {
                "sqlite_available": hasattr(self, 'sqlite_store'),
                "openai_available": hasattr(self, 'openai_client'),
                "vector_search": {
                    "sqlite_embeddings": {
                        "loaded": self.vectors is not None and self.metadata is not None,
                        "total_vectors": len(self.vectors) if self.vectors is not None else 0,
                        "total_metadata": len(self.metadata) if self.metadata is not None else 0
                    },
                    "faiss_index": {
                        "loaded": hasattr(self, 'faiss_store') and self.faiss_store is not None,
                        "index_path": str(self.artifacts_dir / "faiss.index") if hasattr(self, 'artifacts_dir') else None,
                        "index_exists": (self.artifacts_dir / "faiss.index").exists() if hasattr(self, 'artifacts_dir') else False
                    }
                }
            }
            
            # SQLite 통계 추가
            if hasattr(self, 'sqlite_store'):
                try:
                    db_stats = self.sqlite_store.get_database_stats()
                    stats["sqlite_stats"] = db_stats
                except Exception as e:
                    stats["sqlite_stats"] = {"error": str(e)}
            
            return stats
            
        except Exception as e:
            logger.error(f"검색 통계 조회 실패: {e}")
            return {"error": str(e)}
    
    def _get_available_artifacts(self) -> str:
        """사용 가능한 산출물 목록을 동적으로 가져와서 프롬프트용 문자열 생성"""
        try:
            from tools.insightgen_tools import get_artifact_summary
            
            artifact_summary = get_artifact_summary.invoke({"data_dir": str(self.data_dir)})
            
            if "error" in artifact_summary or not artifact_summary.get("summaries"):
                return "   - 사용 가능한 산출물이 없습니다. InsightGen을 먼저 실행하세요."
            
            artifact_lines = []
            for summary in artifact_summary["summaries"]:
                name = summary["name"]
                preview = summary["preview"]
                # 프리뷰에서 첫 번째 줄이나 주요 키워드 추출
                description = preview.split('\n')[0][:100] + "..."
                artifact_lines.append(f"   - {name}: {description}")
            
            return "\n".join(artifact_lines)
            
        except Exception as e:
            logger.warning(f"산출물 목록 조회 실패: {e}")
            return "   - 산출물 조회 실패 (InsightGen을 먼저 실행하세요)"

    def _get_first_available_artifact(self) -> Optional[str]:
        """첫 번째 사용 가능한 산출물 파일명 반환"""
        try:
            from tools.insightgen_tools import list_artifacts
            
            artifacts_info = list_artifacts.invoke({"data_dir": str(self.data_dir)})
            
            # artifacts_info가 None인 경우 처리
            if artifacts_info is None:
                logger.warning("artifacts_info가 None입니다")
                return None
            
            if "error" not in artifacts_info and artifacts_info.get("artifacts"):
                artifacts = artifacts_info.get("artifacts", [])
                if not artifacts:
                    logger.warning("artifacts 리스트가 비어있습니다")
                    return None
                
                # MD 파일 우선, 순서대로 정렬된 첫 번째 파일
                md_files = []
                for artifact in artifacts:
                    if artifact and isinstance(artifact, dict) and artifact.get("name", "").endswith('.md'):
                        md_files.append(artifact["name"])
                
                if md_files:
                    return md_files[0]
                
                # MD 파일이 없으면 첫 번째 파일
                first_artifact = artifacts[0]
                if first_artifact and isinstance(first_artifact, dict):
                    return first_artifact.get("name")
            
            # 사용 가능한 산출물이 없으면 None
            return None
            
        except Exception as e:
            logger.warning(f"첫 번째 산출물 조회 실패: {e}")
            return None

    def _analyze_query_and_select_tools(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """AI 기반 질문 분석 및 도구 선택"""
        try:
            logger.info(f"🤖 LLM 질문 분석 시작: '{query}'")
            
            # 컨텍스트 정보 준비
            context_str = ""
            if context:
                context_parts = []
                
                # 최근 대화 내용
                if context.get("recent_conversations"):
                    recent_queries = [conv["query"] for conv in context["recent_conversations"][-3:]]
                    context_parts.append(f"최근 질문들: {', '.join(recent_queries)}")
                
                # 언급된 심볼들
                if context.get("mentioned_symbols"):
                    context_parts.append(f"이전에 언급된 심볼들: {', '.join(context['mentioned_symbols'][:5])}")
                
                # 언급된 파일들
                if context.get("mentioned_files"):
                    context_parts.append(f"이전에 언급된 파일들: {', '.join(context['mentioned_files'][:3])}")
                
                context_str = "\n".join(context_parts) if context_parts else ""
            
            # 질문 분류 프롬프트 (컨텍스트 포함)
            classification_prompt = f"""
다음 질문을 분석하여 적절한 도구를 선택하세요:

질문: {query}

{context_str}

사용 가능한 도구:
1. get_artifact: 프로젝트 개요, 전체 구조, 흐름 등 전반적 설명
{self._get_available_artifacts()}
2. search_symbols_fts/semantic: 특정 심볼/키워드 검색
3. get_symbol/get_chunks_by_symbol: 심볼/청크 세부 정보
4. get_calls_from/get_calls_to: 호출 관계 분석
5. analyze_file_llm/analyze_symbol_llm: LLM 기반 코드 분석
6. get_source_code: 파일 경로와 라인 범위로 실제 소스코드 가져오기
7. analyze_source_code_with_llm: 실제 소스코드를 LLM으로 분석
8. get_function_source: 함수/클래스 이름으로 실제 소스코드 검색

이전 대화 내용을 참고하여 질문의 맥락을 파악하고 적절한 도구를 선택하세요. 

JSON 형식으로 응답:
{{
    "query_type": "overview|specific",
    "selected_tools": ["tool1", "tool2"],
    "reasoning": "선택 이유 (컨텍스트 고려사항 포함)",
    "artifact_name": "적절한 산출물 파일명 (위 목록에서 선택)"
}}
"""
            
            # LLM으로 도구 선택
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "당신은 코드 분석 도구 선택 전문가입니다. 질문을 정확히 분석하여 적절한 도구를 선택하세요."},
                    {"role": "user", "content": classification_prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            llm_response = response.choices[0].message.content
            logger.info(f"🤖 LLM 응답: {llm_response[:200]}...")
            
            result = self._parse_tool_selection(llm_response)
            logger.info(f"✅ 도구 선택 완료: {result['query_type']} -> {result['selected_tools']}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ 질문 분석 실패: {e}")
            logger.info("🔄 기본 검색 도구로 fallback")
            # 기본값으로 하이브리드 검색 사용
            return {
                "query_type": "specific",
                "selected_tools": ["search_symbols_fts", "search_symbols_semantic"],
                "reasoning": "분석 실패로 기본 검색 사용",
                "artifact_name": None
            }
    
    def _parse_tool_selection(self, llm_response: str) -> Dict[str, Any]:
        """LLM 응답을 파싱하여 도구 선택 정보 추출"""
        # 기본값 정의
        default_result = {
            "query_type": "specific",
            "selected_tools": ["search_symbols_fts"],
            "reasoning": "기본 검색 사용",
            "artifact_name": None
        }
        
        try:
            import json
            import re
            
            logger.info("🔍 LLM 응답 파싱 시작")
            
            # llm_response가 None이거나 빈 문자열인 경우 처리
            if not llm_response or not isinstance(llm_response, str):
                logger.warning("❌ LLM 응답이 비어있거나 문자열이 아님")
                return default_result
            
            # JSON 부분 추출
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group())
                    logger.info(f"✅ JSON 파싱 성공: {data}")
                    
                    # data가 None이거나 dict가 아닌 경우 처리
                    if not data or not isinstance(data, dict):
                        logger.warning("❌ 파싱된 데이터가 올바른 형식이 아님")
                        return default_result
                    
                    return {
                        "query_type": data.get("query_type", "specific"),
                        "selected_tools": data.get("selected_tools", ["search_symbols_fts"]),
                        "reasoning": data.get("reasoning", ""),
                        "artifact_name": data.get("artifact_name", self._get_first_available_artifact())
                    }
                except json.JSONDecodeError as e:
                    logger.warning(f"❌ JSON 디코딩 실패: {e}")
                    return default_result
            else:
                # JSON 파싱 실패 시 기본값
                logger.warning("❌ JSON 패턴을 찾을 수 없음 - 기본값 사용")
                return default_result
                
        except Exception as e:
            logger.error(f"❌ 도구 선택 파싱 실패: {e}")
            return default_result
    
    def _handle_overview_query(self, query: str, tool_selection: Dict[str, Any]) -> ChatResponse:
        """개요/전반적 질문 처리"""
        try:
            # get_artifact 도구 사용
            from tools.insightgen_tools import get_artifact
            
            artifact_name = tool_selection.get("artifact_name") or self._get_first_available_artifact()
            logger.info(f"📁 산출물 로드 시작: {artifact_name}")
            
            artifact_result = get_artifact.invoke({
                "artifact_name": artifact_name,
                "data_dir": str(self.data_dir)
            })
            
            if "error" in artifact_result:
                # 산출물이 없으면 사용 가능한 산출물 목록 확인
                logger.warning(f"❌ 산출물 로드 실패: {artifact_result['error']}")
                
                # 사용 가능한 산출물 목록 조회
                from tools.insightgen_tools import list_artifacts
                available_artifacts = list_artifacts.invoke({"data_dir": str(self.data_dir)})
                
                if "artifacts" in available_artifacts and available_artifacts["artifacts"]:
                    # 첫 번째 사용 가능한 산출물 사용
                    first_artifact = available_artifacts["artifacts"][0]["name"]
                    logger.info(f"🔄 대체 산출물 사용: {first_artifact}")
                    
                    artifact_result = get_artifact.invoke({
                        "artifact_name": first_artifact,
                        "data_dir": str(self.data_dir)
                    })
                    
                    if "error" in artifact_result:
                        logger.warning(f"❌ 대체 산출물도 실패: {artifact_result['error']}")
                        logger.info("🔄 기본 검색으로 fallback")
                        return self._handle_specific_query(query, tool_selection, 10)
                else:
                    logger.warning("❌ 사용 가능한 산출물이 없습니다")
                    logger.info("🔄 기본 검색으로 fallback")
                    return self._handle_specific_query(query, tool_selection, 10)
            
            # 산출물 로드 성공 로깅
            artifact_content = artifact_result.get("content", "")
            logger.info(f"✅ 산출물 로드 성공: {len(artifact_content)}자")
            logger.info(f"📄 산출물 경로: {artifact_result.get('path', 'unknown')}")
            
            # 사용자 질문에 맞는 답변 생성
            logger.info("🤖 LLM 기반 개요 답변 생성 시작")
            answer = self._generate_overview_answer(query, artifact_content)
            logger.info("✅ 개요 답변 생성 완료")
            
            # SearchResult 형태로 변환
            evidence = [SearchResult(
                symbol_name="프로젝트 개요",
                symbol_type="document",
                file_path=artifact_result.get("path", ""),
                start_line=0,
                end_line=0,
                content=artifact_content[:500] + "..." if len(artifact_content) > 500 else artifact_content,
                source="artifact",
                similarity_score=1.0
            )]
            
            return ChatResponse(
                answer=answer,
                evidence=evidence,
                confidence=0.9
            )
            
        except Exception as e:
            logger.error(f"개요 질문 처리 실패: {e}")
            # 실패 시 기본 검색으로 fallback
            return self._handle_specific_query(query, tool_selection, 10)
    
    def _handle_specific_query(self, query: str, tool_selection: Dict[str, Any], top_k: int) -> ChatResponse:
        """특정 코드 질문 처리"""
        try:
            logger.info(f"🔍 특정 코드 질문 처리 시작: {tool_selection.get('selected_tools', [])}")
            
            # 새로운 도구들을 사용해야 하는지 확인
            selected_tools = tool_selection.get("selected_tools", [])
            
            # 실제 소스코드 접근 도구들이 선택된 경우
            if any(tool in selected_tools for tool in ["get_source_code", "analyze_source_code_with_llm", "get_function_source"]):
                logger.info("🔧 실제 소스코드 접근 도구 사용")
                return self._handle_source_code_query(query, tool_selection, top_k)
            
            # 기존 하이브리드 검색 수행
            logger.info(f"🔍 하이브리드 검색 수행: top_k={top_k}")
            search_results = self._hybrid_search(query, top_k)
            
            if not search_results:
                logger.warning("❌ 검색 결과 없음")
                return ChatResponse(
                    answer="관련 코드를 찾을 수 없습니다. 더 구체적인 질문을 해주세요.",
                    evidence=[],
                    confidence=0.0
                )
            
            logger.info(f"✅ 검색 결과: {len(search_results)}개")
            
            # RAG 기반 답변 생성
            logger.info("🤖 RAG 기반 답변 생성 시작")
            answer = self._generate_rag_answer(query, search_results)
            logger.info("✅ RAG 답변 생성 완료")
            
            confidence = self._calculate_confidence(search_results)
            logger.info(f"📊 응답 신뢰도: {confidence:.2f}")
            
            return ChatResponse(
                answer=answer,
                evidence=search_results,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"특정 코드 질문 처리 실패: {e}")
            return ChatResponse(
                answer=f"코드 분석 중 오류가 발생했습니다: {str(e)}",
                evidence=[],
                confidence=0.0
            )
    
    def _handle_source_code_query(self, query: str, tool_selection: Dict[str, Any], top_k: int) -> ChatResponse:
        """실제 소스코드 접근 도구를 사용한 질문 처리"""
        try:
            # 안전한 import 처리
            try:
                from tools.llm_tools import get_function_source, analyze_source_code_with_llm, get_source_code
                logger.info("✅ 소스코드 도구 import 성공")
            except ImportError as e:
                logger.error(f"❌ 소스코드 도구 import 실패: {e}")
                # fallback to basic search
                search_results = self._hybrid_search(query, top_k)
                answer = self._generate_rag_answer(query, search_results)
                return ChatResponse(answer=answer, evidence=search_results, confidence=0.5)
            
            # tool_selection 안전 처리
            if not tool_selection or not isinstance(tool_selection, dict):
                logger.warning("tool_selection이 None이거나 dict가 아님")
                tool_selection = {"selected_tools": []}
            
            selected_tools = tool_selection.get("selected_tools", [])
            logger.info(f"🔧 소스코드 도구 사용: {selected_tools}")
            
            # 함수/클래스 이름을 질문에서 추출
            function_name = self._extract_function_name_from_query(query)
            file_path = self._extract_file_path_from_query(query)
            
            evidence = []
            analysis_results = []
            
            # get_function_source 도구 사용
            if "get_function_source" in selected_tools and function_name:
                logger.info(f"🔍 함수 소스코드 검색: {function_name}")
                
                try:
                    source_result = get_function_source(
                        symbol_name=function_name,
                        file_path=file_path,
                        data_dir=str(self.data_dir)
                    )
                    
                    # None 체크 추가
                    if source_result is None:
                        logger.warning("get_function_source 결과가 None입니다")
                        source_result = {"success": False, "error": "결과가 None"}
                    
                    if source_result.get("success") and source_result.get("results"):
                        for result in source_result["results"]:
                            if "error" not in result:
                                try:
                                    # line_range 안전 처리
                                    line_range = result.get("line_range", "1-1")
                                    if "-" in line_range:
                                        start_line = int(line_range.split("-")[0])
                                        end_line = int(line_range.split("-")[1])
                                    else:
                                        start_line = end_line = int(line_range)
                                    
                                    # SearchResult 형태로 변환
                                    search_result = SearchResult(
                                        symbol_name=result.get("symbol_name", "unknown"),
                                        symbol_type=result.get("symbol_type", "unknown"),
                                        file_path=result.get("file_path", ""),
                                        start_line=start_line,
                                        end_line=end_line,
                                        content=result.get("source_code", ""),
                                        source="source_code_tool",
                                        similarity_score=1.0
                                    )
                                    evidence.append(search_result)
                                    analysis_results.append(result)
                                except Exception as e:
                                    logger.warning(f"SearchResult 변환 실패: {e}")
                                    continue
                    else:
                        logger.warning(f"함수 소스코드 검색 실패: {source_result}")
                        
                except Exception as e:
                    logger.error(f"get_function_source 호출 실패: {e}")
            
            # analyze_source_code_with_llm 도구 사용
            if "analyze_source_code_with_llm" in selected_tools:
                logger.info(f"🤖 소스코드 LLM 분석 수행")
                
                # 이미 찾은 소스코드가 있으면 분석
                for result in analysis_results:
                    try:
                        # line_range 안전 처리
                        line_range = result.get("line_range", "1-1")
                        if "-" in line_range:
                            start_line = int(line_range.split("-")[0])
                            end_line = int(line_range.split("-")[1])
                        else:
                            start_line = end_line = int(line_range)
                            
                        analysis = analyze_source_code_with_llm(
                            file_path=result.get("file_path", ""),
                            start_line=start_line,
                            end_line=end_line,
                            question=query,
                            data_dir=str(self.data_dir)
                        )
                        
                        # None 체크 추가
                        if analysis is None:
                            logger.warning("analyze_source_code_with_llm 결과가 None입니다")
                            continue
                        
                        if analysis.get("success"):
                            # 분석 결과를 SearchResult에 추가
                            for i, ev in enumerate(evidence):
                                if ev.file_path == result.get("file_path", ""):
                                    evidence[i] = SearchResult(
                                        symbol_name=ev.symbol_name,
                                        symbol_type=ev.symbol_type,
                                        file_path=ev.file_path,
                                        start_line=ev.start_line,
                                        end_line=ev.end_line,
                                        content=ev.content,
                                        source="llm_analyzed_source",
                                        similarity_score=1.0,
                                        usage_examples=analysis.get("analysis", "")
                                    )
                                    break
                    except Exception as e:
                        logger.error(f"소스코드 LLM 분석 실패: {e}")
                        continue
            
            # get_source_code 도구 직접 사용 (파일 경로가 명시된 경우)
            if "get_source_code" in selected_tools and file_path:
                logger.info(f"📁 파일 소스코드 직접 접근: {file_path}")
                
                try:
                    source_result = get_source_code(
                        file_path=file_path,
                        data_dir=str(self.data_dir)
                    )
                    
                    # None 체크 추가
                    if source_result is None:
                        logger.warning("get_source_code 결과가 None입니다")
                        source_result = {"success": False, "error": "결과가 None"}
                    
                    if source_result.get("success"):
                        search_result = SearchResult(
                            symbol_name="파일 전체",
                            symbol_type="file",
                            file_path=source_result.get("file_path", file_path),
                            start_line=source_result.get("start_line", 1),
                            end_line=source_result.get("end_line", 1),
                            content=source_result.get("source_code", ""),
                            source="direct_file_access",
                            similarity_score=1.0
                        )
                        evidence.append(search_result)
                    else:
                        logger.warning(f"파일 소스코드 직접 접근 실패: {source_result}")
                        
                except Exception as e:
                    logger.error(f"get_source_code 호출 실패: {e}")
            
            # fallback: 기본 검색도 수행
            if not evidence:
                logger.info("🔄 실제 소스코드를 찾지 못함, 기본 검색 수행")
                search_results = self._hybrid_search(query, top_k)
                evidence.extend(search_results)
            
            if not evidence:
                return ChatResponse(
                    answer="요청하신 소스코드를 찾을 수 없습니다. 함수/클래스 이름이나 파일 경로를 더 구체적으로 명시해주세요.",
                    evidence=[],
                    confidence=0.0
                )
            
            # 실제 소스코드를 포함한 향상된 RAG 답변 생성
            logger.info("🤖 실제 소스코드 기반 RAG 답변 생성")
            answer = self._generate_enhanced_rag_answer(query, evidence, analysis_results)
            
            return ChatResponse(
                answer=answer,
                evidence=evidence,
                confidence=0.95
            )
            
        except Exception as e:
            logger.error(f"소스코드 질문 처리 실패: {e}")
            # fallback to basic search
            search_results = self._hybrid_search(query, top_k)
            answer = self._generate_rag_answer(query, search_results)
            return ChatResponse(
                answer=answer,
                evidence=search_results,
                confidence=0.5
            )
    
    def _extract_function_name_from_query(self, query: str) -> Optional[str]:
        """질문에서 함수/클래스 이름 추출"""
        import re
        
        # 일반적인 패턴들로 함수명 추출 시도
        patterns = [
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:함수|메서드|클래스)',
            r'([a-zA-Z_][a-zA-Z0-9_]*)\(\)',
            r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)',
            r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:에\s*대해|에서|를|을|의)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                function_name = match.group(1)
                # 일반적인 키워드들은 제외
                if function_name not in ['def', 'class', 'import', 'from', 'return', 'if', 'for', 'while']:
                    logger.info(f"추출된 함수명: {function_name}")
                    return function_name
        
        return None
    
    def _extract_file_path_from_query(self, query: str) -> Optional[str]:
        """질문에서 파일 경로 추출"""
        import re
        
        # 파일 경로 패턴들
        patterns = [
            r'([a-zA-Z0-9_./\\]+\.py)',
            r'([a-zA-Z0-9_./\\]+\.java)',
            r'([a-zA-Z0-9_./\\]+\.cpp)',
            r'([a-zA-Z0-9_./\\]+\.c)',
            r'([a-zA-Z0-9_./\\]+\.h)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                file_path = match.group(1)
                logger.info(f"추출된 파일 경로: {file_path}")
                return file_path
        
        return None
    
    def _generate_enhanced_rag_answer(self, query: str, evidence: List[SearchResult], analysis_results: List[Dict]) -> str:
        """실제 소스코드를 포함한 향상된 RAG 답변 생성"""
        try:
            # evidence와 analysis_results가 None이 아닌지 확인
            if evidence is None:
                evidence = []
            if analysis_results is None:
                analysis_results = []
                
            # 기본 컨텍스트 구성
            context = self._build_context(evidence)
            
            # 분석 결과가 있으면 추가
            analysis_context = ""
            if analysis_results:
                analysis_parts = []
                for analysis in analysis_results:
                    if analysis and isinstance(analysis, dict) and analysis.get("analysis"):
                        analysis_parts.append(f"""
[LLM 분석 결과]
파일: {analysis.get('file_path', 'unknown')}
라인: {analysis.get('line_range', 'unknown')}
분석 내용:
{analysis['analysis']}
""")
                analysis_context = "\n".join(analysis_parts)
            
            # 향상된 프롬프트 구성
            enhanced_prompt = f"""
사용자 질문: {query}

다음은 실제 소스코드와 분석 결과입니다:

{context}

{analysis_context}

답변 요구사항:
1. 실제 소스코드를 바탕으로 확신을 가지고 정확한 답변을 제공하세요
2. 추측성 표현('아마도', '추정됩니다', '생각됩니다', '것 같습니다')을 절대 사용하지 마세요
3. 단정적이고 명확한 표현을 사용하세요 ('이 함수는 ~를 수행합니다', '이 코드는 ~합니다')
4. 코드의 구체적인 동작과 구현 내용을 확신 있게 설명하세요
5. 함수/클래스의 역할과 목적을 명확히 단언하세요
6. 코드의 입출력, 매개변수, 반환값 등 상세 정보를 구체적으로 제시하세요
7. 실제 코드 예시나 중요한 코드 라인을 확신 있게 인용하세요
8. 코드의 흐름과 로직을 단계별로 명확히 설명하세요
9. 전문가로서 확신에 찬 답변을 제공하세요
10. 한국어로 답변하세요

실제 소스코드를 분석했으므로 추측 없이 정확하고 구체적인 정보를 제공합니다.
"""
            
            # LLM 호출
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "당신은 실제 소스코드를 분석하는 전문가입니다. 제공된 실제 코드를 바탕으로 확신을 가지고 명확하고 상세한 답변을 제공하세요. 추측성 표현을 사용하지 말고 단정적으로 답변하세요."},
                    {"role": "user", "content": enhanced_prompt}
                ],
                max_tokens=2000,
                temperature=0.1  # 더 확정적인 답변을 위해 낮춤
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"향상된 RAG 답변 생성 실패: {e}")
            # fallback to basic RAG
            return self._generate_rag_answer(query, evidence)
    
    def _generate_overview_answer(self, query: str, artifact_content: str) -> str:
        """산출물 내용을 바탕으로 개요 답변 생성"""
        try:
            prompt = f"""
사용자 질문: {query}

다음은 프로젝트 개요 문서입니다:
{artifact_content}

사용자의 질문에 맞는 답변을 제공하세요:
1. 제공된 문서를 바탕으로 확신을 가지고 답변하세요
2. 추측성 표현('아마도', '추정됩니다', '생각됩니다', '것 같습니다')을 사용하지 마세요
3. 단정적이고 명확한 표현을 사용하세요 ('이 프로젝트는 ~입니다', '시스템은 ~를 수행합니다')
4. 질문의 의도에 맞는 정보를 찾아서 확신 있게 답변하세요
5. 불필요한 수사는 배제하고 사실 기반으로 답변하세요
6. 전문가로서 확신에 찬 답변을 제공하세요
7. 한국어로 자연스럽게 답변하세요
8. 관련 정보가 없으면 "해당 정보가 문서에 포함되어 있지 않습니다"라고 명확히 답변하세요
"""
            
            response = self.openai_client.chat.completions.create(
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "당신은 코드베이스 분석 전문가입니다. 프로젝트 개요 문서를 바탕으로 확신을 가지고 명확한 답변을 제공하세요. 추측성 표현을 사용하지 말고 단정적으로 답변하세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=800,
                temperature=0.1  # 더 확정적인 답변을 위해 낮춤
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"개요 답변 생성 실패: {e}")
            # LLM 실패 시 기본 답변
            return f"프로젝트 개요 정보를 찾았지만, 구체적인 답변을 생성할 수 없습니다. 원본 문서를 확인해주세요.\n\n{artifact_content[:300]}..."
