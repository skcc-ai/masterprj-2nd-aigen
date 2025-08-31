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
    
    def chat(self, query: str, top_k: int = 10) -> ChatResponse:
        """채팅 응답 생성"""
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
            logger.info(f"채팅 요청 처리 시작: '{query}'")
            
            # 1. 일반적인 질문인지 감지
            if self._detect_general_query(query):
                logger.info("일반적인 질문 감지 - 전체 분석 모드로 전환")
                answer = self._handle_general_analysis(query)
                return ChatResponse(
                    answer=answer,
                    evidence=[],
                    confidence=0.8
                )
            
            # 2. 하이브리드 검색 수행
            logger.info("하이브리드 검색 시작")
            search_results = self._hybrid_search(query, top_k)
            logger.info(f"하이브리드 검색 결과: {len(search_results)}개")
            
            if not search_results:
                logger.info("검색 결과가 없습니다")
                return ChatResponse(
                    answer="관련 코드를 찾을 수 없습니다. 더 구체적인 질문을 해주세요.",
                    evidence=[],
                    confidence=0.0
                )
            
            # 3. RAG 기반 답변 생성
            answer = self._generate_rag_answer(query, search_results)
            
            # 4. 응답 구성
            response = ChatResponse(
                answer=answer,
                evidence=search_results,
                confidence=self._calculate_confidence(search_results)
            )
            
            logger.info("채팅 응답 생성 완료")
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
            
            # 1. StructSynthAgent 검색 우선 시도 (개선된 검색 기능)
            if hasattr(self, 'structsynth_agent') and self.structsynth_agent is not None:
                try:
                    logger.info("StructSynthAgent를 사용한 검색 시도")
                    structsynth_results = self.structsynth_agent.search_symbols(query, top_k)
                    
                    logger.info(f"StructSynthAgent 검색 결과: {len(structsynth_results)}개")
                    if structsynth_results:
                        logger.info(f"첫 번째 결과 타입: {type(structsynth_results[0])}")
                        logger.info(f"첫 번째 결과 내용: {structsynth_results[0]}")
                    
                    if structsynth_results:
                        # StructSynthAgent 결과를 SearchResult로 변환
                        search_results = []
                        for i, result in enumerate(structsynth_results):
                            try:
                                # result가 딕셔너리인지 확인
                                if not isinstance(result, dict):
                                    logger.warning(f"결과 {i+1}이 딕셔너리가 아님: {type(result)}")
                                    continue
                                
                                # symbol_info 추출 (여러 가능한 키 시도)
                                symbol_info = result.get("symbol_info", {})
                                if not symbol_info:
                                    # result 자체가 symbol_info일 수 있음
                                    symbol_info = result
                                
                                # 필수 필드 추출 및 기본값 설정
                                symbol_name = symbol_info.get("name", "unknown")
                                symbol_type = symbol_info.get("type", "unknown")
                                file_path = symbol_info.get("file_path", "unknown")
                                start_line = symbol_info.get("start_line", 0)
                                end_line = symbol_info.get("end_line", 0)
                                content = result.get("chunk_content", result.get("content", ""))
                                similarity = result.get("similarity", 0.0)
                                
                                # file_path가 None인 경우 처리
                                if file_path is None:
                                    file_path = "unknown"
                                
                                search_result = SearchResult(
                                    symbol_name=symbol_name,
                                    symbol_type=symbol_type,
                                    file_path=str(file_path),
                                    start_line=int(start_line),
                                    end_line=int(end_line),
                                    content=str(content),
                                    source="structsynth",
                                    similarity_score=float(similarity)
                                )
                                search_results.append(search_result)
                                logger.info(f"결과 {i+1} 변환 완료: {search_result.symbol_name} ({search_result.file_path})")
                                
                            except Exception as e:
                                logger.warning(f"결과 {i+1} 변환 실패: {e}")
                                continue
                        
                        if search_results:
                            logger.info(f"StructSynthAgent 검색 완료: {len(search_results)}개 결과")
                            return search_results
                        else:
                            logger.warning("StructSynthAgent 결과 변환 실패 - 빈 결과")
                    
                except Exception as e:
                    logger.warning(f"StructSynthAgent 검색 실패: {e}")
                    import traceback
                    logger.warning(f"상세 오류: {traceback.format_exc()}")
            
            # 2. 기존 방식으로 fallback (FTS + FAISS)
            logger.info("기존 검색 방식으로 fallback")
            
            # FTS 검색 (SQLite)
            fts_results = self._fts_search(query, top_k)
            
            # FAISS 검색 (벡터 유사도)
            faiss_results = self._faiss_search(query, top_k)
            
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
                    similarity_score=result.get("relevance_score", 0.0)
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
            
            # 1. FAISS 인덱스가 있으면 우선 사용
            if hasattr(self, 'faiss_store') and self.faiss_store is not None:
                return self._faiss_index_search(query, top_k)
            
            # 2. fallback: 로컬 벡터 배열 사용
            if self.vectors is not None and self.metadata is not None:
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
            if hasattr(self.faiss_store, 'search'):
                # FAISSStore의 search 메서드 사용
                search_results = self.faiss_store.search(query_embedding, top_k)
                
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
                            similarity_score=float(similarity)
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
            similarities = []
            for i, vector in enumerate(self.vectors):
                try:
                    similarity = self._cosine_similarity(query_embedding, vector)
                    similarities.append((i, similarity))
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
                        similarity_score=float(similarity)
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
                    {"role": "system", "content": "당신은 코드 분석 전문가입니다. 제공된 코드 컨텍스트를 바탕으로 정확하고 유용한 답변을 제공하세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            answer = response.choices[0].message.content
            logger.info("RAG 답변 생성 완료")
            return answer
            
        except Exception as e:
            logger.error(f"RAG 답변 생성 실패: {e}")
            return f"답변 생성 중 오류가 발생했습니다: {str(e)}"
    
    def _build_context(self, search_results: List[SearchResult]) -> str:
        """검색 결과로부터 컨텍스트 구성"""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            context_part = f"""
            [참조 {i}]
            파일: {result.file_path}
            심볼: {result.symbol_name} ({result.symbol_type})
            라인: {result.start_line}-{result.end_line}
            내용:
            {result.content}
            """
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """프롬프트 구성"""
        return f"""
        사용자 질문: {query}
        
        다음 코드 컨텍스트를 참조하여 답변해주세요:
        {context}
        
        요구사항:
        1. 코드 컨텍스트를 바탕으로 정확한 답변을 제공하세요
        2. 관련 코드의 위치와 기능을 명확히 설명하세요
        3. 필요시 코드 예시를 포함하세요
        4. 한국어로 답변하세요
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
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """코사인 유사도 계산"""
        try:
            # numpy 배열을 float로 변환하여 비교 문제 해결
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
