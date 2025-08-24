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
            deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
            
            if not api_key or not endpoint:
                raise ValueError("Azure OpenAI API 키와 엔드포인트가 필요합니다")
            
            self.openai_client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version
            )
            self.deployment_name = deployment_name
            
            logger.info("Azure OpenAI 클라이언트 초기화 완료")
            
        except Exception as e:
            logger.error(f"Azure OpenAI 클라이언트 초기화 실패: {e}")
            raise
    
    def _init_data_sources(self):
        """데이터 소스 초기화"""
        try:
            # SQLite 데이터베이스 초기화
            self.sqlite_store = SQLiteStore(self.data_dir / "structsynth_code.db")
            
            # 벡터 스토어 로드
            self._load_vector_store()
            
            logger.info("데이터 소스 초기화 완료")
            
        except Exception as e:
            logger.error(f"데이터 소스 초기화 실패: {e}")
            raise
    
    def _load_vector_store(self):
        """벡터 스토어 로드"""
        try:
            # 여러 가능한 경로 시도
            possible_paths = [
                self.artifacts_dir,                    # artifacts_dir 자체
                self.artifacts_dir / "vector_store",   # vector_store 하위 디렉토리
                Path("./artifacts"),                   # artifacts 바로 아래
                Path("./artifacts/vector_store"),      # vector_store 하위
                Path("/app/artifacts"),                # Docker 절대 경로
                Path("/app/artifacts/vector_store"),   # Docker 절대 경로 + vector_store
                Path("artifacts"),                     # 상대 경로
                Path("artifacts/vector_store")         # 상대 경로 + vector_store
            ]
            
            vector_store_path = None
            for path in possible_paths:
                if path.exists():
                    vector_store_path = path
                    logger.info(f"벡터 스토어 경로 발견: {path}")
                    break
            
            if not vector_store_path:
                logger.warning("벡터 스토어가 존재하지 않습니다")
                logger.warning(f"시도한 경로들: {[str(p) for p in possible_paths]}")
                self.vectors = None
                self.metadata = None
                return
            
            # 벡터 데이터 로드
            vectors_file = vector_store_path / "vectors.npy"
            metadata_file = vector_store_path / "metadata.json"
            
            if vectors_file.exists() and metadata_file.exists():
                self.vectors = np.load(vectors_file)
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    import json
                    self.metadata = json.load(f)
                logger.info(f"벡터 스토어 로드 완료: {len(self.vectors)}개 벡터")
            else:
                logger.warning("벡터 스토어 파일이 불완전합니다")
                logger.warning(f"vectors_file: {vectors_file} (존재: {vectors_file.exists()})")
                logger.warning(f"metadata_file: {metadata_file} (존재: {metadata_file.exists()})")
                self.vectors = None
                self.metadata = None
                
        except Exception as e:
            logger.error(f"벡터 스토어 로드 실패: {e}")
            self.vectors = None
            self.metadata = None
    
    def chat(self, query: str, top_k: int = 5) -> ChatResponse:
        """
        사용자 질문에 대한 답변 생성
        
        Args:
            query: 사용자 질문
            top_k: 검색할 결과 수
            
        Returns:
            ChatResponse: 답변과 근거 정보
        """
        logger.info(f"채팅 요청: {query}")
        
        try:
            # 1. 하이브리드 검색 수행
            search_results = self._hybrid_search(query, top_k)
            
            # 2. RAG 기반 답변 생성
            answer = self._generate_rag_answer(query, search_results)
            
            # 3. 응답 구성
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
    
    def _hybrid_search(self, query: str, top_k: int) -> List[SearchResult]:
        """하이브리드 검색 (FTS + FAISS)"""
        try:
            # 1. FTS 검색 (SQLite)
            fts_results = self._fts_search(query, top_k)
            
            # 2. FAISS 검색 (벡터 유사도)
            faiss_results = self._faiss_search(query, top_k)
            
            # 3. 결과 병합 및 중복 제거
            merged_results = self._merge_search_results(fts_results, faiss_results, top_k)
            
            logger.info(f"하이브리드 검색 완료: {len(merged_results)}개 결과")
            return merged_results
            
        except Exception as e:
            logger.error(f"하이브리드 검색 실패: {e}")
            return []
    
    def _fts_search(self, query: str, top_k: int) -> List[SearchResult]:
        """SQLite FTS 검색"""
        try:
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
            # numpy 배열 비교 문제 방지를 위해 명시적 체크
            if self.vectors is None or self.metadata is None:
                logger.warning("벡터 스토어가 로드되지 않았습니다")
                return []
            
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
                        source="faiss",
                        similarity_score=float(similarity)
                    )
                    faiss_results.append(search_result)
            
            logger.info(f"FAISS 검색 완료: {len(faiss_results)}개 결과")
            return faiss_results
            
        except Exception as e:
            logger.error(f"FAISS 검색 실패: {e}")
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
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
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
            return {
                "vector_store_loaded": self.vectors is not None and self.metadata is not None,
                "total_vectors": len(self.vectors) if self.vectors is not None else 0,
                "sqlite_available": hasattr(self, 'sqlite_store'),
                "openai_available": hasattr(self, 'openai_client')
            }
        except Exception as e:
            logger.error(f"검색 통계 조회 실패: {e}")
            return {"error": str(e)}
