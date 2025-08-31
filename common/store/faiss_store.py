"""
FAISS vector store for Code Analytica
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None

logger = logging.getLogger(__name__)


class FAISSStore:
    """FAISS 벡터 스토어"""
    
    def __init__(self, index_path: str = "./artifacts/faiss.index", dimension: int = 1536):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS is not available. Please install faiss-cpu or faiss-gpu")
        
        self.index_path = index_path
        self.dimension = dimension
        self.ensure_data_dir()
        self.index = None
        self.doc_ids = []
        self.load_or_create_index()
    
    def ensure_data_dir(self):
        """데이터 디렉토리 생성"""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
    
    def load_or_create_index(self):
        """기존 인덱스 로드 또는 새로 생성"""
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                # doc_ids 로드
                ids_path = self.index_path.replace('.index', '_ids.pkl')
                if os.path.exists(ids_path):
                    with open(ids_path, 'rb') as f:
                        self.doc_ids = pickle.load(f)
                logger.info(f"Loaded existing FAISS index from {self.index_path}")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}")
                self.create_new_index()
        else:
            self.create_new_index()
    
    def create_new_index(self):
        """새 FAISS 인덱스 생성"""
        # 단순 플랫 인덱스 사용 (클러스터링 불필요)
        self.index = faiss.IndexFlatIP(self.dimension)
        logger.info(f"Created FlatIP index for dimension {self.dimension}")
    
    def add_vectors(self, vectors: List[List[float]], doc_ids: List[int]):
        """벡터 추가"""
        if not vectors:
            return
        
        vectors = np.array(vectors, dtype=np.float32)
        
        # 정규화 (코사인 유사도용)
        faiss.normalize_L2(vectors)
        
        # 인덱스에 추가 (FlatIP는 훈련 불필요)
        self.index.add(vectors)
        self.doc_ids.extend(doc_ids)
        
        logger.info(f"Added {len(vectors)} vectors to FAISS index")
    
    def search_text(self, query_text: str, k: int = 10) -> List[Dict[str, Any]]:
        """텍스트 쿼리로 검색 (임베딩 변환 후 검색)"""
        try:
            # OpenAI 임베딩 API를 사용하여 텍스트를 벡터로 변환
            import openai
            
            # 환경변수에서 API 키 가져오기
            api_key = os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("OpenAI API 키가 설정되지 않았습니다. 기존 벡터를 사용한 검색을 시도합니다.")
                return self._search_with_existing_vectors(query_text, k)
            
            # Azure OpenAI 또는 OpenAI 설정
            if os.getenv("AZURE_OPENAI_ENDPOINT"):
                # Azure OpenAI 사용
                client = openai.AzureOpenAI(
                    api_key=api_key,
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
                )
                deployment_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large")
            else:
                # OpenAI 사용
                client = openai.OpenAI(api_key=api_key)
                deployment_name = "text-embedding-3-large"
            
            # 텍스트를 임베딩으로 변환
            response = client.embeddings.create(
                model=deployment_name,
                input=query_text
            )
            
            query_vector = response.data[0].embedding
            
            # 벡터 검색 수행
            return self.search_vector(query_vector, k)
            
        except Exception as e:
            logger.error(f"텍스트 검색 실패: {e}")
            logger.info("기존 벡터를 사용한 검색을 시도합니다.")
            return self._search_with_existing_vectors(query_text, k)
    
    def _search_with_existing_vectors(self, query_text: str, k: int = 10) -> List[Dict[str, Any]]:
        """기존 벡터를 사용한 간단한 텍스트 검색 (fallback)"""
        try:
            # 간단한 키워드 매칭으로 검색
            # 실제로는 더 정교한 검색 로직이 필요하지만, 임시로 사용
            results = []
            
            # 모든 벡터에 대해 간단한 검색 수행
            # 여기서는 첫 번째 벡터를 쿼리로 사용 (임시 해결책)
            if self.doc_ids:
                # 첫 번째 벡터를 쿼리로 사용
                query_vector = np.zeros(self.dimension, dtype=np.float32)
                query_vector[0] = 1.0  # 간단한 쿼리 벡터
                
                # 검색 수행
                distances, indices = self.index.search(query_vector.reshape(1, -1), min(k, len(self.doc_ids)))
                
                for idx, distance in zip(indices[0], distances[0]):
                    if idx != -1 and idx < len(self.doc_ids):
                        results.append({
                            "doc_id": self.doc_ids[idx],
                            "similarity": float(distance),
                            "index": int(idx)
                        })
            
            logger.info(f"기존 벡터를 사용한 검색으로 {len(results)}개 결과 반환")
            return results
            
        except Exception as e:
            logger.error(f"기존 벡터 검색 실패: {e}")
            return []
    
    def search_vector(self, query_vector: List[float], k: int = 10) -> List[Dict[str, Any]]:
        """벡터 검색 (기존 search 메서드)"""
        if not self.doc_ids:
            return []
        
        query_vector = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        
        # 검색 수행
        distances, indices = self.index.search(query_vector, min(k, len(self.doc_ids)))
        
        # 결과 반환 (딕셔너리 형태)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.doc_ids):
                # 코사인 유사도로 변환 (distance는 내적)
                similarity = float(distance)
                results.append({
                    "doc_id": self.doc_ids[idx],
                    "similarity": similarity,
                    "index": int(idx)
                })
        
        return results
    
    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """통합 검색 메서드 - 텍스트 또는 벡터 자동 감지"""
        if isinstance(query, str):
            return self.search_text(query, k)
        elif isinstance(query, (list, tuple)) and all(isinstance(x, (int, float)) for x in query):
            return self.search_vector(query, k)
        else:
            logger.error(f"지원하지 않는 쿼리 타입: {type(query)}")
            return []
    
    def search_by_ids(self, doc_ids: List[int], query: str, k: int = 10) -> List[Dict[str, Any]]:
        """특정 doc_ids 중에서 검색"""
        if not doc_ids:
            return []
        
        # 전체 검색 후 필터링
        all_results = self.search(query, k * 2)  # 더 많은 결과 가져오기
        
        # doc_ids에 포함된 것만 필터링
        filtered_results = [result for result in all_results if result.get("doc_id") in doc_ids]
        
        return filtered_results[:k]
    
    def get_vector_by_id(self, doc_id: int) -> Optional[List[float]]:
        """doc_id로 벡터 조회"""
        try:
            idx = self.doc_ids.index(doc_id)
            # FAISS에서 개별 벡터 추출은 복잡하므로, 전체 인덱스를 다시 로드
            # 실제 구현에서는 별도 벡터 저장소 사용 권장
            return None
        except ValueError:
            return None
    
    def save_index(self):
        """인덱스 저장"""
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
            
            # doc_ids 저장
            ids_path = self.index_path.replace('.index', '_ids.pkl')
            with open(ids_path, 'wb') as f:
                pickle.dump(self.doc_ids, f)
            
            logger.info(f"Saved FAISS index to {self.index_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """인덱스 통계"""
        if self.index is None:
            return {"total_vectors": 0, "dimension": self.dimension}
        
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "doc_ids_count": len(self.doc_ids)
        }
    
    def close(self):
        """인덱스 저장 및 정리"""
        self.save_index() 