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
    
    def __init__(self, index_path: str = "./data/faiss.index", dimension: int = 1536):
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
        # IVF100, Flat 인덱스 (빠른 검색과 정확도 균형)
        quantizer = faiss.IndexFlatIP(self.dimension)
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        self.doc_ids = []
        logger.info(f"Created new FAISS index with dimension {self.dimension}")
    
    def add_vectors(self, vectors: List[List[float]], doc_ids: List[int]):
        """벡터 추가"""
        if not vectors:
            return
        
        vectors = np.array(vectors, dtype=np.float32)
        
        # 정규화 (코사인 유사도용)
        faiss.normalize_L2(vectors)
        
        # 인덱스에 추가
        if len(self.doc_ids) == 0:
            # 첫 번째 배치: 인덱스 훈련
            self.index.train(vectors)
        
        self.index.add(vectors)
        self.doc_ids.extend(doc_ids)
        
        logger.info(f"Added {len(vectors)} vectors to FAISS index")
    
    def search(self, query_vector: List[float], k: int = 10) -> List[Tuple[int, float]]:
        """벡터 검색"""
        if not self.doc_ids:
            return []
        
        query_vector = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        
        # 검색 수행
        distances, indices = self.index.search(query_vector, min(k, len(self.doc_ids)))
        
        # 결과 반환 (doc_id, similarity_score)
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.doc_ids):
                # 코사인 유사도로 변환 (distance는 내적)
                similarity = float(distance)
                results.append((self.doc_ids[idx], similarity))
        
        return results
    
    def search_by_ids(self, doc_ids: List[int], query_vector: List[float], k: int = 10) -> List[Tuple[int, float]]:
        """특정 doc_ids 중에서 검색"""
        if not doc_ids:
            return []
        
        # 전체 검색 후 필터링
        all_results = self.search(query_vector, k * 2)  # 더 많은 결과 가져오기
        
        # doc_ids에 포함된 것만 필터링
        filtered_results = [(doc_id, score) for doc_id, score in all_results if doc_id in doc_ids]
        
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