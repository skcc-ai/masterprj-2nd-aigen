"""
Vector Store - 코드 심볼을 벡터로 변환하여 저장
LLM 분석 결과와 함께 검색 가능한 벡터 인덱스 생성
"""

import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple
import openai
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class VectorStore:
    """코드 심볼을 벡터로 변환하여 저장하는 벡터 스토어"""
    
    def __init__(self, api_key: Optional[str] = None, 
                 endpoint: Optional[str] = None,
                 embedding_deployment: str = "text-embedding-3-small",
                 storage_dir: str = "vector_store"):
        
        # Azure OpenAI 설정
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.embedding_deployment = embedding_deployment or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
        
        if not self.api_key:
            raise RuntimeError("Azure OpenAI API key required")
        
        # Azure OpenAI 클라이언트 설정
        self.client = openai.AzureOpenAI(
            api_key=self.api_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=self.endpoint
        )
        
        # 저장 디렉토리 설정
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # 벡터 저장소 초기화
        self.vectors: List[Tuple[List[float], Dict[str, Any]]] = []
        self.metadata_file = self.storage_dir / "metadata.json"
        self.vectors_file = self.storage_dir / "vectors.npy"
        
        # 기존 데이터 로드
        self._load_existing_data()
        
        logger.info(f"VectorStore initialized with Azure OpenAI deployment: {self.embedding_deployment}")

    def add_symbol(self, symbol_data: Dict[str, Any], file_path: str) -> bool:
        """심볼을 벡터로 변환하여 저장"""
        try:
            # 심볼을 텍스트로 변환
            text_content = self._symbol_to_text(symbol_data, file_path)
            
            # 임베딩 생성
            embedding = self._create_embedding(text_content)
            if not embedding:
                return False
            
            # 메타데이터 구성
            metadata = {
                "file_path": file_path,
                "symbol_type": symbol_data.get("type", "unknown"),
                "symbol_name": symbol_data.get("name", "unknown"),
                "location": symbol_data.get("location", {}),
                "llm_analysis": symbol_data.get("llm_analysis", {}),
                "text_content": text_content[:500],  # 검색용 텍스트 (앞부분만)
                "timestamp": self._get_timestamp()
            }
            
            # 벡터와 메타데이터 저장
            self.vectors.append((embedding, metadata))
            
            logger.info(f"✅ 심볼 벡터 저장 완료: {symbol_data.get('name', 'unknown')} in {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"⚠️ 심볼 벡터 저장 실패: {e}")
            return False

    def add_file_symbols(self, ast_data: Dict[str, Any]) -> int:
        """파일의 모든 심볼을 벡터로 변환하여 저장"""
        file_path = ast_data.get("file", {}).get("path", "unknown")
        symbols = ast_data.get("symbols", [])
        
        added_count = 0
        for symbol in symbols:
            if self.add_symbol(symbol, file_path):
                added_count += 1
        
        logger.info(f"✅ {file_path}: {added_count}/{len(symbols)}개 심볼 벡터 저장 완료")
        return added_count

    def add_chunk(self, chunk_data: Dict[str, Any]) -> bool:
        """청크를 벡터로 변환하여 저장"""
        try:
            # 청크 데이터를 심볼 형태로 변환
            symbol_data = {
                "name": chunk_data.get("symbol_name", "unknown"),
                "type": chunk_data.get("symbol_type", "chunk"),
                "content": chunk_data.get("content", ""),
                "location": chunk_data.get("location", {}),
                "metadata": chunk_data.get("metadata", {})
            }
            
            file_path = chunk_data.get("file_path", "unknown")
            
            # add_symbol 메서드 사용
            return self.add_symbol(symbol_data, file_path)
            
        except Exception as e:
            logger.error(f"⚠️ 청크 벡터 저장 실패: {e}")
            return False

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """쿼리와 유사한 심볼 검색"""
        try:
            # 쿼리 임베딩 생성
            query_embedding = self._create_embedding(query)
            if not query_embedding:
                return []
            
            # 코사인 유사도 계산
            similarities = []
            for vector, metadata in self.vectors:
                similarity = self._cosine_similarity(query_embedding, vector)
                similarities.append((similarity, metadata))
            
            # 유사도 순으로 정렬
            similarities.sort(key=lambda x: x[0], reverse=True)
            
            # 상위 k개 결과 반환
            results = []
            for similarity, metadata in similarities[:top_k]:
                result = metadata.copy()
                result["similarity_score"] = float(similarity)
                results.append(result)
            
            logger.info(f"🔍 검색 완료: '{query}' -> {len(results)}개 결과")
            return results
            
        except Exception as e:
            logger.error(f"⚠️ 검색 실패: {e}")
            return []

    def save(self) -> bool:
        """벡터와 메타데이터를 파일에 저장"""
        try:
            # 메타데이터 저장
            metadata_list = [metadata for _, metadata in self.vectors]
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata_list, f, ensure_ascii=False, indent=2)
            
            # 벡터 저장
            vectors_array = np.array([vector for vector, _ in self.vectors])
            np.save(self.vectors_file, vectors_array)
            
            logger.info(f"✅ 벡터 스토어 저장 완료: {len(self.vectors)}개 벡터")
            return True
            
        except Exception as e:
            logger.error(f"⚠️ 벡터 스토어 저장 실패: {e}")
            return False

    def load(self) -> bool:
        """저장된 벡터와 메타데이터 로드"""
        try:
            if not self.metadata_file.exists() or not self.vectors_file.exists():
                logger.info("저장된 벡터 데이터가 없습니다.")
                return True
            
            # 메타데이터 로드
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                metadata_list = json.load(f)
            
            # 벡터 로드
            vectors_array = np.load(self.vectors_file)
            
            # 벡터와 메타데이터 결합
            self.vectors = list(zip(vectors_array, metadata_list))
            
            logger.info(f"✅ 벡터 스토어 로드 완료: {len(self.vectors)}개 벡터")
            return True
            
        except Exception as e:
            logger.error(f"⚠️ 벡터 스토어 로드 실패: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """벡터 스토어 통계 정보"""
        if not self.vectors:
            return {"total_vectors": 0}
        
        # 파일별 심볼 수
        file_counts = {}
        symbol_type_counts = {}
        
        for _, metadata in self.vectors:
            file_path = metadata.get("file_path", "unknown")
            symbol_type = metadata.get("symbol_type", "unknown")
            
            file_counts[file_path] = file_counts.get(file_path, 0) + 1
            symbol_type_counts[symbol_type] = symbol_type_counts.get(symbol_type, 0) + 1
        
        return {
            "total_vectors": len(self.vectors),
            "unique_files": len(file_counts),
            "file_counts": file_counts,
            "symbol_type_counts": symbol_type_counts
        }

    def _symbol_to_text(self, symbol_data: Dict[str, Any], file_path: str) -> str:
        """심볼 데이터를 검색 가능한 텍스트로 변환"""
        text_parts = []
        
        # 기본 정보
        text_parts.append(f"Symbol: {symbol_data.get('name', 'unknown')}")
        text_parts.append(f"Type: {symbol_data.get('type', 'unknown')}")
        text_parts.append(f"File: {file_path}")
        
        # 시그니처
        if symbol_data.get('signature'):
            text_parts.append(f"Signature: {symbol_data['signature']}")
        
        # 메타데이터
        metadata = symbol_data.get('metadata', {})
        if metadata.get('parameters'):
            text_parts.append(f"Parameters: {', '.join(metadata['parameters'])}")
        if metadata.get('return_type'):
            text_parts.append(f"Return type: {metadata['return_type']}")
        if metadata.get('bases'):
            text_parts.append(f"Bases: {', '.join(metadata['bases'])}")
        
        # LLM 분석 결과
        llm_analysis = symbol_data.get('llm_analysis', {})
        if llm_analysis.get('purpose'):
            text_parts.append(f"Purpose: {llm_analysis['purpose']}")
        if llm_analysis.get('summary'):
            text_parts.append(f"Summary: {llm_analysis['summary']}")
        if llm_analysis.get('dependencies'):
            text_parts.append(f"Dependencies: {', '.join(llm_analysis['dependencies'])}")
        if llm_analysis.get('key_operations'):
            text_parts.append(f"Key operations: {', '.join(llm_analysis['key_operations'])}")
        
        # 본문 내용 (간단한 요약)
        body = symbol_data.get('body', {})
        if body.get('nodes'):
            node_types = [node.get('type', 'unknown') for node in body['nodes'][:5]]  # 처음 5개만
            text_parts.append(f"Body nodes: {', '.join(node_types)}")
        
        return " | ".join(text_parts)

    def _create_embedding(self, text: str) -> Optional[List[float]]:
        """텍스트를 임베딩 벡터로 변환"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_deployment
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"임베딩 생성 실패: {e}")
            return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """두 벡터 간의 코사인 유사도 계산"""
        try:
            vec1_array = np.array(vec1)
            vec2_array = np.array(vec2)
            
            dot_product = np.dot(vec1_array, vec2_array)
            norm1 = np.linalg.norm(vec1_array)
            norm2 = np.linalg.norm(vec2_array)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        except Exception:
            return 0.0

    def _load_existing_data(self):
        """기존 저장 데이터 로드"""
        try:
            self.load()
        except Exception as e:
            logger.warning(f"기존 데이터 로드 실패: {e}")

    def _get_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().isoformat()
