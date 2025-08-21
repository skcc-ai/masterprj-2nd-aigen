"""
Code Indexer - Azure OpenAI 임베딩 및 인덱싱
코드 청크를 벡터로 변환하여 검색 가능하게 만듦
"""

import logging
import os
import openai
from typing import List, Optional

logger = logging.getLogger(__name__)


class CodeIndexer:
    """Azure OpenAI API 기반 코드 인덱서"""

    def __init__(self, api_key: Optional[str] = None, 
                 endpoint: Optional[str] = None,
                 deployment: str = "text-embedding-3-small"):
        
        # Azure OpenAI 설정
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.deployment = deployment or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
        
        if not self.api_key:
            raise RuntimeError("Azure OpenAI API key required")
        
        # Azure OpenAI 클라이언트 설정
        self.client = openai.AzureOpenAI(
            api_key=self.api_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            azure_endpoint=self.endpoint
        )
        
        logger.info(f"CodeIndexer initialized with Azure OpenAI deployment: {self.deployment}")

    def create_embedding(self, content: str) -> Optional[List[float]]:
        try:
            response = self.client.embeddings.create(
                input=content,
                model=self.deployment
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            # fallback: 더미 벡터 반환
            return [0.0] * 1536 