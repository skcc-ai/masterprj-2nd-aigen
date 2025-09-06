"""
InsightGen 도구들 - 산출물 불러오기 및 관리
"""

import os
import json
from typing import Any, Dict, List, Optional
from pathlib import Path

from ._compat import tool


@tool("get_artifact", return_direct=False)
def get_artifact(artifact_name: str, data_dir: str = "./data") -> Dict[str, Any]:
    """InsightGen 산출물을 불러옵니다.
    
    Args:
        artifact_name: 산출물 파일명 (예: "01-프로젝트-개요.md")
        data_dir: 데이터 디렉토리 경로
    
    Returns:
        산출물 내용과 메타데이터
    """
    try:
        # 산출물 파일 경로 구성
        artifact_path = os.path.join(data_dir, "insightgen", artifact_name)
        
        if not os.path.exists(artifact_path):
            return {
                "error": "artifact_not_found",
                "artifact_name": artifact_name,
                "path": artifact_path,
                "available_artifacts": _list_available_artifacts(data_dir)
            }
        
        # 파일 내용 읽기
        with open(artifact_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 파일 메타데이터
        file_stat = os.stat(artifact_path)
        
        return {
            "content": content,
            "path": artifact_path,
            "size": file_stat.st_size,
            "modified_time": file_stat.st_mtime,
            "artifact_name": artifact_name
        }
        
    except Exception as e:
        return {
            "error": "read_failed",
            "artifact_name": artifact_name,
            "detail": str(e)
        }


@tool("list_artifacts", return_direct=False)
def list_artifacts(data_dir: str = "./data") -> Dict[str, Any]:
    """사용 가능한 InsightGen 산출물 목록을 반환합니다.
    
    Args:
        data_dir: 데이터 디렉토리 경로
    
    Returns:
        산출물 목록과 메타데이터
    """
    try:
        artifacts_dir = os.path.join(data_dir, "insightgen")
        
        if not os.path.exists(artifacts_dir):
            return {
                "error": "artifacts_dir_not_found",
                "path": artifacts_dir,
                "artifacts": []
            }
        
        artifacts = []
        for filename in os.listdir(artifacts_dir):
            if filename.endswith(('.md', '.json', '.txt')):
                file_path = os.path.join(artifacts_dir, filename)
                file_stat = os.stat(file_path)
                
                artifacts.append({
                    "name": filename,
                    "path": file_path,
                    "size": file_stat.st_size,
                    "modified_time": file_stat.st_mtime,
                    "type": "markdown" if filename.endswith('.md') else "json" if filename.endswith('.json') else "text"
                })
        
        # 파일명 순으로 정렬
        artifacts.sort(key=lambda x: x["name"])
        
        return {
            "artifacts": artifacts,
            "total_count": len(artifacts),
            "artifacts_dir": artifacts_dir
        }
        
    except Exception as e:
        return {
            "error": "list_failed",
            "detail": str(e),
            "artifacts": []
        }


@tool("get_artifact_summary", return_direct=False)
def get_artifact_summary(data_dir: str = "./data") -> Dict[str, Any]:
    """InsightGen 산출물들의 요약 정보를 반환합니다.
    
    Args:
        data_dir: 데이터 디렉토리 경로
    
    Returns:
        산출물 요약 정보
    """
    try:
        artifacts_info = list_artifacts(data_dir)
        
        if "error" in artifacts_info:
            return artifacts_info
        
        # 각 산출물의 첫 200자 미리보기 생성
        summaries = []
        for artifact in artifacts_info["artifacts"]:
            try:
                with open(artifact["path"], 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 첫 200자 추출
                preview = content[:200].replace('\n', ' ').strip()
                if len(content) > 200:
                    preview += "..."
                
                summaries.append({
                    "name": artifact["name"],
                    "preview": preview,
                    "size": artifact["size"],
                    "type": artifact["type"]
                })
            except Exception as e:
                summaries.append({
                    "name": artifact["name"],
                    "preview": f"읽기 실패: {str(e)}",
                    "size": artifact["size"],
                    "type": artifact["type"]
                })
        
        return {
            "summaries": summaries,
            "total_count": len(summaries),
            "artifacts_dir": artifacts_info["artifacts_dir"]
        }
        
    except Exception as e:
        return {
            "error": "summary_failed",
            "detail": str(e)
        }


def _list_available_artifacts(data_dir: str) -> List[str]:
    """사용 가능한 산출물 파일명 목록을 반환합니다."""
    try:
        artifacts_dir = os.path.join(data_dir, "insightgen")
        if not os.path.exists(artifacts_dir):
            return []
        
        return [f for f in os.listdir(artifacts_dir) 
                if f.endswith(('.md', '.json', '.txt'))]
    except:
        return []
