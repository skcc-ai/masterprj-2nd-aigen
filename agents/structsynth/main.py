#!/usr/bin/env python3
"""
Entry point for StructSynth Agent
코드 저장소 분석 및 AST 추출 파이프라인 실행
"""

import argparse
import logging
import os
import sys
from dotenv import load_dotenv

# 상대 import 문제 해결을 위해 경로 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.structsynth.agent import StructSynthAgent

# .env 로딩
load_dotenv()

# 환경변수 로딩 확인
def check_env_variables():
    """환경변수 설정 확인"""
    required_vars = [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_VERSION",
        "AZURE_OPENAI_DEPLOYMENT_NAME"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logging.warning(f"⚠️  누락된 환경변수: {missing_vars}")
        return False
    else:
        logging.info("✅ 모든 필수 환경변수가 설정되었습니다")
        return True

def main():
    parser = argparse.ArgumentParser(description="StructSynth Agent - 코드 구조 분석")
    parser.add_argument("repo_path", help="Path to source code repository")
    parser.add_argument("--artifacts-dir", default="./artifacts", help="Output artifacts directory")
    parser.add_argument("--data-dir", default="./data", help="Data storage directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # 환경변수 확인
    if not check_env_variables():
        logging.warning("⚠️  일부 환경변수가 누락되었습니다. LLM 분석이 제한될 수 있습니다.")

    repo_path = os.path.abspath(args.repo_path)
    
    # repo_path가 파일인 경우 해당 파일이 속한 디렉토리를 전달
    if os.path.isfile(repo_path):
        repo_dir = os.path.dirname(repo_path)
        logging.info(f"📁 파일 경로 감지됨: {repo_path}")
        logging.info(f"📁 분석 대상 디렉토리: {repo_dir}")
        repo_path = repo_dir
    else:
        logging.info(f"📁 디렉토리 경로 감지됨: {repo_path}")
    
    # 절대 경로로 artifacts와 data 디렉토리 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    artifacts_dir = os.path.abspath(args.artifacts_dir)
    data_dir = os.path.abspath(args.data_dir)
    
    logging.info(f"🚀 StructSynth Agent 시작: {repo_path}")
    logging.info(f"📁 결과물 저장 위치: {artifacts_dir}")
    logging.info(f"📁 데이터 저장 위치: {data_dir}")

    try:
        # Agent 생성 및 실행 (절대 경로 전달)
        agent = StructSynthAgent(
            repo_path=repo_path,
            artifacts_dir=artifacts_dir,
            data_dir=data_dir
        )
        
        # 저장소 분석 실행
        agent.analyze_repository()
        
    except Exception as e:
        logging.error(f"❌ StructSynth Agent 실행 실패: {e}")
        raise

if __name__ == "__main__":
    main()
