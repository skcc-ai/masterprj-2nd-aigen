"""
StructSynth Agent - Main Execution Script
코드 파싱 및 분석 준비 에이전트 실행 스크립트
"""

import argparse
import logging
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from .agent import StructSynthAgent

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('structsynth.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="StructSynth Agent - Code Parsing and Analysis")
    
    parser.add_argument(
        "repo_path",
        help="분석할 저장소 경로"
    )
    
    parser.add_argument(
        "--artifacts-dir",
        default="./artifacts",
        help="아티팩트 출력 디렉토리 (기본값: ./artifacts)"
    )
    
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="데이터 저장 디렉토리 (기본값: ./data)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="상세 로그 출력"
    )
    
    args = parser.parse_args()
    
    # 로깅 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 에이전트 초기화
        logger.info("Initializing StructSynth Agent...")
        agent = StructSynthAgent(
            repo_path=args.repo_path,
            artifacts_dir=args.artifacts_dir,
            data_dir=args.data_dir
        )
        
        # 상태 출력
        status = agent.get_status()
        logger.info(f"Agent initialized successfully")
        logger.info(f"Repository: {status['repo_path']}")
        logger.info(f"Artifacts directory: {status['artifacts_dir']}")
        logger.info(f"Data directory: {status['data_dir']}")
        
        # 저장소 분석 시작
        logger.info("Starting repository analysis...")
        results = agent.analyze_repository()
        
        # 결과 출력
        logger.info("Analysis completed!")
        logger.info(f"Total files: {results['total_files']}")
        logger.info(f"Parsed files: {results['parsed_files']}")
        logger.info(f"Failed files: {results['failed_files']}")
        logger.info(f"Total symbols: {results['total_symbols']}")
        logger.info(f"Total chunks: {results['total_chunks']}")
        logger.info(f"Total calls: {results['total_calls']}")
        logger.info(f"Processing time: {results['processing_time']:.2f}s")
        
        # 출력 경로 출력
        logger.info("Output files generated:")
        for key, path in results['output_paths'].items():
            logger.info(f"  {key}: {path}")
        
        if results['errors']:
            logger.warning(f"Encountered {len(results['errors'])} errors:")
            for error in results['errors']:
                logger.warning(f"  - {error}")
        
        # 에이전트 정리
        agent.close()
        
        logger.info("StructSynth Agent completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 