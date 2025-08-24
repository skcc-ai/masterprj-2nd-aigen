#!/bin/bash

# Code Analytica 전체 시스템 실행 스크립트 (Docker Compose)
# 백엔드와 UI를 모두 실행하여 완전한 시스템을 제공

set -e  # 오류 발생 시 스크립트 중단

echo "🚀 Code Analytica 전체 시스템을 Docker Compose로 실행합니다..."
echo "=================================================="

# 기존 컨테이너 정리
echo "🧹 기존 컨테이너 정리 중..."
docker-compose down --remove-orphans

# Docker 이미지 재빌드 (필요시)
echo "🔨 Docker 이미지 빌드 중..."
docker-compose build --no-cache

# 서비스 시작
echo "🚀 서비스 시작 중..."
docker-compose up -d

# 서비스 상태 확인
echo "📊 서비스 상태 확인 중..."
sleep 15
docker-compose ps

echo ""
echo "✅ 모든 서비스가 시작되었습니다!"
echo "=================================================="
echo ""
echo "🌐 백엔드 API: http://localhost:8000"
echo "📚 API 문서: http://localhost:8000/docs"
echo "🎨 Streamlit UI: http://localhost:8501"
echo ""
echo "📋 유용한 명령어:"
echo "   서비스 상태 확인: docker-compose ps"
echo "   로그 확인: docker-compose logs -f"
echo "   백엔드 로그: docker-compose logs -f backend"
echo "   UI 로그: docker-compose logs -f ui"
echo "   서비스 중지: docker-compose down"
echo "   서비스 재시작: docker-compose restart"
echo ""
echo "🔍 백엔드 헬스체크:"
curl -s http://localhost:8000/health || echo "백엔드 아직 시작 중..."

echo ""
echo "🎯 사용 방법:"
echo "1. 브라우저에서 http://localhost:8501 접속"
echo "2. Code Analysis 탭에서 디렉토리 경로 입력"
echo "3. StructSynth로 분석 시작 클릭"
echo "4. 분석 완료 후 Code Chat 탭에서 질문"
echo ""
echo "✨ 시스템이 성공적으로 실행되었습니다!"
