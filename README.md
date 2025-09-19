# Code Analytica

> AI 기반 코드 분석 및 대화형 코드 이해 시스템

## 프로젝트 개요

Code Analytica는 AI 기반의 코드 분석 및 대화형 코드 이해 시스템입니다. 대규모 레거시 코드베이스 분석의 복잡성을 해결하고, 개발자가 코드를 보다 쉽고 빠르게 이해할 수 있도록 돕는 4단계 AI 에이전트 파이프라인을 제공합니다.

### 주요 기능

- **StructSynth Agent**: 다중 언어 코드 구조 분석 (Python, Java, C/C++)
- **InsightGen Agent**: AI 기반 맞춤형 문서 자동 생성
- **EvalGuard Agent**: 코드 품질 평가 및 보안 분석 (개발 예정)
- **CodeChat Agent**: 하이브리드 검색 + RAG 기반 대화형 코드 탐색

### 🏗️ 시스템 아키텍처

```
Presentation Layer
├── Streamlit Web UI (포트: 8501)
├── Code Chat Interface
└── Documentation Viewer

API Gateway Layer
├── FastAPI Backend (포트: 8000)
├── API Documentation (/docs)
└── RESTful API Endpoints

AI Agent Layer
├── Agent1: StructSynth (코드 구조 분석)
├── Agent2: InsightGen (문서 생성)
├── Agent3: EvalGuard (품질 평가)
└── Agent4: CodeChat (대화형 탐색)

Data Layer
├── SQLite (메타데이터 저장)
├── FAISS (벡터 검색)
└── Artifacts (분석 결과)
```

## 빠른 시작

### 전제 조건

- **Docker & Docker Compose** (권장)
- **Python 3.11+** (로컬 실행 시)
- **Azure OpenAI API 키** (필수)

### 1. Docker Compose로 실행 (권장)

```bash
# 저장소 클론
git clone https://github.com/your-username/code-analytica.git
cd code-analytica

# 환경 변수 설정
cp .env.example .env
# .env 파일에서 Azure OpenAI 설정 입력

# 전체 시스템 실행
./run_all.sh
```

실행 후 접속:
- **웹 UI**: http://localhost:8501
- **API 문서**: http://localhost:8000/docs
- **백엔드 API**: http://localhost:8000

### 2. 로컬 개발 환경

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 백엔드 실행
python api/main.py

# UI 실행 (새 터미널)
streamlit run ui/streamlit_app/app.py
```

### 3. 환경 변수 설정

`.env` 파일 생성:

```env
# Azure OpenAI 설정
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

## 사용법

### 1. 코드 분석 시작

1. **웹 UI 접속**: http://localhost:8501
2. **Code Analysis 탭**에서 분석할 디렉토리 경로 입력
3. **"분석 시작"** 클릭하여 StructSynth 에이전트 실행
4. 분석 완료까지 대기 (대규모 프로젝트: ~25분)

### 2. 대화형 코드 탐색

1. **Code Chat 탭**으로 이동
2. 코드에 대한 질문 입력:
   - "이 프로젝트의 주요 기능은 무엇인가요?"
   - "데이터베이스 연결은 어떻게 처리되나요?"
   - "성능 병목점은 어디에 있나요?"
3. AI가 코드 분석 결과를 바탕으로 상세한 답변 제공

### 3. 문서 및 인사이트 확인

1. **Docs 탭**에서 자동 생성된 문서 확인
2. 프로젝트 구조, 심볼 분석, 호출 관계 등 체계적인 분석 결과 제공

## API 문서

### 주요 엔드포인트

#### 에이전트 관리
```http
GET /api/agents
POST /api/agents/{agent_id}/run
POST /api/agents/run-all
```

#### 검색 및 채팅
```http
POST /api/search
POST /api/chat
```

#### 시스템 정보
```http
GET /health
GET /api/modules
```

### API 사용 예제

```python
import requests

# 전체 에이전트 실행
response = requests.post("http://localhost:8000/api/agents/run-all", json={
    "repo_path": "/path/to/your/project",
    "artifacts_dir": "./artifacts",
    "data_dir": "./data"
})

# 코드 채팅
response = requests.post("http://localhost:8000/api/chat", json={
    "query": "이 프로젝트의 주요 구성 요소는 무엇인가요?",
    "top_k": 5
})
```

자세한 API 문서는 http://localhost:8000/docs 에서 확인하세요.

## 프로젝트 구조

```
Code Analytica/
├── agents/                 # AI 에이전트 모듈
│   ├── structsynth/       # 코드 구조 분석 에이전트
│   ├── codechat/          # 코드 채팅 에이전트
│   ├── insightgen/        # 인사이트 생성 에이전트
│   └── evalguard/         # 코드 품질 평가 에이전트
├── api/                   # FastAPI 백엔드
├── ui/                    # Streamlit 프론트엔드
│   └── streamlit_app/     # 웹 UI 애플리케이션
├── common/                # 공통 모듈 (저장소, 스키마)
│   └── store/             # SQLite, FAISS 저장소
├── tools/                 # 도구 함수들
├── configs/               # 설정 파일
├── data/                  # 데이터베이스 및 벡터 인덱스
├── artifacts/             # 분석 결과 산출물
├── docker-compose.yml     # Docker Compose 설정
├── Dockerfile            # Docker 이미지 설정
├── requirements.txt      # Python 의존성
└── run_all.sh           # 전체 시스템 실행 스크립트
```

## 기술 스택

### Backend
- **FastAPI**: 고성능 비동기 웹 프레임워크
- **Azure OpenAI**: GPT-4 및 임베딩 모델
- **SQLite**: 메타데이터 저장
- **FAISS**: 고성능 벡터 유사도 검색
- **Tree-sitter**: 다중 언어 코드 파싱

### Frontend
- **Streamlit**: 대화형 웹 UI
- **Python**: 백엔드 통합

### Infrastructure
- **Docker & Docker Compose**: 컨테이너화 배포
- **Uvicorn**: ASGI 서버

### AI/ML
- **LangChain**: LLM 오케스트레이션
- **OpenAI Embeddings**: 텍스트 임베딩
- **RAG (Retrieval-Augmented Generation)**: 검색 기반 생성

## 개발 현황

- **StructSynth Agent**: 완료 (다중 언어 파싱, 벡터 저장)
- **CodeChat Agent**: 완료 (하이브리드 검색, RAG)
- **InsightGen Agent**: 완료 (AI 문서 생성)
- **EvalGuard Agent**: 개발 중 (코드 품질 평가)
- **Web UI**: 완료 (분석, 채팅, 문서 뷰어)
- **API Gateway**: 완료 (RESTful API)
- **Docker 배포**: 완료 (원클릭 실행)

## 기여 방법

### 개발 환경 설정

1. **저장소 포크 및 클론**
```bash
git clone https://github.com/your-username/code-analytica.git
cd code-analytica
```

2. **개발 환경 설정**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. **개발용 추가 패키지 설치**
```bash
pip install pytest black flake8 mypy
```

### 코드 스타일

- **Black**: 코드 포맷팅
- **Flake8**: 린팅
- **MyPy**: 타입 체크

```bash
# 코드 포맷팅
black .

# 린팅
flake8 .

# 타입 체크
mypy .
```

### 테스트

```bash
# 단위 테스트 실행
pytest

# 커버리지 포함
pytest --cov=.
```

### 기여 가이드라인

1. **이슈 생성**: 새로운 기능이나 버그 리포트
2. **브랜치 생성**: `feature/feature-name` 또는 `fix/bug-name`
3. **커밋 메시지**: 명확하고 설명적인 메시지
4. **Pull Request**: 상세한 설명과 테스트 결과 포함

### 문제 해결

**일반적인 문제들:**

1. **Azure OpenAI 연결 오류**
   - API 키와 엔드포인트 확인
   - 배포 모델명 확인 (GPT-4, text-embedding-ada-002)

2. **Docker 실행 오류**
   - Docker가 실행 중인지 확인
   - 포트 충돌 확인 (8000, 8501)

3. **분석 실패**
   - 디렉토리 경로 확인
   - 지원 언어 확인 (Python, Java, C/C++)




