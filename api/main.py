"""
Code Analytica FastAPI Backend - 모든 Agent를 통합 관리하는 API 서버
상위 레벨에서 agents, common, ui 모듈들을 통합 관리
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 콘솔 출력
        logging.FileHandler('./artifacts/backend.log')  # 파일 출력
    ]
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Code Analytica API",
    description="코드 분석 및 인사이트 생성 플랫폼 API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 모델
class RepositoryRequest(BaseModel):
    repo_path: str = Field(..., description="분석할 저장소 경로")
    artifacts_dir: Optional[str] = Field("./artifacts", description="결과물 저장 디렉토리")
    data_dir: Optional[str] = Field("./data", description="데이터 저장 디렉토리")

class SearchRequest(BaseModel):
    query: str = Field(..., description="검색 쿼리")
    top_k: Optional[int] = Field(10, description="검색 결과 수")

class ChatRequest(BaseModel):
    query: str = Field(..., description="코드 관련 질문")
    top_k: Optional[int] = Field(5, description="검색 결과 수")
    session_id: Optional[str] = Field("default", description="대화 세션 ID")
    user_id: Optional[str] = Field(None, description="사용자 ID")

class AgentResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Agent 매니저
class AgentManager:
    """모든 Agent를 관리하는 매니저 클래스"""
    
    def __init__(self):
        self.agents = {}
        self.initialize_agents()
    
    def initialize_agents(self):
        """사용 가능한 Agent들을 초기화"""
        try:
            # Agent1: StructSynth (코드 구조 분석)
            from agents.structsynth.agent import StructSynthAgent
            self.agents['agent1'] = {
                'name': 'StructSynth',
                'description': '코드 구조 분석 및 AST 추출',
                'class': StructSynthAgent,
                'status': 'ready'
            }
            logger.info(" Agent1 (StructSynth) 초기화 완료")
        except Exception as e:
            logger.warning(f"⚠️  Agent1 초기화 실패: {e}")
            self.agents['agent1'] = {
                'name': 'StructSynth',
                'description': '코드 구조 분석 및 AST 추출',
                'status': 'error',
                'error': str(e)
            }
        
        # Agent2: InsightGen (코드 인사이트 생성)
        try:
            # 임시로 더미 Agent 생성 (실제 구현 전까지)
            self.agents['agent2'] = {
                'name': 'InsightGen',
                'description': '코드 인사이트 및 패턴 분석',
                'class': None,
                'status': 'not_implemented'
            }
            logger.info(" Agent2 (InsightGen) 초기화 완료 (구현 예정)")
        except Exception as e:
            logger.warning(f"⚠️  Agent2 초기화 실패: {e}")
        
        # Agent3: EvalGuard (코드 품질 평가)
        try:
            self.agents['agent3'] = {
                'name': 'EvalGuard',
                'description': '코드 품질 평가 및 보안 검사',
                'class': None,
                'status': 'not_implemented'
            }
            logger.info(" Agent3 (EvalGuard) 초기화 완료 (구현 예정)")
        except Exception as e:
            logger.warning(f"⚠️  Agent3 초기화 실패: {e}")
        
        # Agent4: CodeChat (코드 채팅)
        try:
            from agents.codechat.agent import CodeChatAgent
            self.agents['agent4'] = {
                'name': 'CodeChat',
                'description': '하이브리드 검색(FTS+FAISS) + RAG 기반 코드 채팅',
                'class': CodeChatAgent,
                'status': 'ready'
            }
            logger.info(" Agent4 (CodeChat) 초기화 완료")
        except Exception as e:
            logger.warning(f"⚠️  Agent4 초기화 실패: {e}")
            self.agents['agent4'] = {
                'name': 'CodeChat',
                'description': '하이브리드 검색(FTS+FAISS) + RAG 기반 코드 채팅',
                'class': None,
                'status': 'error',
                'error': str(e)
            }
    
    def get_agent_status(self) -> Dict[str, Any]:
        """모든 Agent의 상태 반환"""
        return {
            'total_agents': len(self.agents),
            'agents': self.agents
        }
    
    def run_agent(self, agent_id: str, repo_path: str, **kwargs) -> Dict[str, Any]:
        """특정 Agent 실행"""
        if agent_id not in self.agents:
            return {
                'success': False,
                'error': f'Agent {agent_id} not found'
            }
        
        agent_info = self.agents[agent_id]
        if agent_info['status'] != 'ready':
            return {
                'success': False,
                'error': f'Agent {agent_id} is not ready: {agent_info["status"]}'
            }
        
        try:
            # Agent 실행
            if agent_id == 'agent1':
                return self._run_agent1(repo_path, **kwargs)
            elif agent_id == 'agent4':
                return self._run_agent4(repo_path, **kwargs)
            else:
                return {
                    'success': False,
                    'error': f'Agent {agent_id} is not implemented yet'
                }
                
        except Exception as e:
            logger.error(f"Agent {agent_id} 실행 실패: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _run_agent1(self, repo_path: str, **kwargs) -> Dict[str, Any]:
        """Agent1 (StructSynth) 실행"""
        try:
            artifacts_dir = kwargs.get('artifacts_dir', './artifacts')
            data_dir = kwargs.get('data_dir', './data')
            
            # StructSynthAgent는 repo_path만 필수 인자로 받음
            agent = self.agents['agent1']['class'](repo_path)
            
            # 저장소 분석 실행
            agent.analyze_repository()
            
            # 결과 통계 수집
            stats = {}
            if hasattr(agent, 'get_vector_store_stats'):
                stats = agent.get_vector_store_stats()
            
            return {
                'success': True,
                'agent': 'agent1',
                'repo_path': repo_path,
                'message': '코드 구조 분석 완료',
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"Agent1 실행 실패: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def _run_agent4(self, repo_path: str, **kwargs) -> Dict[str, Any]:
        """Agent4 (CodeChat) 실행"""
        try:
            artifacts_dir = kwargs.get('artifacts_dir', './artifacts')
            data_dir = kwargs.get('data_dir', './data')
            
            # CodeChatAgent는 repo_path만 필수 인자로 받음
            agent = self.agents['agent4']['class'](repo_path) # Changed from agent2 to agent4
            
            # 채팅 실행
            response = agent.chat(kwargs.get('query', ''))
            
            return {
                'success': True,
                'agent': 'agent4', # Changed from agent2 to agent2
                'repo_path': repo_path,
                'message': '코드 채팅 완료',
                'response': response
            }
            
        except Exception as e:
            logger.error(f"Agent4 실행 실패: {e}")
            return {
                'success': False,
                'error': str(e)
            }

# Agent 매니저 초기화
agent_manager = AgentManager()

# API 엔드포인트들

@app.get("/", tags=["Root"])
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Code Analytica API",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "running",
        "modules": {
            "agents": "코드 분석 에이전트들",
            "common": "공통 모듈 (schemas, store, azure)",
            "ui": "사용자 인터페이스"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "message": "Code Analytica API is running",
        "modules": {
            "agents": "",
            "common": "", 
            "ui": ""
        }
    }

@app.get("/api/agents", tags=["Agents"], response_model=Dict[str, Any])
async def get_agents():
    """사용 가능한 Agent 목록 반환"""
    return agent_manager.get_agent_status()

@app.post("/api/agents/{agent_id}/run", tags=["Agents"], response_model=AgentResponse)
async def run_agent(agent_id: str, request: RepositoryRequest):
    """특정 Agent 실행"""
    try:
        result = agent_manager.run_agent(
            agent_id, 
            request.repo_path,
            artifacts_dir=request.artifacts_dir,
            data_dir=request.data_dir
        )
        
        if result['success']:
            return AgentResponse(
                success=True,
                message=result['message'],
                data=result
            )
        else:
            raise HTTPException(status_code=400, detail=result['error'])
            
    except Exception as e:
        logger.error(f"Agent 실행 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/run-all", tags=["Agents"], response_model=AgentResponse)
async def run_all_agents(request: RepositoryRequest):
    """모든 Agent 순차 실행"""
    try:
        results = {}
        
        # 각 Agent 순차 실행
        for agent_id in agent_manager.agents.keys():
            logger.info(f"🔄 {agent_id} 실행 중...")
            result = agent_manager.run_agent(
                agent_id, 
                request.repo_path,
                artifacts_dir=request.artifacts_dir,
                data_dir=request.data_dir
            )
            results[agent_id] = result
            
            if not result['success']:
                logger.warning(f"⚠️  {agent_id} 실행 실패: {result.get('error')}")
        
        return AgentResponse(
            success=True,
            message="모든 Agent 실행 완료",
            data={"results": results}
        )
        
    except Exception as e:
        logger.error(f"모든 Agent 실행 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search", tags=["Search"], response_model=AgentResponse)
async def search_symbols(request: SearchRequest):
    """심볼 검색 (Agent1 결과 활용)"""
    try:
        # Agent1의 벡터 스토어에서 검색
        try:
            from agents.structsynth.agent import StructSynthAgent
            # 임시로 현재 디렉토리 사용
            agent = StructSynthAgent(
                repo_path=".",
                artifacts_dir="./artifacts",
                data_dir="./data"
            )
            
            if agent.vector_store:
                results = agent.search_symbols(request.query, request.top_k)
                return AgentResponse(
                    success=True,
                    message=f"검색 완료: {len(results)}개 결과",
                    data={
                        "query": request.query,
                        "results": results
                    }
                )
            else:
                raise HTTPException(
                    status_code=500, 
                    detail="Vector store not available"
                )
                
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Search failed: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"검색 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat", tags=["Chat"], response_model=AgentResponse)
async def chat_with_code(request: ChatRequest):
    """코드 관련 질문에 대한 답변 생성 (Agent2 결과 활용)"""
    try:
        logger.info(f"=== 채팅 요청 수신 ===")
        logger.info(f"쿼리: '{request.query}' (타입: {type(request.query)}, 길이: {len(request.query)})")
        logger.info(f"top_k: {request.top_k}")
        
        # Agent2 (CodeChat) 실행
        try:
            from agents.codechat.agent import CodeChatAgent
            logger.info("CodeChatAgent 임포트 성공")
            
            # 임시로 현재 디렉토리 사용
            agent = CodeChatAgent(
                repo_path=".",
                artifacts_dir="./artifacts",
                data_dir="./data"
            )
            logger.info("CodeChatAgent 초기화 완료")
            
            # 채팅 요청 처리 (AI 컨텍스트 관리 포함)
            logger.info("CodeChatAgent.chat() 호출 시작 (AI 컨텍스트 모드)")
            response = agent.chat(
                query=request.query, 
                top_k=request.top_k,
                session_id=request.session_id,
                user_id=request.user_id
            )
            logger.info(f"CodeChatAgent.chat() 응답: {response.answer[:100]}...")
            logger.info(f"근거 수: {len(response.evidence)}개")
            logger.info(f"신뢰도: {response.confidence:.2f}")
            
            return AgentResponse(
                success=True,
                message=f"채팅 완료: {len(response.evidence)}개 근거",
                data={
                    "query": request.query,
                    "response": response
                }
            )
                
        except Exception as e:
            logger.error(f"CodeChatAgent 실행 실패: {e}")
            import traceback
            logger.error(f"상세 오류: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500, 
                detail=f"Chat failed: {str(e)}"
            )
            
    except Exception as e:
        logger.error(f"채팅 처리 중 오류: {e}")
        import traceback
        logger.error(f"상세 오류: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/modules", tags=["Modules"])
async def get_modules():
    """사용 가능한 모듈 정보 반환"""
    return {
        "agents": {
            "description": "코드 분석 에이전트들",
            "path": "agents/",
            "status": "active"
        },
        "common": {
            "description": "공통 모듈 (schemas, store, azure)",
            "path": "common/",
            "status": "active"
        },
        "ui": {
            "description": "사용자 인터페이스",
            "path": "ui/",
            "status": "active"
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    # Docker 환경에서는 reload=False, 로컬에서는 reload=True
    reload_enabled = os.environ.get("DOCKER_ENV", "false").lower() != "true"
    
    logger.info(f"🚀 Code Analytica FastAPI Backend 시작 (포트: {port})")
    logger.info(f"🔄 Reload 모드: {'활성화' if reload_enabled else '비활성화'}")
    
    uvicorn.run(app, host=host, port=port, reload=reload_enabled) 