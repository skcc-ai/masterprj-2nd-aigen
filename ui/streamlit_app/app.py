"""
Code Analytica - Streamlit UI
AI 기반 코드 분석 및 채팅 시스템의 사용자 인터페이스
"""

import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime
import requests
from typing import Dict, Any, Optional

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 페이지 설정
st.set_page_config(
    page_title="Code Analytica",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 2rem;
    text-align: center;
}

.main-header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: bold;
}

.main-header p {
    margin: 0.5rem 0 0 0;
    font-size: 1.2rem;
    opacity: 0.9;
}

.status-card {
    background: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
}

.status-success {
    border-left: 4px solid #28a745;
}

.status-warning {
    border-left: 4px solid #ffc107;
}

.status-error {
    border-left: 4px solid #dc3545;
}

.analysis-section, .chat-section {
    background: white;
    border: 1px solid #e9ecef;
    border-radius: 10px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

class CodeAnalyticaUI:
    """Code Analytica 메인 UI 클래스"""
    
    def __init__(self):
        """UI 초기화"""
        self.api_url = self._get_api_url()
        self.init_session_state()
    
    def _get_api_url(self) -> str:
        return "http://backend:8000"
    
    def init_session_state(self):
        """세션 상태 초기화"""
        if 'analysis_completed' not in st.session_state:
            st.session_state.analysis_completed = False
        if 'analysis_output' not in st.session_state:
            st.session_state.analysis_output = None
        if 'analysis_directory' not in st.session_state:
            st.session_state.analysis_directory = None
        if 'codechat_agent' not in st.session_state:
            st.session_state.codechat_agent = None
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
    
    def run(self):
        """메인 UI 실행"""
        self.render_header()
        self.render_tabs()
        self.render_footer()
    
    def render_header(self):
        """헤더 렌더링"""
        st.markdown(
            '<div class="main-header">'
            '<h1>Code Analytica</h1>'
            '<p>AI 기반 코드 분석 및 채팅 시스템</p>'
            '</div>',
            unsafe_allow_html=True
        )
    
    def render_tabs(self):
        """탭 렌더링"""
        tab1, tab2, tab3, tab4 = st.tabs([
            "Code Analysis", "Code Chat", "Docs", "Status"
        ])
        
        with tab1:
            self.render_code_analysis()
        with tab2:
            self.render_code_chat()
        with tab3:
            self.render_docs()
        with tab4:
            self.render_status()
    
    def render_code_analysis(self):
        """코드 분석 탭 렌더링"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Repository Configuration")
            st.info("분석할 코드 디렉토리의 경로를 직접 입력하세요")
            
            # 디렉토리 경로 입력
            directory_path = self.get_directory_input()
            
            if directory_path.strip():
                self.handle_directory_analysis(directory_path)
        
        with col2:
            self.render_analysis_status()
    
    def get_directory_input(self) -> str:
        """디렉토리 입력 UI"""
        return st.text_input(
            "분석할 디렉토리 경로",
            value=st.session_state.get("selected_directory", ""),
            placeholder="/Users/username/Desktop/my-project",
            help="분석할 코드 디렉토리의 전체 경로를 입력하세요"
        )
    
    def handle_directory_analysis(self, directory_path: str):
        """디렉토리 분석 처리"""
        path_obj = Path(directory_path)
        if path_obj.exists() and path_obj.is_dir():
            st.success(f"유효한 디렉토리: {directory_path}")
            self.show_directory_preview(path_obj)
            
            if st.button("분석 시작", type="primary"):
                self.start_analysis(directory_path)
        else:
            st.error("유효하지 않은 디렉토리 경로입니다.")
    
    def show_directory_preview(self, path_obj: Path):
        """디렉토리 미리보기"""
        try:
            files = list(path_obj.rglob("*.py")) + list(path_obj.rglob("*.js")) + list(path_obj.rglob("*.java"))
            if files:
                st.info(f"발견된 코드 파일: {len(files)}개")
                with st.expander("파일 목록 보기"):
                    for file in files[:20]:
                        st.text(f"• {file.relative_to(path_obj)}")
                    if len(files) > 20:
                        st.text(f"... 및 {len(files) - 20}개 더")
            else:
                st.warning("코드 파일을 찾을 수 없습니다.")
        except Exception as e:
            st.warning(f"파일 스캔 실패: {str(e)}")
    
    def start_analysis(self, directory_path: str):
        """분석 시작"""
        with st.spinner("StructSynth 에이전트 실행 중..."):
            success, output = self.run_structsynth_analysis(directory_path)
            if success:
                st.session_state.analysis_completed = True
                st.session_state.analysis_output = output
                st.session_state.analysis_directory = directory_path
                st.rerun()
    
    def render_analysis_status(self):
        """분석 상태 렌더링"""
        st.subheader("분석 상태")
        
        if st.session_state.analysis_completed:
            st.markdown(
                '<div class="status-card status-success">분석 완료</div>',
                unsafe_allow_html=True
            )
            st.info("이제 CodeChat과 Docs를 사용할 수 있습니다!")
            
            if st.session_state.analysis_directory:
                st.success(f"분석된 디렉토리: {st.session_state.analysis_directory}")
            
            if st.session_state.analysis_output:
                with st.expander("분석 결과 요약"):
                    output_text = st.session_state.analysis_output
                    if len(output_text) > 500:
                        output_text = output_text[:500] + "..."
                    st.text(output_text)
        else:
            st.markdown(
                '<div class="status-card status-warning">분석 대기 중</div>',
                unsafe_allow_html=True
            )
            st.info("디렉토리를 입력하고 StructSynth 분석을 시작하세요.")
    
    def render_code_chat(self):
        """코드 채팅 탭 렌더링"""
        st.subheader("Code Chat")
        st.info("코드베이스에 대해 질문하고 AI가 답변해드립니다.")
        
        # 에이전트 초기화 상태 확인
        self.check_agent_status()
        
        # 에이전트 초기화
        if not st.session_state.codechat_agent:
            self.init_codechat_agent()
        
        # 채팅 인터페이스 렌더링
        if st.session_state.codechat_agent:
            self.render_chat_interface()
    
    def check_agent_status(self):
        """에이전트 상태 확인 및 표시"""
        if st.session_state.codechat_agent:
            try:
                # 간단한 상태 확인
                agent = st.session_state.codechat_agent
                if hasattr(agent, 'sqlite_store') and agent.sqlite_store:
                    st.success("✅ CodeChat Agent가 정상적으로 초기화되었습니다.")
                else:
                    st.warning("⚠️ CodeChat Agent가 부분적으로 초기화되었습니다.")
            except Exception as e:
                st.error(f"❌ CodeChat Agent 상태 확인 실패: {str(e)}")
                st.session_state.codechat_agent = None
        else:
            st.info("🔄 CodeChat Agent를 초기화하는 중...")
    
    def init_codechat_agent(self):
        """CodeChat 에이전트 초기화"""
        if st.session_state.codechat_agent is None:
            try:
                from agents.codechat.agent import CodeChatAgent
                
                # 프로젝트 루트 경로 계산
                project_root = Path(__file__).parent.parent.parent
                artifacts_dir = project_root / "artifacts"
                data_dir = project_root / "data"
                
                # CodeChatAgent 초기화
                st.session_state.codechat_agent = CodeChatAgent(
                    repo_path=str(project_root),
                    artifacts_dir=str(artifacts_dir),
                    data_dir=str(data_dir)
                )
                st.success("CodeChat Agent 초기화 완료!")
                
            except Exception as e:
                st.error(f"CodeChat Agent 초기화 실패: {str(e)}")
                st.info("에이전트 초기화에 실패했습니다. 환경 변수와 경로를 확인해주세요.")
                st.session_state.codechat_agent = None
    
    def render_chat_interface(self):
        """채팅 인터페이스 렌더링"""
        # 입력 영역
        col1, col2 = st.columns([4, 1])
        with col1:
            user_query = st.text_input(
                "코드에 대해 질문하세요:",
                placeholder="예: 데이터 처리 함수는 어떻게 작동하나요?"
                # key 제거 - 세션 상태와의 충돌 방지
            )
        with col2:
            search_top_k = st.number_input("검색 결과 수", min_value=1, max_value=10, value=5)
            send_button = st.button("질문하기", type="primary")
        
        # 질문 처리 - 더 엄격한 검증
        if send_button:
            if user_query and user_query.strip():
                self.process_chat_query(user_query.strip(), search_top_k)
            else:
                st.error("❌ 질문을 입력해주세요.")
        
        # 채팅 히스토리
        self.render_chat_history()
    
    def process_chat_query(self, query: str, top_k: int):
        """채팅 질문 처리"""
        # 상세한 query 검증 및 로깅
        st.info(f"🔍 질문 검증 중: '{query}' (타입: {type(query)}, 길이: {len(query) if query else 0})")
        
        if query is None:
            st.error("❌ 질문이 None입니다.")
            return
        
        if not isinstance(query, str):
            st.error(f"❌ 질문이 문자열이 아닙니다: {type(query)}")
            return
        
        if not query.strip():
            st.error("❌ 질문이 비어있습니다.")
            return
        
        with st.spinner("분석 중..."):
            try:
                # 백엔드 API를 통해 채팅 요청 전송
                response = self._send_chat_request_to_backend(query.strip(), top_k)
                
                if response and response.get("success"):
                    # 사용자 메시지 추가
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": query.strip(),
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                    
                    # 어시스턴트 응답 추가
                    chat_data = response.get("data", {})
                    chat_response = chat_data.get("response", {})
                    
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": chat_response.get("answer", "응답을 생성할 수 없습니다."),
                        "evidence": chat_response.get("evidence", []),
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
                    
                    st.rerun()
                else:
                    st.error(f"❌ 백엔드 응답 실패: {response.get('error', '알 수 없는 오류')}")
                    
            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")
                st.info("💡 백엔드 연결에 문제가 있을 수 있습니다.")
    
    def render_chat_history(self):
        """채팅 히스토리 렌더링"""
        st.markdown("---")
        st.subheader("채팅 히스토리")
        
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                self.render_user_message(message)
            else:
                self.render_assistant_message(message)
        
        # 히스토리 초기화 버튼
        if st.session_state.chat_history:
            if st.button("채팅 히스토리 초기화"):
                st.session_state.chat_history = []
                st.rerun()
    
    def render_user_message(self, message: Dict[str, Any]):
        """사용자 메시지 렌더링"""
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-start; margin: 0.5rem 0;">
                <div style="background: #e3f2fd; padding: 1rem; border-radius: 15px; max-width: 70%;">
                    <strong>사용자</strong><br>
                    {message["content"]}
                    <div style="font-size: 0.8rem; color: #666; text-align: right;">
                        {message['timestamp']}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def render_assistant_message(self, message: Dict[str, Any]):
        """어시스턴트 메시지 렌더링"""
        st.markdown(
            f"""
            <div style="display: flex; justify-content: flex-end; margin: 0.5rem 0;">
                <div style="background: #f1f8e9; padding: 1rem; border-radius: 15px; max-width: 70%;">
                    <strong>어시스턴트</strong><br>
                    {message["content"]}
                    <div style="font-size: 0.8rem; color: #666; text-align: right;">
                        {message['timestamp']}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 근거 정보 표시
        if "evidence" in message and message["evidence"]:
            self.render_evidence(message["evidence"])
    
    def render_evidence(self, evidence_list: list):
        """근거 정보 렌더링"""
        for i, evidence in enumerate(evidence_list, 1):
            # evidence 객체의 타입과 구조 확인
            try:
                # SearchResult 객체인지 확인하고 안전하게 속성 접근
                if hasattr(evidence, 'symbol_name') and hasattr(evidence, 'symbol_type'):
                    # SearchResult 객체인 경우
                    symbol_name = evidence.symbol_name
                    symbol_type = evidence.symbol_type
                    file_path = getattr(evidence, 'file_path', 'N/A')
                    start_line = getattr(evidence, 'start_line', 0)
                    end_line = getattr(evidence, 'end_line', 0)
                    source = getattr(evidence, 'source', 'N/A')
                    similarity_score = getattr(evidence, 'similarity_score', 0.0)
                    content = getattr(evidence, 'content', '')
                else:
                    # 딕셔너리인 경우
                    symbol_name = evidence.get('symbol_name', 'Unknown')
                    symbol_type = evidence.get('symbol_type', 'Unknown')
                    file_path = evidence.get('file_path', 'N/A')
                    start_line = evidence.get('start_line', 0)
                    end_line = evidence.get('end_line', 0)
                    source = evidence.get('source', 'N/A')
                    similarity_score = evidence.get('similarity_score', 0.0)
                    content = evidence.get('content', '')
                
                with st.expander(f"근거 {i}: {symbol_name} ({symbol_type})"):
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        st.markdown(f"**파일**: `{file_path}`")
                        st.markdown(f"**라인**: {start_line}-{end_line}")
                        st.markdown(f"**소스**: {source} (유사도: {similarity_score:.4f})")
                    with col2:
                        st.code(content, language="python")
                        
            except Exception as e:
                # 오류 발생 시 기본 정보만 표시
                st.error(f"근거 {i} 렌더링 오류: {str(e)}")
                st.json(evidence)  # 디버깅을 위해 원본 데이터 표시
    
    def render_docs(self):
        """문서 탭 렌더링"""
        st.subheader("개발 진행 중...")
    
    def render_status(self):
        """상태 정보 탭 렌더링"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("시스템 정보")
            st.info(f"**Python 버전**: {sys.version}")
            st.info(f"**Streamlit 버전**: {st.__version__}")
            st.info(f"**작업 디렉토리**: {Path.cwd()}")
            
            st.subheader("분석 상태")
            if st.session_state.analysis_completed:
                st.success("소스코드 분석 완료")
            else:
                st.warning("소스코드 분석 대기 중")
        
        with col2:
            st.subheader("에이전트 상태")
            if st.session_state.codechat_agent:
                st.success("CodeChat Agent 활성화")
            else:
                st.warning("CodeChat Agent 비활성화")
    
    def render_footer(self):
        """푸터 렌더링"""
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666;'>"
            "Code Analytica - AI 기반 코드 분석 시스템 | "
            f"© 2024 | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            "</div>",
            unsafe_allow_html=True
        )
    
    def run_structsynth_analysis(self, directory_path: str) -> tuple[bool, str]:
        """StructSynth 에이전트를 실행하여 코드 분석 수행 (API 호출)"""
        try:
            # 요청 데이터 준비
            api_url = f"{self.api_url}/api/agents/run-all"
            request_data = {
                "repo_path": directory_path,
                "artifacts_dir": "./artifacts",
                "data_dir": "./data"
            }
            
            st.info("StructSynth 에이전트 실행 중... (API 호출)")
            st.info(f"API 엔드포인트: {api_url}")
            st.info(f"분석 대상: {directory_path}")
            
            # API 요청 전송
            response = requests.post(api_url, json=request_data, timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    st.success("코드 분석 완료")
                    return True, str(result.get('data', {}))
                else:
                    st.error(f"API 분석 실패: {result.get('error', 'Unknown error')}")
                    return False, result.get('error', 'Unknown error')
            else:
                st.error(f"API 요청 실패: HTTP {response.status_code}")
                st.error(f"응답 내용: {response.text}")
                return False, f"HTTP {response.status_code}: {response.text}"
                
        except requests.exceptions.ConnectionError:
            st.error("API 서버에 연결할 수 없습니다.")
            st.info("FastAPI 백엔드를 먼저 실행해주세요:")
            st.markdown("""
            1. **터미널에서 백엔드 실행**: `python api/main.py`
            2. **또는 Docker로 실행**: `docker run -p 8000:8000 code-analytica:latest`
            3. **API 서버가 실행되면 다시 시도하세요**
            """)
            return False, "API connection failed"
        except requests.exceptions.Timeout:
            st.error("API 요청 시간 초과 (5분)")
            return False, "API timeout"
        except Exception as e:
            st.error(f"API 요청 오류: {str(e)}")
            st.error(f"오류 타입: {type(e).__name__}")
            import traceback
            st.error(f"상세 오류: {traceback.format_exc()}")
            return False, str(e)

    def _send_chat_request_to_backend(self, query: str, top_k: int) -> Optional[Dict[str, Any]]:
        """백엔드 API로 채팅 요청 전송"""
        try:
            import requests
            
            # 백엔드 API URL
            backend_url = f"{self.api_url}/api/chat"
            
            # 요청 데이터
            request_data = {
                "query": query,
                "top_k": top_k
            }
            
            st.info(f"🚀 백엔드 API 호출: {backend_url}")
            st.info(f"📤 요청 데이터: {request_data}")
            
            # POST 요청 전송
            response = requests.post(
                backend_url,
                json=request_data,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ 백엔드 응답 성공: {result.get('message', '')}")
                return result
            else:
                st.error(f"❌ 백엔드 응답 실패 (HTTP {response.status_code}): {response.text}")
                return None
                
        except requests.exceptions.ConnectionError:
            st.error("❌ 백엔드 서버에 연결할 수 없습니다.")
            st.info("백엔드 서버가 실행 중인지 확인해주세요.")
            return None
        except requests.exceptions.Timeout:
            st.error("❌ 백엔드 응답 시간 초과")
            return None
        except Exception as e:
            st.error(f"❌ 백엔드 요청 실패: {str(e)}")
            return None

# 메인 실행
if __name__ == "__main__":
    ui = CodeAnalyticaUI()
    ui.run()
