import streamlit as st
from pathlib import Path
import json

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None  # optional


def render(ui):
    st.subheader("Docs")
    st.info("InsightGen 에이전트를 실행하고 산출물을 확인할 수 있습니다.")

    # 간단한 스타일
    st.markdown(
        """
        <style>
        .ig-row { display:flex; gap:12px; }
        .ig-card { flex:1; background:#fff; border:1px solid #e6e8eb; border-radius:10px; padding:12px; box-shadow:0 1px 3px rgba(0,0,0,.06); }
        .ig-title { font-weight:700; font-size:1.05rem; margin-bottom:6px; }
        .ig-badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:.75rem; background:#eef2ff; color:#3730a3; margin-left:6px; }
        .ig-kv { margin:6px 0; }
        .ig-kv b { color:#334155; }
        .artifact-card { background:#fafafa; border:1px solid #ececec; border-radius:8px; padding:12px; margin:8px 0; }
        .artifact-meta { color:#64748b; font-size:.85rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("InsightGen Agent 실행", type="primary"):
            run_insightgen()
    with col2:
        st.caption("버튼을 누르면 플래너 실행 → 툴 호출 → 산출물 생성까지 진행됩니다.")

    # 툴 설명 사전 로드(툴 이름 -> 설명)
    tool_desc_map = load_tool_descriptions()

    result = st.session_state.get("insightgen_result")

    # 항상 보이는 디버그 섹션: 사용 가능한 도구 설명
    st.markdown("---")
    with st.expander("사용 가능한 도구 설명(디버그)"):
        if tool_desc_map:
            st.markdown(f"총 {len(tool_desc_map)}개 도구 로드")
            for n, d in tool_desc_map.items():
                st.markdown(f"- `{n}`: {d or '설명 없음'}")
        else:
            st.info("도구 설명을 불러오지 못했습니다. (get_tools 실패 또는 docstring 미정의)")
    if result:
        st.markdown("---")
        st.subheader("Planner 결과")

        # 카드 형식으로 plans 3개 가로 배치 (+ 선택된 산출물 메타 병합)
        plans = result.get("plans") or []
        selected = result.get("selected_outputs") or []
        meta_by_name = {m.get("name"): m for m in selected if isinstance(m, dict)}
        if plans:
            # 한 줄에 최대 3개씩 칼럼으로 배치 (Streamlit 권장 방식)
            max_cols = 3
            for i in range(0, len(plans), max_cols):
                row_plans = plans[i:i+max_cols]
                cols = st.columns(len(row_plans))
                for idx, plan in enumerate(row_plans):
                    with cols[idx]:
                        meta = meta_by_name.get(plan.get("name")) if meta_by_name else None
                        render_plan_card(plan, meta, tool_desc_map)

        # 선택적: selected_outputs 원본 JSON
        if selected:
            with st.expander("선택된 산출물(원본 JSON)"):
                st.json(selected)

        # 상태/오류 표시
        status, notice, error = result.get("status"), result.get("notice"), result.get("error")
        if status:
            st.info(f"상태: {status}")
        if notice:
            st.caption(notice)
        if error:
            st.error(error)

        # (상단에서 항상 표시하도록 이동)

        st.subheader("최종 산출물(Artifacts)")
        artifacts = result.get("artifacts", [])
        if artifacts:
            for p in artifacts:
                show_artifact(Path(p))
        else:
            st.info("생성된 산출물이 없습니다.")

        # 디버그: 툴 호출 로그
        tlog = result.get("tool_call_log")
        if tlog:
            with st.expander("Tool call log (debug)"):
                st.json(tlog)

    # 산출물 디렉토리 브라우저(결과가 없거나 산출물 리스트가 비어있을 때만 표시)
    if not result or not result.get("artifacts"):
        st.markdown("---")
        st.subheader("산출물 디렉토리: data/insightgen")
        project_root = Path(__file__).parent.parent.parent
        artifacts_dir = project_root / "data" / "insightgen"
        if artifacts_dir.exists():
            files = sorted(artifacts_dir.glob("*"))
            if not files:
                st.info("디렉토리에 파일이 없습니다.")
            for fp in files:
                show_artifact(fp)
        else:
            st.info("디렉토리가 아직 생성되지 않았습니다. InsightGen을 먼저 실행하세요.")


def run_insightgen():
    try:
        if load_dotenv:
            load_dotenv()
        from agents.insightgen.agent import InsightGenAgent
        project_root = Path(__file__).parent.parent.parent
        agent = InsightGenAgent(data_dir=str(project_root / "data"), repo_path=str(project_root))
        with st.spinner("InsightGen Agent 실행 중..."):
            result = agent.analyze()
        st.session_state.insightgen_result = result
        st.success("InsightGen 실행 완료. 결과와 산출물을 아래에서 확인하세요.")
    except Exception as e:
        st.session_state.insightgen_result = {"status": "failed", "error": str(e)}
        st.error(f"InsightGen 실행 실패: {e}")


def show_artifact(fp: Path):
    try:
        stat = fp.stat()
        size_kb = max(1, stat.st_size // 1024)
        st.markdown(f"<div class='artifact-card'><div class='artifact-meta'>📄 {fp.name} · {size_kb} KB</div>", unsafe_allow_html=True)
        with st.expander("내용 보기", expanded=False):
            if fp.suffix.lower() == ".json":
                try:
                    data = json.loads(fp.read_text(encoding="utf-8"))
                    st.json(data)
                except Exception:
                    st.code(fp.read_text(encoding="utf-8"), language="json")
            elif fp.suffix.lower() in {".md", ".txt"}:
                st.markdown(fp.read_text(encoding="utf-8"))
            else:
                st.code(fp.read_text(encoding="utf-8", errors="ignore"))
        st.markdown("</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"파일 표시 실패: {fp.name} ({e})")


def render_plan_card(plan: dict, meta: dict | None, tool_desc_map: dict[str, str]):
    # 필수 콘텐츠가 전혀 없으면 카드 출력 생략
    has_any = any([
        plan.get('name'), plan.get('file_ext'), plan.get('tool_invocations'), plan.get('prompt_template')
    ])
    if not has_any:
        return

    st.markdown("<div class='ig-card'>", unsafe_allow_html=True)
    title = (plan.get('name') or '-').strip()
    ext = (plan.get('file_ext') or '-').strip()
    st.markdown(f"<div class='ig-title'>{title}<span class='ig-badge'>{ext}</span></div>", unsafe_allow_html=True)

    # reason / evaluation_criteria
    if meta:
        reason = meta.get('reason')
        evals = meta.get('evaluation_criteria') or []
        if reason:
            st.markdown(f"<div class='ig-kv'><b>이유:</b> {reason}</div>", unsafe_allow_html=True)
        if evals:
            st.markdown("<div class='ig-kv'><b>평가 기준:</b></div>", unsafe_allow_html=True)
            for e in evals:
                st.markdown(f"- {e}")

    # Tools with 친화적 설명
    invs = plan.get('tool_invocations', []) or []
    if invs:
        st.markdown("<div class='ig-kv'><b>사용 도구:</b></div>", unsafe_allow_html=True)
        for inv in invs:
            tname = inv.get('name','?')
            alias = inv.get('alias') or tname
            params = inv.get('params', {})
            purpose = inv.get('purpose') or ''
            desc = tool_desc_map.get(tname) or '설명 없음'
            st.markdown(f"- `{tname}` as `{alias}` → {json.dumps(params, ensure_ascii=False)}")
            if desc:
                st.caption(f"도구 설명: {desc}")
            if purpose:
                st.caption(f"목적: {purpose}")

    # Prompt Template (상세)
    tmpl = plan.get('prompt_template') or ''
    if tmpl:
        st.markdown("<div class='ig-kv'><b>Prompt Template:</b></div>", unsafe_allow_html=True)
        st.code(tmpl)

    st.markdown("</div>", unsafe_allow_html=True)


def load_tool_descriptions() -> dict[str, str]:
    try:
        from tools import get_tools
        d = {}
        for t in get_tools():
            # description은 데코레이터에서 docstring으로 채워짐
            desc = getattr(t, 'description', None)
            name = getattr(t, 'name', None)
            if name:
                d[name] = (desc or '').strip()
        return d
    except Exception:
        return {}


