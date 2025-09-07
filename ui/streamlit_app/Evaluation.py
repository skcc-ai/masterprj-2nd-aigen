import json
import os
from pathlib import Path
import streamlit as st


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _load_trace(trace_path: Path) -> list:
    try:
        if not trace_path.exists():
            return []
        lines = trace_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(ln) for ln in lines if ln.strip()]
    except Exception:
        return []

def _get_block_events_for_output(trace_events: list, output_name: str, output_path: Path) -> list:
    """Return the most recent block of events for this output using report_path matching.
    Strategy: find latest 'report' event whose data.report_path endswith this output's report path,
    then expand to its surrounding start_once block.
    """
    # expected report file basenames
    base1 = f"{output_name}.report.json"
    base2 = f"{Path(output_name).stem}.report.json"

    report_idx = -1
    for idx in range(len(trace_events) - 1, -1, -1):
        ev = trace_events[idx]
        if ev.get("event") == "report":
            data = ev.get("data", {})
            rpath = (data.get("report_path") or "").replace("\\", "/")
            rbase = Path(rpath).name
            if rbase == base1 or rbase == base2:
                report_idx = idx
                break
    if report_idx == -1:
        return []
    # walk back to start_once
    start_idx = report_idx
    while start_idx >= 0 and trace_events[start_idx].get("event") != "start_once":
        start_idx -= 1
    if start_idx < 0:
        start_idx = report_idx
    # collect until next start_once
    block = []
    for ev in trace_events[start_idx:]:
        if ev.get("event") == "start_once" and len(block) > 0:
            break
        block.append(ev)
    return block


def render(ui):
    st.subheader("Evaluation")
    st.info("Evaluation에이전트를 실행하고 품질 평가 및 재생산을 할 수 있습니다.")
    project_root = Path(__file__).parent.parent.parent
    outputs_dir = project_root / "data" / "insightgen"
    eval_dir = project_root / "artifacts" / "eval"
    evalguard_out_dir = project_root / "data" / "evalguard"
    toolruns_dir = project_root / "artifacts" / "toolruns"

    # session flags for per-output fresh-run visibility
    if 'eval_runs' not in st.session_state:
        st.session_state.eval_runs = {}

    # Global info
    st.caption("InsightGen 산출물(md/json)을 평가하고, 80점 미만이면 개선 후 재생성합니다. 보고서는 artifacts/eval/*.report.json 에 저장됩니다.")
    show_prev = st.checkbox("이전 평가 결과도 표시", value=False)

    st.markdown("---")
    st.subheader("Outputs & Scores")

    if not outputs_dir.exists():
        st.info("산출물 디렉토리가 없습니다. 먼저 InsightGen을 실행하세요.")
        return

    files = sorted(outputs_dir.glob("*"))
    if not files:
        st.info("표시할 산출물이 없습니다.")
        return

    # load trace once
    trace_path_global = project_root / "artifacts" / "eval" / "trace.jsonl"
    trace_lines = []
    if trace_path_global.exists():
        try:
            trace_lines = [json.loads(ln) for ln in trace_path_global.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except Exception:
            trace_lines = []

    for fp in files:
        name = fp.name
        report_path = eval_dir / f"{name}.report.json"
        # fallback: some outputs might have different base naming; try without extension variants
        if not report_path.exists():
            base = Path(name).stem
            alt = eval_dir / f"{base}.report.json"
            if alt.exists():
                report_path = alt
        cols = st.columns([2, 1])
        with cols[0]:
            st.markdown(f"**{name}**")
            # Per-output Evaluation Agent run
            run_col, _ = st.columns([1, 3])
            with run_col:
                if st.button(f"이 산출물 평가 실행", key=f"run_{name}"):
                    with st.spinner(f"Evaluating {name}..."):
                        try:
                            import subprocess, sys
                            # criteria file 선택(optional)
                            crit_path = toolruns_dir / f"{name}.meta.json"
                            cmd = [
                                sys.executable, "-m", "agents.evalguard.runner",
                                "--output_path", str(fp),
                                "--snapshots_dir", str(toolruns_dir),
                            ]
                            if crit_path.exists():
                                cmd += ["--criteria_json", str(crit_path)]
                            st.caption(" ".join(cmd))
                            # ensure eval output dirs exist and force sync
                            subprocess.run(cmd, cwd=str(project_root), check=True)
                            st.session_state.eval_runs[name] = True
                            # after run, force rerender with latest files
                            st.rerun()
                            st.success("Eval finished. Report updated.")
                        except Exception as e:
                            st.error(f"Eval run failed: {e}")
            with st.expander("산출물 보기"):
                st.markdown(_read_text(fp))
            # Show improved output (if exists)
            improved_path = evalguard_out_dir / name
            if improved_path.exists():
                with st.expander("개선된 산출물 보기 (EvalGuard)"):
                    st.markdown(_read_text(improved_path))
        with cols[1]:
            # Snapshot availability hint
            snap_exists = (toolruns_dir / f"{Path(name).stem}.json").exists() or (toolruns_dir / f"{name}.json").exists()
            if not snap_exists:
                st.warning("스냅샷 파일이 없어 규칙 평가 정확도가 제한됩니다. (artifacts/toolruns/<name>.json)")
            if report_path.exists():
                rep = _read_json(report_path)
                # Show blank before first run in this session unless user opted to show previous
                if not st.session_state.eval_runs.get(name) and not show_prev:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("최초 평균", "-")
                    with c2:
                        st.metric("최종 평균", "-")
                    st.caption("이 산출물에 대한 평가를 먼저 수행하세요.")
                elif rep:
                    avg = rep.get("average", 0)
                    # compute first average from trace block (for this output only)
                    block = _get_block_events_for_output(trace_lines, name, fp)
                    first_avg = None
                    for ev in block:
                        if ev.get("event") == "report":
                            data = ev.get("data", {})
                            first_avg = (data.get("average") if isinstance(data, dict) else None)
                            break
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("최초 평균", first_avg if first_avg is not None else "-")
                    with c2:
                        st.metric("최종 평균", avg)
                    # show recursion info by scanning trace for this file
                    iters = sum(1 for ev in block if ev.get("event") == "iteration")
                    improves = sum(1 for ev in block if ev.get("event") == "improve_prompt")
                    st.caption(f"passes(avg≥80): {rep.get('passes')}, 마지막 반복: {rep.get('iteration')}, 총 반복(추정): {iters}")
                    if improves:
                        st.info(f"재귀 수행됨: 프롬프트 재설계 {improves}회")
                    if rep.get("llm_skipped"):
                        st.warning(rep.get("notice") or "LLM 평가가 수행되지 않았습니다.")
                    per = rep.get("per_criterion", {}) or {}
                    with st.expander("세부 점수(per_criterion)"):
                        st.json(per)
                    per_min_ok = all(((v or {}).get("score", 0) >= 70) for v in per.values())
                    if per_min_ok:
                        st.success("최종 산출물: 모든 평가 기준(≥70) 충족")
                    else:
                        st.error("최종 산출물: 일부 평가 기준이 미달(70 미만)")
                    # 최초 1회 프롬프트/응답/개선 프롬프트 (이 산출물 블록 기준)
                    with st.expander("최초 1회 평가 프롬프트/응답/개선 프롬프트"):
                        try:
                            first_eval_prompt = next((ev.get("data") for ev in block if ev.get("event") == "llm_eval_prompt"), None)
                            first_eval_resp = next((ev.get("data") for ev in block if ev.get("event") == "llm_eval_response"), None)
                            first_improve_prompt = next((ev.get("data") for ev in block if ev.get("event") == "improve_prompt"), None)
                            st.markdown("**평가 프롬프트(1회차)**")
                            st.code(json.dumps(first_eval_prompt or {}, ensure_ascii=False, indent=2), language="json")
                            st.markdown("**평가 응답(1회차)**")
                            st.code(json.dumps(first_eval_resp or {}, ensure_ascii=False, indent=2), language="json")
                            st.markdown("**개선 프롬프트(1회차)**")
                            st.code(json.dumps(first_improve_prompt or {}, ensure_ascii=False, indent=2), language="json")
                        except Exception:
                            st.caption("trace 읽기 실패")
                    # 루프 중단 사유 표시(10회 초과/개선 불가)
                    stop_reason = None
                    if any(ev.get("event") == "final_saved" and (ev.get("data") or {}).get("reason") == "loop_exhausted" for ev in block):
                        stop_reason = "loop_exhausted"
                    elif any(ev.get("event") == "final_saved" and (ev.get("data") or {}).get("reason") == "no_improvement" for ev in block):
                        stop_reason = "no_improvement"
                    if stop_reason == "loop_exhausted":
                        st.warning("평가 10회 수행 후 중단되었습니다.")
                    elif stop_reason == "no_improvement":
                        st.warning("개선 텍스트를 생성하지 못해 중단되었습니다.")
                else:
                    st.warning("리포트 파싱 실패")
            else:
                if not st.session_state.eval_runs.get(name) and not show_prev:
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("최초 평균", "-")
                    with c2:
                        st.metric("최종 평균", "-")
                    st.caption("이 산출물에 대한 평가를 먼저 수행하세요.")
                else:
                    st.info("평가 리포트 없음")


