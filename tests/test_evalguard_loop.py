import json
import os

from agents.evalguard.contracts import Criterion, LoopConfig
from agents.evalguard.loop import eval_loop


def setup_sample(tmpdir):
    data_dir = tmpdir.mkdir("data").mkdir("insightgen")
    artifacts = tmpdir.mkdir("artifacts")
    toolruns = artifacts.mkdir("toolruns")

    out_path = os.path.join(str(data_dir), "02-sample.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# 샘플 산출물\n\n- 파일 수: 3\n- 파일 목록: a.py, b.py, c.py")

    snap = {
        "files": [{"path": "a.py"}, {"path": "b.py"}, {"path": "c.py"}],
    }
    snap_path = os.path.join(str(toolruns), "02-sample.json")
    with open(snap_path, "w", encoding="utf-8") as f:
        json.dump(snap, f)

    return out_path, os.path.relpath(snap_path, start=str(artifacts)).replace("\\", "/")


def test_eval_loop_reaches_threshold(tmpdir):
    out_path, ref = setup_sample(tmpdir)
    criteria = [
        Criterion(name="파일 존재 여부", type="rule", weight=1.0),
        Criterion(name="파일 개수", type="rule", weight=1.5, tolerance_pct=10.0),
        Criterion(name="경로/수치 정확도", type="rule", weight=1.2),
    ]
    cfg = LoopConfig(pass_threshold=80.0, max_iterations=1)
    report = eval_loop(out_path, criteria, ref, cfg)
    assert report.average >= 80.0
    assert report.passes is True


