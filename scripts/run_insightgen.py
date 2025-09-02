import os
from datetime import datetime

from agents.insightgen.pipeline import run_pipeline


def main() -> None:
    job_id = os.getenv("JOB_ID") or datetime.now().strftime("run-%Y%m%d-%H%M%S")
    out = run_pipeline(
        db_url=os.getenv("INSIGHTGEN_DB_URL", "sqlite:///data/structsynth_code.db"),
        src_root=os.getenv("SRC_ROOT", "."),
        out_dir=os.getenv("OUT_DIR", "data/insightgen"),
        job_id=job_id,
        entries=int(os.getenv("ENTRIES", "5")),
        sinks=int(os.getenv("SINKS", "5")),
        max_depth=int(os.getenv("MAX_DEPTH", "5")),
        max_paths=int(os.getenv("MAX_PATHS_PER_ENTRY", "3")),
        topk=int(os.getenv("TOPK_SYMBOLS", "8")),
    )
    print(out["out_dir"])  # Print output directory path


if __name__ == "__main__":
    main()


