import sqlite3
import json
import os
from typing import List, Dict, Any


def main() -> None:
    db_path = os.path.join('data', 'structsynth_code.db')
    if not os.path.exists(db_path):
        print(json.dumps({"error": "db_not_found", "path": db_path}, ensure_ascii=False))
        return

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    result: Dict[str, Any] = {}

    # Tables
    tables = [row[0] for row in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()]
    result['tables'] = tables

    # Schema
    schema: Dict[str, List[Dict[str, Any]]] = {}
    for t in tables:
        cols = [
            {
                'name': r[1],
                'type': r[2],
                'notnull': r[3],
                'default': r[4],
                'pk': r[5],
            }
            for r in cur.execute(f"PRAGMA table_info({t})").fetchall()
        ]
        schema[t] = cols
    result['schema'] = schema

    # Counts
    counts: Dict[str, Any] = {}
    for t in tables:
        try:
            counts[t] = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except Exception as e:
            counts[t] = str(e)
    result['counts'] = counts

    def sample(query: str, limit: int = 10) -> List[Dict[str, Any]]:
        try:
            rows = cur.execute(query).fetchmany(limit)
            return [dict(r) for r in rows]
        except Exception as e:
            return [{"error": str(e)}]

    # Samples from each table
    if 'files' in tables:
        result['files'] = sample(
            "SELECT id, path, language, llm_summary, created_at FROM files ORDER BY id DESC",
            20,
        )
    if 'symbols' in tables:
        result['symbols'] = sample(
            "SELECT id, name, type, language, start_line, end_line, llm_summary FROM symbols ORDER BY id DESC",
            20,
        )
    if 'chunks' in tables:
        result['chunks'] = sample(
            "SELECT id, symbol_id, chunk_type, LENGTH(content) AS len, tokens, embedding_id FROM chunks ORDER BY id LIMIT 20",
            20,
        )
    if 'calls' in tables:
        result['calls'] = sample(
            "SELECT id, caller_id, callee_id, call_type, confidence FROM calls ORDER BY id LIMIT 20",
            20,
        )
    if 'embeddings' in tables:
        result['embeddings'] = sample(
            "SELECT id, object_type, object_id, dimension, created_at FROM embeddings ORDER BY id LIMIT 20",
            20,
        )
        result['embedding_dims'] = sample(
            "SELECT dimension, COUNT(*) AS n FROM embeddings GROUP BY dimension ORDER BY n DESC",
            100,
        )
    if 'runs' in tables:
        result['runs'] = sample(
            "SELECT id, agent_name, status, started_at, finished_at FROM runs ORDER BY id DESC",
            10,
        )

    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()


