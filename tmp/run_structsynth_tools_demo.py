import json
import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools import get_tools


def find_tool(tools, name):
    for t in tools:
        if t.name == name:
            return t
    raise KeyError(f"Tool not found: {name}")


def has_tool(tools, name):
    try:
        find_tool(tools, name)
        return True
    except KeyError:
        return False


def safe_invoke(tool, kwargs):
    try:
        return tool.invoke(kwargs)
    except Exception as e:
        return {"error": str(e), "tool": tool.name}


def main():
    print("[demo] loading tools...")
    tools = get_tools()
    print(f"[demo] loaded {len(tools)} tools")
    out = {}

    # Basic stats and listings
    print("[demo] calling get_database_stats...")
    out["get_database_stats"] = safe_invoke(find_tool(tools, "get_database_stats"), {"data_dir": "./data"})
    print("[demo] calling list_files...")
    out["list_files"] = safe_invoke(find_tool(tools, "list_files"), {"data_dir": "./data"})
    print("[demo] calling list_symbols...")
    out["list_symbols"] = safe_invoke(find_tool(tools, "list_symbols"), {"limit": 200, "data_dir": "./data"})

    # Searches (prefer FTS to avoid FAISS dependency)
    print("[demo] calling search_symbols_fts...")
    out["search_symbols_fts"] = safe_invoke(
        find_tool(tools, "search_symbols_fts"), {"query": "parse", "top_k": 5, "data_dir": "./data"}
    )
    print("[demo] calling search_symbols_semantic (with fallback)...")
    out["search_symbols_semantic"] = safe_invoke(
        find_tool(tools, "search_symbols_semantic"), {"query": "data processing", "top_k": 5, "repo_path": ".", "data_dir": "./data"}
    )

    # Drill-down on a symbol
    first_symbol_id = None
    if isinstance(out.get("list_symbols"), list) and out["list_symbols"]:
        first_symbol_id = out["list_symbols"][0].get("id")
    elif isinstance(out.get("search_symbols_fts"), list) and out["search_symbols_fts"]:
        first_symbol_id = out["search_symbols_fts"][0].get("id") or out["search_symbols_fts"][0].get("symbol_info", {}).get("id")

    if first_symbol_id:
        print(f"[demo] drilling down symbol_id={first_symbol_id}...")
        out["get_symbol"] = safe_invoke(find_tool(tools, "get_symbol"), {"symbol_id": int(first_symbol_id), "data_dir": "./data"})
        chunks = safe_invoke(find_tool(tools, "get_chunks_by_symbol"), {"symbol_id": int(first_symbol_id), "data_dir": "./data"})
        out["get_chunks_by_symbol"] = chunks
        # Fetch first chunk detail if available
        if isinstance(chunks, list) and chunks:
            first_chunk_id = chunks[0].get("id")
            if first_chunk_id:
                print(f"[demo] fetching chunk_id={first_chunk_id}...")
                out["get_chunk"] = safe_invoke(find_tool(tools, "get_chunk"), {"chunk_id": int(first_chunk_id), "data_dir": "./data"})

    # Find a function symbol to test call graph and chunks
    func_symbol_id = None
    if isinstance(out.get("list_symbols"), list):
        for s in out["list_symbols"]:
            if s.get("type") == "function":
                func_symbol_id = s.get("id")
                break

    if func_symbol_id:
        print(f"[demo] testing calls/chunks for function symbol_id={func_symbol_id}...")
        out["get_calls_from"] = safe_invoke(find_tool(tools, "get_calls_from"), {"caller_id": int(func_symbol_id), "data_dir": "./data"})
        out["get_calls_to"] = safe_invoke(find_tool(tools, "get_calls_to"), {"callee_id": int(func_symbol_id), "data_dir": "./data"})
        func_chunks = safe_invoke(find_tool(tools, "get_chunks_by_symbol"), {"symbol_id": int(func_symbol_id), "data_dir": "./data"})
        out["get_chunks_by_symbol_function"] = func_chunks
        if isinstance(func_chunks, list) and func_chunks:
            out["get_chunk_function"] = safe_invoke(find_tool(tools, "get_chunk"), {"chunk_id": int(func_chunks[0].get("id")), "data_dir": "./data"})

    # Try get_chunk on a likely ID=1 for coverage
    out["get_chunk_maybe_1"] = safe_invoke(find_tool(tools, "get_chunk"), {"chunk_id": 1, "data_dir": "./data"})

    # LLM tools (expected to error without Azure env)
    first_file_path = None
    if isinstance(out.get("list_files"), list) and out["list_files"]:
        first_file_path = out["list_files"][0].get("path")
    if has_tool(tools, "analyze_file_llm") and first_file_path:
        out["analyze_file_llm"] = safe_invoke(find_tool(tools, "analyze_file_llm"), {"file_path": first_file_path, "data_dir": "./data"})
    else:
        out["analyze_file_llm"] = {"skipped": True, "reason": "tool_not_registered"}
    if has_tool(tools, "analyze_symbol_llm") and first_symbol_id:
        out["analyze_symbol_llm"] = safe_invoke(find_tool(tools, "analyze_symbol_llm"), {"symbol_id": int(first_symbol_id), "data_dir": "./data"})
    else:
        out["analyze_symbol_llm"] = {"skipped": True, "reason": "tool_not_registered"}
    if has_tool(tools, "analyze_chunk_llm"):
        out["analyze_chunk_llm"] = safe_invoke(find_tool(tools, "analyze_chunk_llm"), {"chunk_id": 1, "data_dir": "./data"})
    else:
        out["analyze_chunk_llm"] = {"skipped": True, "reason": "tool_not_registered"}

    # Run summary (skip vector store stats to avoid FAISS requirement)
    print("[demo] calling get_latest_run...")
    out["get_latest_run"] = safe_invoke(find_tool(tools, "get_latest_run"), {"agent_name": "StructSynth", "data_dir": "./data"})
    # Optional calls (will safely return error dicts if unavailable)
    out["get_vector_store_stats"] = safe_invoke(find_tool(tools, "get_vector_store_stats"), {"repo_path": ".", "data_dir": "./data"})
    out["get_analysis_summary"] = safe_invoke(find_tool(tools, "get_analysis_summary"), {"repo_path": ".", "data_dir": "./data"})
    out["run_repository_analysis"] = safe_invoke(find_tool(tools, "run_repository_analysis"), {"repo_path": "test_files", "artifacts_dir": "./artifacts", "data_dir": "./data"})

    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())


