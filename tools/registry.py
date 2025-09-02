import os
from typing import List

try:
    from langchain.tools import BaseTool  # type: ignore
    from langgraph.graph import StateGraph  # type: ignore
    from langgraph.prebuilt import ToolNode  # type: ignore
    _has_langgraph = True
except Exception:
    BaseTool = object  # type: ignore
    StateGraph = None  # type: ignore
    ToolNode = None  # type: ignore
    _has_langgraph = False

from .structsynth_tools import (
    search_symbols_semantic,
    search_symbols_fts,
    get_symbol,
    get_chunks_by_symbol,
    get_chunk,
    get_calls_from,
    get_calls_to,
    list_files,
    list_symbols,
    get_database_stats,
    get_vector_store_stats,
    get_analysis_summary,
    run_repository_analysis,
    get_latest_run,
)

try:
    from .llm_tools import analyze_file_llm, analyze_symbol_llm, analyze_chunk_llm  # type: ignore
    _has_llm_tools = True
except Exception:
    _has_llm_tools = False


def get_tools() -> List[BaseTool]:
    tools: List[BaseTool] = [
        search_symbols_semantic,
        search_symbols_fts,
        get_symbol,
        get_chunks_by_symbol,
        get_chunk,
        get_calls_from,
        get_calls_to,
        list_files,
        list_symbols,
        get_database_stats,
        get_vector_store_stats,
        get_analysis_summary,
        run_repository_analysis,
        get_latest_run,
    ]

    if _has_llm_tools and all(
        os.getenv(var)
        for var in (
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
        )
    ):
        tools.extend([analyze_file_llm, analyze_symbol_llm, analyze_chunk_llm])

    return tools


def get_graph():
    if not _has_langgraph:
        raise RuntimeError("LangGraph 미설치: get_graph를 사용하려면 langgraph를 설치하세요.")
    tools = get_tools()
    tool_node = ToolNode(tools)
    graph = StateGraph(dict)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("tools")
    graph.set_finish_point("tools")
    return graph.compile()


