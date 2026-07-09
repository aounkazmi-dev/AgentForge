from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv
import sqlite3
import io
import contextlib
import pandas as pd
import numpy as np
from scipy import stats

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    schema_summary: str


# ---------------------------------------------------------------------------
#  SQL tool (same as Phase 3)
# ---------------------------------------------------------------------------
_active_con = None


def set_active_connection(con):
    """Call this once, right after data_ingestion.ingest() runs, so the SQL
    tool below has something to query against."""
    global _active_con
    _active_con = con


@tool
def run_sql(query: str) -> str:
    """Run a read-only SQL query against the 'data' table (DuckDB syntax)
    and return the result as text. Use this whenever the user's question
    requires aggregation, filtering, sorting, or computing something from
    the dataset rather than just describing its structure. The table is
    always called 'data'. Example: SELECT city, AVG(salary) AS avg_salary
    FROM data GROUP BY city ORDER BY avg_salary DESC LIMIT 1."""
    if _active_con is None:
        return "No dataset is currently loaded. Ask the user to upload a file first."
    try:
        result_df = _active_con.sql(query).df()
        if result_df.empty:
            return "The query ran successfully but returned no rows."
        return result_df.to_string(index=False)
    except Exception as e:
        return f"SQL error: {e}. Check the column names against the schema and try again."


# ---------------------------------------------------------------------------
#  NEW: Python / stats tool
# ---------------------------------------------------------------------------
_active_df = None


def set_active_dataframe(df):
    """Call this once, right after data_ingestion.ingest() runs, so the
    Python tool below has a DataFrame to analyze."""
    global _active_df
    _active_df = df


# A locked-down set of builtins. Notably ABSENT: open, __import__, exec,
# eval, input, compile, getattr/setattr/delattr (reflection tricks), and
# anything filesystem/network related. This is not bulletproof — a
# determined attacker with local code execution can usually escape a
# Python-level sandbox eventually — but it stops the LLM from accidentally
# (or if prompt-injected, deliberately) doing something like reading/writing
# arbitrary files, since it physically cannot reach `open` from inside exec().
_SAFE_BUILTINS = {
    "print": print, "len": len, "range": range, "sum": sum, "min": min,
    "max": max, "sorted": sorted, "abs": abs, "round": round, "list": list,
    "dict": dict, "set": set, "tuple": tuple, "enumerate": enumerate,
    "zip": zip, "str": str, "int": int, "float": float, "bool": bool,
    "type": type, "isinstance": isinstance,
}


@tool
def run_python(code: str) -> str:
    """Execute Python code to perform statistical analysis on the dataset,
    which is available as a pandas DataFrame called `df`. You also have
    `pd`, `np`, and `stats` (scipy.stats) available. Use this for analysis
    SQL can't easily express: correlations, hypothesis tests, distributions,
    custom multi-step calculations. You MUST use print() to output whatever
    you want to see back — only printed output is returned to you, the
    return value of the code is not captured. Example code:
    print(df['salary'].corr(df['age']))"""
    if _active_df is None:
        return "No dataset is currently loaded. Ask the user to upload a file first."

    safe_globals = {
        "__builtins__": _SAFE_BUILTINS,
        "df": _active_df,
        "pd": pd,
        "np": np,
        "stats": stats,
    }

    output_buffer = io.StringIO()
    try:
        # redirect_stdout captures whatever the executed code print()s,
        # instead of it going to the actual terminal running this app.
        with contextlib.redirect_stdout(output_buffer):
            exec(code, safe_globals)
        output = output_buffer.getvalue()
        if not output.strip():
            return "Code ran successfully but printed nothing. Use print() to output results."
        return output
    except Exception as e:
        return f"Error running code: {e}. Double-check column names and syntax, then try again."


# Bind BOTH tools to the LLM
llm_with_tools = llm.bind_tools([run_sql, run_python])


def chat_node(state: ChatState):
    messages = state["messages"]
    schema_summary = state.get("schema_summary", "")

    if schema_summary:
        system_prompt = (
            "You are a data analyst assistant. A dataset has been loaded "
            "and is available to you in two forms:\n"
            "1. A DuckDB table called 'data' — use the run_sql tool for "
            "aggregation, filtering, sorting, joins.\n"
            "2. A pandas DataFrame called 'df' — use the run_python tool for "
            "correlations, statistical tests, distributions, or anything SQL "
            "can't easily express.\n\n"
            f"Schema:\n{schema_summary}\n\n"
            "Pick whichever tool fits the question. When the user just asks "
            "about the structure of the data, you can answer directly from "
            "the schema above without calling a tool."
        )
        llm_input = [SystemMessage(content=system_prompt)] + messages
    else:
        llm_input = messages

    response = llm_with_tools.invoke(llm_input)
    return {"messages": [response]}


conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
# NEW: ToolNode now knows about both tools
graph.add_node("tools", ToolNode([run_sql, run_python]))

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition, {"tools": "tools", END: END})
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


def retrive_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)