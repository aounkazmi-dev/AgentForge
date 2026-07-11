"""
The LangGraph graph itself: state definition, the main chat node, the
text-based code-execution fallback, and routing between them. All the
actual tool implementations live in tools.py; small shared helpers live
in utils.py. This file is just wiring.
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import io
import contextlib
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go

from tools import (
    ALL_TOOLS,
    _thread_id_from_config,
    set_active_dataframe,
    set_active_connection,
    get_last_figure,
    clear_last_figure,
    set_last_figure,
    get_active_dataframe,
)
from utils import get_text_content, extract_python_code, SAFE_BUILTINS

load_dotenv()

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    max_tokens=4096,
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    schema_summary: str


llm_with_tools = llm.bind_tools(ALL_TOOLS)


def code_exec_node(state: ChatState, config: RunnableConfig):
    """Runs whatever python code block the LLM just wrote in plain text
    (the fallback for anything the 6 structured tools don't cover), using
    THIS thread's dataframe, then feeds the printed output back in as a
    new message so chat_node can turn it into a final answer."""
    thread_id = _thread_id_from_config(config)
    df = get_active_dataframe(thread_id)

    last_msg = state["messages"][-1]
    text = get_text_content(last_msg.content)
    code = extract_python_code(text)

    if not code:
        return {"messages": [HumanMessage(content="[Code execution] No code block found to run.")]}

    safe_globals = {
        "__builtins__": SAFE_BUILTINS,
        "df": df,
        "pd": pd,
        "np": np,
        "stats": stats,
    }
    output_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, safe_globals)
        output = output_buffer.getvalue().strip() or "Code ran successfully but printed nothing."

        maybe_fig = safe_globals.get("fig")
        if isinstance(maybe_fig, go.Figure):
            set_last_figure(maybe_fig.to_json(), thread_id)
            output += "\n\nA chart was created and is ready to display to the user."
    except Exception as e:
        output = f"Error running the code: {e}"

    return {"messages": [HumanMessage(
        content=f"[Code execution result]\n{output}\n\nNow give the user a clear, final answer based on this result. Do not write more code unless it's truly necessary."
    )]}


def route_after_agent(state: ChatState):
    """Decide what happens after chat_node runs: a real tool call, a
    text-based code block, or we're done."""
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tools"
    text = get_text_content(last_msg.content)
    if extract_python_code(text):
        return "code_exec"
    return END


def chat_node(state: ChatState):
    messages = state["messages"]
    schema_summary = state.get("schema_summary", "")

    if schema_summary:
        system_prompt = (
            "You are a data analyst assistant. A dataset has been loaded "
            "and is available to you. You have these tools:\n"
            "1. run_sql — for aggregation, filtering, sorting, joins via SQL "
            "against the DuckDB table called 'data'.\n"
            "2. compute_correlation — for checking if two numeric columns "
            "are related.\n"
            "3. describe_column — for summary statistics or distribution of "
            "one column.\n"
            "4. run_ttest — for testing if a numeric column differs "
            "significantly between two groups.\n"
            "5. create_chart — for visualizing the data as a bar, line, "
            "scatter, histogram, or box chart.\n"
            "6. generate_data_report — for an overall summary/report of "
            "the whole dataset (descriptive stats, missing values, top "
            "correlations, category breakdowns). When you use this tool, "
            "don't just repeat the numbers back — write a short business-"
            "insight narrative: what stands out, what's worth investigating "
            "further, and what it might mean.\n\n"
            "If none of these tools fit what's being asked, you may instead "
            "write a single python code block (```python ... ```) using the "
            "pandas DataFrame `df`, plus `pd`, `np`, and `stats` (scipy.stats) "
            "which are all available. Use print() for anything you want to "
            "see back. If you create a chart this way using plotly, assign "
            "it to a variable named exactly `fig` and it will be displayed "
            "automatically. Only do this when the 6 tools above genuinely "
            "don't cover the question — prefer the structured tools "
            "whenever they fit.\n\n"
            f"Schema:\n{schema_summary}\n\n"
            "When the user just asks about the structure of the data, you "
            "can answer directly from the schema above without any tool."
        )
        llm_input = [SystemMessage(content=system_prompt)] + messages
    else:
        llm_input = messages

    try:
        response = llm_with_tools.invoke(llm_input)
    except Exception as e:

        print(f"\n[chat_node] llm_with_tools.invoke failed: {e}\n")
        if "tool_use_failed" in str(e) or "Failed to call a function" in str(e):
            response = llm.invoke(llm_input)
        else:
            raise
    return {"messages": [response]}



checkpointer = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(ALL_TOOLS))
graph.add_node("code_exec", code_exec_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", route_after_agent, {"tools": "tools", "code_exec": "code_exec", END: END})
graph.add_edge("tools", "chat_node")
graph.add_edge("code_exec", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)