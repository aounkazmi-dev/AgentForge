from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import sqlite3
import re
import io
import contextlib
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

load_dotenv()

# NEW: max_output_tokens raised from the default (~1024). gemini-2.5-flash
# spends part of its output budget on internal "thinking" before writing
# the visible answer or a tool call — with a small budget, it can burn
# all of it thinking and return an empty message, which is what you saw.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_output_tokens=4096,
    temperature=0,
    thinking_budget=0,  # NEW: try disabling internal "thinking" — if this
                        # line throws a TypeError (unsupported kwarg on your
                        # installed langchain-google-genai version), just
                        # delete this line and tell me the version you have.
)


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
#  Structured stats tools (replaces free-form run_python)
# ---------------------------------------------------------------------------
# NEW APPROACH: instead of asking the LLM to write arbitrary Python code as
# a string argument (which reliably triggers Gemini's MALFORMED_FUNCTION_CALL
# for code-like content), each statistical operation is its own tool with
# plain, simple arguments — just column names and operation names. This is
# both more reliable for function-calling AND safer (no exec() of arbitrary
# code at all for these common operations).
_active_df = None


def set_active_dataframe(df):
    """Call this once, right after data_ingestion.ingest() runs, so the
    stats tools below have a DataFrame to analyze."""
    global _active_df
    _active_df = df


@tool
def compute_correlation(column_a: str, column_b: str) -> str:
    """Compute the Pearson correlation coefficient between two numeric
    columns in the dataset. Use this when the user asks whether two
    columns are related, correlated, or move together."""
    if _active_df is None:
        return "No dataset is currently loaded."
    try:
        corr = _active_df[column_a].corr(_active_df[column_b])
        return f"Correlation coefficient between '{column_a}' and '{column_b}': {corr:.4f}"
    except Exception as e:
        return f"Error computing correlation: {e}. Check that both column names exist and are numeric."


@tool
def describe_column(column: str) -> str:
    """Get summary statistics for a single column: count, mean, std, min,
    max, and quartiles for numeric columns, or value counts for
    categorical/text columns. Use this for 'describe', 'summarize', or
    'distribution of' style questions about one column."""
    if _active_df is None:
        return "No dataset is currently loaded."
    try:
        if column not in _active_df.columns:
            return f"Column '{column}' not found. Available columns: {list(_active_df.columns)}"
        series = _active_df[column]
        if pd.api.types.is_numeric_dtype(series):
            return series.describe().to_string()
        else:
            return series.value_counts().to_string()
    except Exception as e:
        return f"Error describing column: {e}"


@tool
def run_ttest(numeric_column: str, group_column: str) -> str:
    """Run an independent two-sample t-test comparing a numeric column
    across exactly two groups defined by a categorical column. Use this
    when the user asks if there's a statistically significant difference
    between two groups (e.g. 'is salary different between two cities?').
    The group_column must have exactly two unique values for this to work."""
    if _active_df is None:
        return "No dataset is currently loaded."
    try:
        groups = _active_df[group_column].dropna().unique()
        if len(groups) != 2:
            return f"'{group_column}' has {len(groups)} unique groups, but a t-test needs exactly 2. Groups found: {list(groups)}"
        sample_a = _active_df[_active_df[group_column] == groups[0]][numeric_column].dropna()
        sample_b = _active_df[_active_df[group_column] == groups[1]][numeric_column].dropna()
        t_stat, p_value = stats.ttest_ind(sample_a, sample_b)
        return (
            f"T-test comparing '{numeric_column}' between {group_column}='{groups[0]}' "
            f"and {group_column}='{groups[1]}': t-statistic={t_stat:.4f}, p-value={p_value:.4f}. "
            f"{'Statistically significant (p < 0.05)' if p_value < 0.05 else 'Not statistically significant (p >= 0.05)'}."
        )
    except Exception as e:
        return f"Error running t-test: {e}"


# ---------------------------------------------------------------------------
#  NEW: Visualization tool
# ---------------------------------------------------------------------------
# A chart is different from everything before it: run_sql/compute_correlation
# etc. all return TEXT the LLM reads and talks about. A chart is an OBJECT —
# the LLM can't "see" a Plotly figure, and printing it as text is useless.
# So this tool does two things: (1) gives the LLM a short text confirmation
# to talk about, and (2) stashes the actual figure (as JSON) in a
# module-level variable, completely separate from the message history, for
# the frontend to pick up and render later (that wiring happens in Phase 8).
_last_figure = None


def get_last_figure():
    """Frontend calls this after invoking the graph to get the most recently
    created chart (as a Plotly-compatible JSON string), or None if no chart
    was created on that turn."""
    return _last_figure


def clear_last_figure():
    """Frontend calls this before each new user turn, so an old chart from
    a previous question doesn't get shown again by mistake."""
    global _last_figure
    _last_figure = None


@tool
def create_chart(chart_type: str, x_column: str, y_column: str = "", agg: str = "", color_column: str = "") -> str:
    """Create a chart to visualize the dataset. chart_type must be one of:
    'bar', 'line', 'scatter', 'histogram', 'box'. x_column is required.
    y_column is required for bar/line/scatter/box, not used for histogram.
    agg (optional) is one of 'mean', 'sum', 'count', 'median' — use this for
    bar/line charts when the user wants an aggregate per category (e.g.
    'average salary by city' -> chart_type='bar', x_column='city',
    y_column='salary', agg='mean'). color_column (optional) splits the
    chart into colored groups by another column."""
    if _active_df is None:
        return "No dataset is currently loaded."

    df = _active_df
    color = color_column or None
    try:
        if agg and y_column and chart_type in ("bar", "line"):
            plot_df = df.groupby(x_column, as_index=False)[y_column].agg(agg)
        else:
            plot_df = df

        if chart_type == "bar":
            fig = px.bar(plot_df, x=x_column, y=y_column, color=color)
        elif chart_type == "line":
            fig = px.line(plot_df, x=x_column, y=y_column, color=color)
        elif chart_type == "scatter":
            fig = px.scatter(plot_df, x=x_column, y=y_column, color=color)
        elif chart_type == "histogram":
            fig = px.histogram(plot_df, x=x_column, color=color)
        elif chart_type == "box":
            fig = px.box(plot_df, x=x_column, y=y_column, color=color)
        else:
            return f"Unknown chart_type '{chart_type}'. Use one of: bar, line, scatter, histogram, box."

        global _last_figure
        _last_figure = fig.to_json()

        desc = f"Created a {chart_type} chart, x={x_column}"
        if y_column:
            desc += f", y={y_column}"
        if agg:
            desc += f", aggregated by {agg}"
        desc += ". The chart is ready to display to the user."
        return desc
    except Exception as e:
        return f"Error creating chart: {e}. Check that the column names exist in the dataset."


# Bind all tools to the LLM
llm_with_tools = llm.bind_tools([run_sql, compute_correlation, describe_column, run_ttest, create_chart])


# ---------------------------------------------------------------------------
#  NEW: text-based code fallback (the answer to "endless possibilities")
# ---------------------------------------------------------------------------
# Instead of asking the LLM to pass code AS A TOOL ARGUMENT (which goes
# through Gemini's strict function-call JSON encoding and reliably breaks —
# that's what MALFORMED_FUNCTION_CALL was), we let the LLM write a python
# code block in its normal reply text when none of the structured tools fit.
# We detect that block ourselves and execute it. No function-calling schema
# is involved for the code at all, so the encoding bug never triggers.

_SAFE_BUILTINS = {
    "print": print, "len": len, "range": range, "sum": sum, "min": min,
    "max": max, "sorted": sorted, "abs": abs, "round": round, "list": list,
    "dict": dict, "set": set, "tuple": tuple, "enumerate": enumerate,
    "zip": zip, "str": str, "int": int, "float": float, "bool": bool,
    "type": type, "isinstance": isinstance,
}


def get_text_content(content) -> str:
    """Gemini sometimes returns content as a string, sometimes as a list of
    {'type': 'text', 'text': ...} blocks (you saw this back in Phase 3's
    output). This normalizes either shape into a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content) if content else ""


def extract_python_code(text: str):
    """Look for a ```python ... ``` fenced block in the LLM's reply text."""
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if not match:
        match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else None


def code_exec_node(state: ChatState):
    """Runs whatever python code block the LLM just wrote in plain text,
    then feeds the printed output back in as a new message so chat_node
    can turn it into a final answer on the next loop."""
    last_msg = state["messages"][-1]
    text = get_text_content(last_msg.content)
    code = extract_python_code(text)

    if not code:
        # Shouldn't happen given our routing, but fail safe rather than crash.
        return {"messages": [HumanMessage(content="[Code execution] No code block found to run.")]}

    safe_globals = {
        "__builtins__": _SAFE_BUILTINS,
        "df": _active_df,
        "pd": pd,
        "np": np,
        "stats": stats,
        "px": px,
        "go": go,
    }
    output_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(output_buffer):
            exec(code, safe_globals)
        output = output_buffer.getvalue().strip() or "Code ran successfully but printed nothing."

        # NEW: if the code created a variable called `fig` that's a real
        # Plotly figure, capture it the same way create_chart does — this
        # covers custom charts too unusual for the structured tool.
        maybe_fig = safe_globals.get("fig")
        if isinstance(maybe_fig, go.Figure):
            global _last_figure
            _last_figure = maybe_fig.to_json()
            output += "\n\nA chart was created and is ready to display to the user."
    except Exception as e:
        output = f"Error running the code: {e}"

    # NEW: fed back as a HumanMessage (not a ToolMessage, since this isn't a
    # real LangChain tool call) so chat_node's next invoke sees it as new
    # information to respond to.
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
            "scatter, histogram, or box chart.\n\n"
            "If none of these tools fit what's being asked, you may instead "
            "write a single python code block (```python ... ```) using the "
            "pandas DataFrame `df`, plus `pd`, `np`, `stats` (scipy.stats), "
            "and `px`/`go` (plotly.express / plotly.graph_objects) which are "
            "all available. Use print() for anything you want to see back. "
            "If you create a chart this way, assign it to a variable named "
            "exactly `fig` and it will be displayed automatically. Only do "
            "this when the 5 tools above genuinely don't cover the "
            "question — prefer the structured tools whenever they fit.\n\n"
            f"Schema:\n{schema_summary}\n\n"
            "When the user just asks about the structure of the data, you "
            "can answer directly from the schema above without any tool."
        )
        llm_input = [SystemMessage(content=system_prompt)] + messages
    else:
        llm_input = messages

    # NEW: Gemini occasionally returns finish_reason == MALFORMED_FUNCTION_CALL
    # for tool calls with code-like string arguments (quotes/newlines confuse
    # its own function-call encoding). This is usually transient, so we just
    # retry the same request a couple of times before giving up.
    max_attempts = 3
    response = None
    for attempt in range(max_attempts):
        response = llm_with_tools.invoke(llm_input)
        finish_reason = getattr(response, "response_metadata", {}).get("finish_reason")
        if finish_reason != "MALFORMED_FUNCTION_CALL":
            break
        print(f"  [retry] MALFORMED_FUNCTION_CALL on attempt {attempt + 1}/{max_attempts}, retrying...")

    return {"messages": [response]}


conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode([run_sql, compute_correlation, describe_column, run_ttest, create_chart]))
# NEW: the text-based code fallback node
graph.add_node("code_exec", code_exec_node)

graph.add_edge(START, "chat_node")
# NEW: route_after_agent checks for a real tool call OR a code block in the text
graph.add_conditional_edges("chat_node", route_after_agent, {"tools": "tools", "code_exec": "code_exec", END: END})
graph.add_edge("tools", "chat_node")
graph.add_edge("code_exec", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


def retrive_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)