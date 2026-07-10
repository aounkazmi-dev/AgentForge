from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
import re
import io
import contextlib
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

load_dotenv()

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0,
    max_tokens=4096,
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    schema_summary: str


# ---------------------------------------------------------------------------
#  NEW: per-thread (per-user-session) storage, instead of shared globals
# ---------------------------------------------------------------------------
# Previously _active_df / _active_con / _last_figure were single module-level
# variables — fine for one person testing locally, but on a deployed app
# with more than one visitor at a time, User A's dataset could silently get
# overwritten by User B's upload, and each would see the wrong tool results.
#
# The fix: keep a dict keyed by thread_id (already unique per browser session
# in this app), and have tools automatically receive the current thread_id
# via RunnableConfig — LangChain/LangGraph auto-injects a parameter typed
# `config: RunnableConfig` into a tool call without the LLM ever seeing or
# needing to fill it in, since it's not part of the tool's exposed schema.
_session_dataframes: dict = {}
_session_connections: dict = {}
_session_figures: dict = {}


def _thread_id_from_config(config: RunnableConfig) -> str:
    """Pull the current thread_id out of the run config, with a fallback
    so nothing crashes if a tool is ever called without one."""
    if not config:
        return "default"
    return config.get("configurable", {}).get("thread_id", "default")


def set_active_dataframe(df, thread_id: str):
    """Call this once per upload (with the CURRENT session's thread_id),
    right after data_ingestion.ingest() runs."""
    _session_dataframes[thread_id] = df


def set_active_connection(con, thread_id: str):
    """Call this once per upload (with the CURRENT session's thread_id),
    right after data_ingestion.ingest() runs."""
    _session_connections[thread_id] = con


def get_last_figure(thread_id: str):
    """Frontend calls this after invoking the graph to get the most recently
    created chart for THIS thread, or None if no chart was created."""
    return _session_figures.get(thread_id)


def clear_last_figure(thread_id: str):
    """Frontend calls this before each new user turn, so an old chart from
    a previous question doesn't get shown again by mistake."""
    _session_figures[thread_id] = None


# ---------------------------------------------------------------------------
#  SQL tool
# ---------------------------------------------------------------------------
@tool
def run_sql(query: str, config: RunnableConfig) -> str:
    """Run a read-only SQL query against the 'data' table (DuckDB syntax)
    and return the result as text. Use this whenever the user's question
    requires aggregation, filtering, sorting, or computing something from
    the dataset rather than just describing its structure. The table is
    always called 'data'. Example: SELECT city, AVG(salary) AS avg_salary
    FROM data GROUP BY city ORDER BY avg_salary DESC LIMIT 1."""
    thread_id = _thread_id_from_config(config)
    con = _session_connections.get(thread_id)
    if con is None:
        return "No dataset is currently loaded. Ask the user to upload a file first."
    try:
        result_df = con.sql(query).df()
        if result_df.empty:
            return "The query ran successfully but returned no rows."
        return result_df.to_string(index=False)
    except Exception as e:
        return f"SQL error: {e}. Check the column names against the schema and try again."


# ---------------------------------------------------------------------------
#  Structured stats tools
# ---------------------------------------------------------------------------
@tool
def compute_correlation(column_a: str, column_b: str, config: RunnableConfig) -> str:
    """Compute the Pearson correlation coefficient between two numeric
    columns in the dataset. Use this when the user asks whether two
    columns are related, correlated, or move together."""
    thread_id = _thread_id_from_config(config)
    df = _session_dataframes.get(thread_id)
    if df is None:
        return "No dataset is currently loaded."
    try:
        corr = df[column_a].corr(df[column_b])
        return f"Correlation coefficient between '{column_a}' and '{column_b}': {corr:.4f}"
    except Exception as e:
        return f"Error computing correlation: {e}. Check that both column names exist and are numeric."


@tool
def describe_column(column: str, config: RunnableConfig) -> str:
    """Get summary statistics for a single column: count, mean, std, min,
    max, and quartiles for numeric columns, or value counts for
    categorical/text columns. Use this for 'describe', 'summarize', or
    'distribution of' style questions about one column."""
    thread_id = _thread_id_from_config(config)
    df = _session_dataframes.get(thread_id)
    if df is None:
        return "No dataset is currently loaded."
    try:
        if column not in df.columns:
            return f"Column '{column}' not found. Available columns: {list(df.columns)}"
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            return series.describe().to_string()
        else:
            return series.value_counts().to_string()
    except Exception as e:
        return f"Error describing column: {e}"


@tool
def run_ttest(numeric_column: str, group_column: str, config: RunnableConfig) -> str:
    """Run an independent two-sample t-test comparing a numeric column
    across exactly two groups defined by a categorical column. Use this
    when the user asks if there's a statistically significant difference
    between two groups (e.g. 'is salary different between two cities?').
    The group_column must have exactly two unique values for this to work."""
    thread_id = _thread_id_from_config(config)
    df = _session_dataframes.get(thread_id)
    if df is None:
        return "No dataset is currently loaded."
    try:
        groups = df[group_column].dropna().unique()
        if len(groups) != 2:
            return f"'{group_column}' has {len(groups)} unique groups, but a t-test needs exactly 2. Groups found: {list(groups)}"
        sample_a = df[df[group_column] == groups[0]][numeric_column].dropna()
        sample_b = df[df[group_column] == groups[1]][numeric_column].dropna()
        t_stat, p_value = stats.ttest_ind(sample_a, sample_b)
        return (
            f"T-test comparing '{numeric_column}' between {group_column}='{groups[0]}' "
            f"and {group_column}='{groups[1]}': t-statistic={t_stat:.4f}, p-value={p_value:.4f}. "
            f"{'Statistically significant (p < 0.05)' if p_value < 0.05 else 'Not statistically significant (p >= 0.05)'}."
        )
    except Exception as e:
        return f"Error running t-test: {e}"


# ---------------------------------------------------------------------------
#  Visualization tool
# ---------------------------------------------------------------------------
@tool
def create_chart(chart_type: str, x_column: str, config: RunnableConfig, y_column: str = "", agg: str = "", color_column: str = "") -> str:
    """Create a chart to visualize the dataset. chart_type must be one of:
    'bar', 'line', 'scatter', 'histogram', 'box'. x_column is required.
    y_column is required for bar/line/scatter/box, not used for histogram.
    agg (optional) is one of 'mean', 'sum', 'count', 'median' — use this for
    bar/line charts when the user wants an aggregate per category (e.g.
    'average salary by city' -> chart_type='bar', x_column='city',
    y_column='salary', agg='mean'). color_column (optional) splits the
    chart into colored groups by another column."""
    thread_id = _thread_id_from_config(config)
    df = _session_dataframes.get(thread_id)
    if df is None:
        return "No dataset is currently loaded."

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

        _session_figures[thread_id] = fig.to_json()

        desc = f"Created a {chart_type} chart, x={x_column}"
        if y_column:
            desc += f", y={y_column}"
        if agg:
            desc += f", aggregated by {agg}"
        desc += ". The chart is ready to display to the user."
        return desc
    except Exception as e:
        return f"Error creating chart: {e}. Check that the column names exist in the dataset."


# ---------------------------------------------------------------------------
#  Insight/Report tool
# ---------------------------------------------------------------------------
@tool
def generate_data_report(config: RunnableConfig) -> str:
    """Generate a comprehensive overview of the entire dataset: descriptive
    statistics for all numeric columns, a missing-value summary, the
    strongest correlations between numeric columns, and category breakdowns
    for text/categorical columns. Use this when the user asks for a
    'report', 'summary', 'overview', 'key insights', or to 'analyze the
    whole dataset' — as opposed to a question about one specific column
    or relationship, which the other tools handle better."""
    thread_id = _thread_id_from_config(config)
    df = _session_dataframes.get(thread_id)
    if df is None:
        return "No dataset is currently loaded."

    lines = [f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns", ""]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    if numeric_cols:
        lines.append("=== Numeric column summary ===")
        lines.append(df[numeric_cols].describe().to_string())
        lines.append("")

    null_counts = df.isnull().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        lines.append("=== Missing values ===")
        lines.append(null_counts.to_string())
        lines.append("")
    else:
        lines.append("=== Missing values ===")
        lines.append("None — the dataset has no missing values.")
        lines.append("")

    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        seen_pairs = set()
        scored_pairs = []
        for col_a in numeric_cols:
            for col_b in numeric_cols:
                if col_a == col_b:
                    continue
                pair_key = frozenset([col_a, col_b])
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                corr_value = corr_matrix.loc[col_a, col_b]
                scored_pairs.append((col_a, col_b, corr_value))
        scored_pairs.sort(key=lambda item: abs(item[2]), reverse=True)

        if scored_pairs:
            lines.append("=== Strongest correlations ===")
            for col_a, col_b, corr_value in scored_pairs[:3]:
                lines.append(f"{col_a} vs {col_b}: {corr_value:.3f}")
            lines.append("")

    if cat_cols:
        lines.append("=== Categorical column breakdowns (top values) ===")
        for col in cat_cols[:5]:
            lines.append(f"-- {col} --")
            lines.append(df[col].value_counts().head(5).to_string())
        lines.append("")

    return "\n".join(lines)


llm_with_tools = llm.bind_tools([run_sql, compute_correlation, describe_column, run_ttest, create_chart, generate_data_report])


# ---------------------------------------------------------------------------
#  Text-based code fallback
# ---------------------------------------------------------------------------
_SAFE_BUILTINS = {
    "print": print, "len": len, "range": range, "sum": sum, "min": min,
    "max": max, "sorted": sorted, "abs": abs, "round": round, "list": list,
    "dict": dict, "set": set, "tuple": tuple, "enumerate": enumerate,
    "zip": zip, "str": str, "int": int, "float": float, "bool": bool,
    "type": type, "isinstance": isinstance,
}


def get_text_content(content) -> str:
    """Normalizes content that might be a plain string or a list of
    {'type': 'text', 'text': ...} blocks (a shape some providers use)
    into a plain string."""
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


def code_exec_node(state: ChatState, config: RunnableConfig):
    """Runs whatever python code block the LLM just wrote in plain text,
    using THIS thread's dataframe, then feeds the printed output back in
    as a new message so chat_node can turn it into a final answer."""
    thread_id = _thread_id_from_config(config)
    df = _session_dataframes.get(thread_id)

    last_msg = state["messages"][-1]
    text = get_text_content(last_msg.content)
    code = extract_python_code(text)

    if not code:
        return {"messages": [HumanMessage(content="[Code execution] No code block found to run.")]}

    safe_globals = {
        "__builtins__": _SAFE_BUILTINS,
        "df": df,
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

        maybe_fig = safe_globals.get("fig")
        if isinstance(maybe_fig, go.Figure):
            _session_figures[thread_id] = maybe_fig.to_json()
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
            "pandas DataFrame `df`, plus `pd`, `np`, `stats` (scipy.stats), "
            "and `px`/`go` (plotly.express / plotly.graph_objects) which are "
            "all available. Use print() for anything you want to see back. "
            "If you create a chart this way, assign it to a variable named "
            "exactly `fig` and it will be displayed automatically. Only do "
            "this when the 6 tools above genuinely don't cover the "
            "question — prefer the structured tools whenever they fit.\n\n"
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


# NEW: in-memory checkpointer instead of a SQLite file. Conversation state
# now only lives as long as the server process is running — a browser
# refresh gets a brand new thread_id (see the frontend), so there's nothing
# stale to accidentally reload and error out on. A real server restart wipes
# everything cleanly too, since there's no file left behind to reopen.
checkpointer = MemorySaver()

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode([run_sql, compute_correlation, describe_column, run_ttest, create_chart, generate_data_report]))
graph.add_node("code_exec", code_exec_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", route_after_agent, {"tools": "tools", "code_exec": "code_exec", END: END})
graph.add_edge("tools", "chat_node")
graph.add_edge("code_exec", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)