"""
All the data-analyst tools the agent can call, plus the per-thread storage
they read from. Kept separate from backend.py so the graph-wiring file
doesn't have to also contain 250+ lines of tool implementations.
"""

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
import pandas as pd
import numpy as np
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go


# ---------------------------------------------------------------------------
#  Per-thread (per-user-session) storage
# ---------------------------------------------------------------------------
# Keyed by thread_id instead of being single shared variables, so two people
# using the deployed app at once don't see each other's dataset.
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


def set_last_figure(fig_json: str, thread_id: str):
    """Used by backend.py's code_exec_node to register a chart produced by
    the text-based code fallback, the same way create_chart does below."""
    _session_figures[thread_id] = fig_json


def get_active_dataframe(thread_id: str):
    """Used by backend.py's code_exec_node to hand the current thread's
    DataFrame into the sandboxed exec() environment."""
    return _session_dataframes.get(thread_id)


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


# Convenience list — backend.py binds/registers tools using this, instead of
# needing to name all six individually in multiple places.
ALL_TOOLS = [run_sql, compute_correlation, describe_column, run_ttest, create_chart, generate_data_report]