"""
Phase 1: Data Ingestion

Responsibilities:
- Accept an uploaded CSV/Excel file (as bytes or a file path)
- Load it into a pandas DataFrame
- Register that DataFrame as a table inside a DuckDB connection (so the SQL
  agent can later query it with plain SQL)
- Produce a compact "schema summary" string that we can hand to the LLM so it
  knows what columns/types/sample values exist without dumping the whole
  dataset into the prompt.

This module has NO LangGraph / LangChain / Streamlit dependency on purpose.
Keeping it standalone means we can unit-test it in isolation before wiring it
into the chatbot graph or the UI.
"""

import io
import duckdb
import pandas as pd


def load_dataframe(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """
    Load a CSV or Excel file (given as raw bytes + its original filename)
    into a pandas DataFrame.
    """
    buffer = io.BytesIO(file_bytes)

    if filename.lower().endswith(".csv"):
        df = pd.read_csv(buffer)
    elif filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(buffer)
    else:
        raise ValueError(f"Unsupported file type: {filename}. Use .csv, .xlsx, or .xls")

    # Basic cleanup: strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """
    Create an in-memory DuckDB connection.
    In-memory is fine here because the DataFrame is the source of truth;
    DuckDB just gives us a SQL interface over it.
    """
    return duckdb.connect(database=":memory:")


def register_dataframe(con: duckdb.DuckDBPyConnection, df: pd.DataFrame, table_name: str = "data") -> None:
    """
    Register a pandas DataFrame as a queryable DuckDB table.
    After this, SQL like `SELECT * FROM data LIMIT 5` works via `con.sql(...)`.
    """
    con.register(table_name, df)


def profile_dataframe(df: pd.DataFrame, table_name: str = "data", max_sample_rows: int = 3) -> str:
    """
    Build a compact, LLM-friendly text summary of the dataset:
    - table name
    - row/column counts
    - each column's dtype, null count, and a few sample values
    - a small sample of actual rows

    This is what gets injected into the system prompt so the LLM can reason
    about the data without us sending the entire dataset as tokens.
    """
    n_rows, n_cols = df.shape
    lines = [
        f"Table name: {table_name}",
        f"Rows: {n_rows} | Columns: {n_cols}",
        "",
        "Columns:",
    ]

    for col in df.columns:
        dtype = str(df[col].dtype)
        n_nulls = int(df[col].isna().sum())
        sample_vals = df[col].dropna().unique()[:3]
        sample_str = ", ".join(str(v) for v in sample_vals)
        lines.append(f"  - {col} ({dtype}, {n_nulls} nulls) — e.g. {sample_str}")

    lines.append("")
    lines.append(f"Sample rows (first {max_sample_rows}):")
    lines.append(df.head(max_sample_rows).to_string(index=False))

    return "\n".join(lines)


def ingest(file_bytes: bytes, filename: str, table_name: str = "data"):
    """
    Convenience wrapper that runs the full Phase 1 pipeline:
    load -> register in DuckDB -> profile.

    Returns:
        df       : pandas.DataFrame
        con      : duckdb.DuckDBPyConnection (with the df registered as a table)
        schema_summary : str, ready to drop into a system prompt
    """
    df = load_dataframe(file_bytes, filename)
    con = get_duckdb_connection()
    register_dataframe(con, df, table_name)
    schema_summary = profile_dataframe(df, table_name)
    return df, con, schema_summary


if __name__ == "__main__":
    # Quick manual test using a small synthetic CSV, so you can run
    # `python data_ingestion.py` and see it work end to end before
    # touching Streamlit or LangGraph at all.
    sample_csv = """name,age,city,salary
Alice,29,New York,72000
Bob,34,Chicago,65000
Charlie,,Chicago,58000
Diana,41,New York,91000
Eve,25,Austin,
""".encode("utf-8")

    df, con, schema_summary = ingest(sample_csv, "sample.csv")

    print("=== Schema Summary (this is what the LLM will see) ===")
    print(schema_summary)

    print("\n=== Test SQL query via DuckDB ===")
    result = con.sql("SELECT city, AVG(salary) AS avg_salary FROM data GROUP BY city").df()
    print(result)