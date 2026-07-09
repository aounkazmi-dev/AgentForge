"""
Run this with: python test_phase5.py

Checks that asking for a chart triggers the create_chart tool, and that
get_last_figure() returns a valid Plotly figure JSON afterward.
"""

import uuid
from data_ingestion import ingest
from langgraph_database_backend_phase5 import (
    chatbot, set_active_connection, set_active_dataframe,
    get_last_figure, clear_last_figure,
)

sample_csv = """name,age,city,salary
Alice,29,New York,72000
Bob,34,Chicago,65000
Charlie,31,Chicago,58000
Diana,41,New York,91000
Eve,25,Austin,50000
Frank,38,Austin,68000
Grace,45,New York,95000
Henry,28,Chicago,61000
""".encode("utf-8")

df, con, schema_summary = ingest(sample_csv, "sample.csv")
set_active_connection(con)
set_active_dataframe(df)
clear_last_figure()

thread_id = str(uuid.uuid4())
CONFIG = {"configurable": {"thread_id": thread_id}}

result = chatbot.invoke(
    {
        "messages": [{"role": "user", "content": "Show me a bar chart of average salary by city."}],
        "schema_summary": schema_summary,
    },
    config=CONFIG,
)

print("=== Full message trace ===")
for msg in result["messages"]:
    role = msg.type
    content = msg.content
    tool_calls = getattr(msg, "tool_calls", None)
    print(f"\n[{role}]")
    if tool_calls:
        print(f"  tool_calls: {tool_calls}")
    if content:
        print(f"  content: {content}")

print("\n=== Checking get_last_figure() ===")
fig_json = get_last_figure()
if fig_json:
    print(f"Got a figure! JSON length: {len(fig_json)} characters")
    print(f"First 200 chars: {fig_json[:200]}")
else:
    print("No figure was captured — something didn't fire correctly.")