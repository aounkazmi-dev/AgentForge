"""
Run this with: python test_phase4_codefallback.py

Asks a question none of the 3 structured tools (compute_correlation,
describe_column, run_ttest) can directly answer, to confirm the LLM
falls back to writing a python code block, and that we correctly detect
and execute it.
"""

import uuid
from data_ingestion import ingest
from langgraph_database_backend_phase4 import chatbot, set_active_connection, set_active_dataframe

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

thread_id = str(uuid.uuid4())
CONFIG = {"configurable": {"thread_id": thread_id}}

# NEW: this specifically asks for a coefficient of variation (std / mean),
# which none of our 4 tools compute directly — this should force the
# code-block fallback path.
result = chatbot.invoke(
    {
        "messages": [{"role": "user", "content": "What is the coefficient of variation (standard deviation divided by mean) for the salary column?"}],
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