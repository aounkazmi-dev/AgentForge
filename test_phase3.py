"""
Run this with: python test_phase3.py

This checks that the agent <-> tool loop actually works: the LLM should
call run_sql itself (we never write or run the SQL by hand here), and the
final answer should contain an actual computed number, not just SQL text.
"""

import uuid
from data_ingestion import ingest
from langgraph_database_backend_phase3 import chatbot, set_active_connection

sample_csv = """name,age,city,salary
Alice,29,New York,72000
Bob,34,Chicago,65000
Charlie,,Chicago,58000
Diana,41,New York,91000
Eve,25,Austin,50000
""".encode("utf-8")

df, con, schema_summary = ingest(sample_csv, "sample.csv")

# NEW: this is the extra step Phase 3 requires — point the SQL tool at
# this connection before asking anything that needs real computation.
set_active_connection(con)

thread_id = str(uuid.uuid4())
CONFIG = {"configurable": {"thread_id": thread_id}}

result = chatbot.invoke(
    {
        "messages": [{"role": "user", "content": "Which city has the highest average salary? Give me the exact number."}],
        "schema_summary": schema_summary,
    },
    config=CONFIG,
)

print("=== Full message trace (so you can see the tool call happen) ===")
for msg in result["messages"]:
    role = msg.type
    content = msg.content
    tool_calls = getattr(msg, "tool_calls", None)
    print(f"\n[{role}]")
    if tool_calls:
        print(f"  tool_calls: {tool_calls}")
    if content:
        print(f"  content: {content}")

print("\n=== Final answer only ===")
print(result["messages"][-1].content)