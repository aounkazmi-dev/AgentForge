"""
Quick manual test: does the chatbot actually 'know' about the dataset now?

Run this with: python test_phase2.py
(needs your .env with the Gemini API key, same as your existing project)
"""

import uuid
from data_ingestion import ingest
from langgraph_database_backend_phase2 import chatbot

# 1. Fake a file upload, same as Phase 1 test
sample_csv = """name,age,city,salary
Alice,29,New York,72000
Bob,34,Chicago,65000
Charlie,,Chicago,58000
Diana,41,New York,91000
Eve,25,Austin,
""".encode("utf-8")

df, con, schema_summary = ingest(sample_csv, "sample.csv")

# 2. Start a fresh thread, same pattern as your frontend uses
thread_id = str(uuid.uuid4())
CONFIG = {"configurable": {"thread_id": thread_id}}

# 3. First call: pass BOTH messages and schema_summary.
#    schema_summary only needs to be passed once — it gets checkpointed
#    against this thread_id and will still be there on later calls.
result = chatbot.invoke(
    {
        "messages": [{"role": "user", "content": "What columns does this dataset have, and which column has missing values?"}],
        "schema_summary": schema_summary,
    },
    config=CONFIG,
)
print("=== Answer 1 ===")
print(result["messages"][-1].content)

# 4. Second call on the SAME thread: only pass messages.
#    If schema-awareness is working, the LLM should still know the schema
#    even though we didn't pass schema_summary again.
result2 = chatbot.invoke(
    {"messages": [{"role": "user", "content": "Based on that, which city has the highest average salary?"}]},
    config=CONFIG,
)
print("\n=== Answer 2 (schema_summary NOT re-sent, should still know the data) ===")
print(result2["messages"][-1].content)