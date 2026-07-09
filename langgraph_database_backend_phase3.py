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

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    schema_summary: str


# ---------------------------------------------------------------------------
#  NEW: the SQL tool
# ---------------------------------------------------------------------------
# LangChain tools are plain functions — the LLM only ever sees the docstring
# and the type hints, never the function body. It reads the docstring to
# decide WHEN to call this tool and WHAT argument (query) to pass.
#
# The tricky bit: `run_sql` needs the DuckDB connection created in Phase 1's
# `ingest()`, but LangChain tools can't take extra arguments beyond what the
# LLM provides. For a single-user local app, the simplest fix is a
# module-level variable that gets set once, right after a file is uploaded.
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


# Bind the tool to the LLM so it can choose to call it
llm_with_tools = llm.bind_tools([run_sql])


# ---------------------------------------------------------------------------
#  chat_node — now calls llm_with_tools instead of llm directly
# ---------------------------------------------------------------------------
def chat_node(state: ChatState):
    messages = state["messages"]
    schema_summary = state.get("schema_summary", "")

    if schema_summary:
        system_prompt = (
            "You are a data analyst assistant. A dataset has been loaded "
            "and is available to you as a DuckDB table called 'data'. "
            "Here is its schema:\n\n"
            f"{schema_summary}\n\n"
            "When the user asks something that requires computing, filtering, "
            "sorting, or aggregating the data, use the run_sql tool instead of "
            "guessing the answer yourself. When the user just asks about the "
            "structure of the data, you can answer directly from the schema above."
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
# NEW: the tool-executing node. ToolNode is a LangGraph prebuilt that knows
# how to read a tool_call off the last AIMessage, run the matching tool
# function, and package the result back as a ToolMessage.
graph.add_node("tools", ToolNode([run_sql]))

graph.add_edge(START, "chat_node")
# NEW: conditional edge. tools_condition looks at the last message —
# if it contains a tool call, route to "tools"; otherwise route to END.
graph.add_conditional_edges("chat_node", tools_condition, {"tools": "tools", END: END})
# NEW: after the tool runs, loop back to chat_node so the LLM can read the
# tool's result and either call another tool or give a final answer.
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


def retrive_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)