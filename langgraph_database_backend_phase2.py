from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import sqlite3

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    # NEW: holds the schema_summary text produced by data_ingestion.ingest().
    # Not wrapped in Annotated/add_messages, so each graph invocation either
    # overwrites it (if you pass a new value in) or, if you don't pass it,
    # LangGraph keeps whatever was last checkpointed for this thread_id.
    schema_summary: str


def chat_node(state: ChatState):
    messages = state["messages"]
    schema_summary = state.get("schema_summary", "")

    if schema_summary:
        # NEW: prepend a system message so the LLM knows what dataset it's
        # working with. We build this fresh every call instead of storing it
        # inside `messages`, so it doesn't get duplicated across turns.
        system_prompt = (
            "You are a data analyst assistant. A dataset has been loaded "
            "and is available to you. Here is its schema:\n\n"
            f"{schema_summary}\n\n"
            "Answer the user's questions about this dataset as accurately as "
            "possible. If you don't have a tool to run SQL or Python yet, "
            "reason from the schema and sample rows shown above."
        )
        llm_input = [SystemMessage(content=system_prompt)] + messages
    else:
        # No dataset loaded yet — behave exactly like before.
        llm_input = messages

    response = llm.invoke(llm_input)
    return {"messages": [response]}


conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
# Checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


def retrive_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)