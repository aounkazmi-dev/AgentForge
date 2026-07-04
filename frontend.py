import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

#    ---------helpers
def generate_thread_id():
    return uuid.uuid4()

def reset_chat():
    st.session_state["message_history"] = []
    st.session_state["thread_id"] = generate_thread_id()
    add_thread(st.session_state["thread_id"])

def add_thread(thread_id):
    if thread_id not in st.session_state["thread_history"]:
        st.session_state["thread_history"].append(thread_id)

def load_conversation(thread_id):
    messages = chatbot.get_state(config={'configurable': {'thread_id': thread_id}}).values['messages']
    result = []
    for msg in messages:
        role = "user" if msg.type == "human" else "assistant"
        result.append({"role": role, "content": msg.content})
    return result


# ----- session things ---
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
if "thread_history" not in st.session_state:
    st.session_state["thread_history"] = []

add_thread(st.session_state["thread_id"])
# ---- side bar -----
st.sidebar.title("LangGraph Chatbot")
if st.sidebar.button("New chat"):
    reset_chat()
st.sidebar.header("Message History")
for index, thread_id in enumerate(reversed(st.session_state["thread_history"])):
    if st.sidebar.button(f"Chat {index + 1}", key=str(thread_id)):
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = load_conversation(thread_id)

# ------------- main ui -----------


for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])



user_input = st.chat_input("Type here")

if user_input:

    # first add the message to message_history
    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input
        }
    )

    CONFIG = {
    "configurable": {
        "thread_id": st.session_state["thread_id"]
    }
}

    with st.chat_message("user"):
        st.text(user_input)

    with st.chat_message("assistant"):

        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },
                config=CONFIG,
                stream_mode="messages"
            )
        )

   
    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_message
        }
    )