import streamlit as st
from langgraph_database_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid
from langgraph_database_backend import retrive_all_threads


st.set_page_config(
    page_title="Chat Buddy",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    html, body, [class*="css"]  {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #1f2430 0%, #14161c 60%, #0e0f13 100%);
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #171a21 0%, #101217 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    section[data-testid="stSidebar"] h1 {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f5f5f7;
        padding-bottom: 0.2rem;
    }

    section[data-testid="stSidebar"] h2 {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b8f9a;
        margin-top: 1.2rem;
    }

    /* New chat button */
    section[data-testid="stSidebar"] .stButton:first-of-type button {
        background: linear-gradient(135deg, #6C63FF, #4B47D6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1rem;
        font-weight: 600;
        width: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 14px rgba(108,99,255,0.35);
    }
    section[data-testid="stSidebar"] .stButton:first-of-type button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 18px rgba(108,99,255,0.5);
    }

    /* Thread history buttons */
    section[data-testid="stSidebar"] .stButton button {
        background: rgba(255,255,255,0.04);
        color: #d7d8de;
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 8px;
        text-align: left;
        padding: 0.5rem 0.8rem;
        margin-bottom: 0.35rem;
        width: 100%;
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    section[data-testid="stSidebar"] .stButton button:hover {
        background: rgba(108,99,255,0.15);
        border-color: rgba(108,99,255,0.4);
        color: #ffffff;
    }

    /* ---------- Chat bubbles ---------- */
    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
        animation: fadeIn 0.25s ease-in-out;
    }

    /* user bubble */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #4B47D6, #6C63FF);
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) p,
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) * {
        color: #ffffff !important;
    }

    /* assistant bubble */
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
        background: #1e2129;
        border: 1px solid rgba(255,255,255,0.06);
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) p,
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) * {
        color: #e6e6ea !important;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to   { opacity: 1; transform: translateY(0); }
    }

    /* ---------- Chat input box ---------- */
    div[data-testid="stChatInput"] {
        background: #1a1c22;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    div[data-testid="stChatInput"] textarea {
        color: #f2f2f4 !important;
    }

    /* ---------- Main title spacing ---------- */
    .block-container {
        padding-top: 2rem;
        max-width: 800px;
    }

    /* ---------- Scrollbar polish ---------- */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.3); }
    </style>
    """,
    unsafe_allow_html=True,
)

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
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    messages = state.values.get('messages', [])
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
    st.session_state["thread_history"] = retrive_all_threads()

add_thread(st.session_state["thread_id"])
# ---- side bar -----
st.sidebar.title("💬 Data Buddy")
if st.sidebar.button("➕ New chat"):
    reset_chat()
st.sidebar.header("Message History")
for index, thread_id in enumerate(reversed(st.session_state["thread_history"])):
    if st.sidebar.button(f"🗂️ Chat {index + 1}", key=str(thread_id)):
        st.session_state["thread_id"] = thread_id
        st.session_state["message_history"] = load_conversation(thread_id)

# ------------- main ui -----------

st.title("Data Buddy")

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