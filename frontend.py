import streamlit as st
from backend import chatbot
from tools import set_active_connection, set_active_dataframe, get_last_figure, clear_last_figure
from utils import get_text_content
from langchain_core.messages import HumanMessage
from data_ingestion import ingest
import plotly.io as pio
import uuid


st.set_page_config(
    page_title="Data Buddy",
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

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) {
        background: linear-gradient(135deg, #4B47D6, #6C63FF);
    }
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) p,
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) * {
        color: #ffffff !important;
    }

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

    div[data-testid="stChatInput"] {
        background: #1a1c22;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    div[data-testid="stChatInput"] textarea {
        color: #f2f2f4 !important;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 800px;
    }

    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.3); }
    </style>
    """,
    unsafe_allow_html=True,
)

#    ---------helpers
def generate_thread_id():

    return str(uuid.uuid4())

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
        result.append({"role": role, "content": get_text_content(msg.content)})
    return result


# ----- session things ---
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
if "thread_history" not in st.session_state:
    st.session_state["thread_history"] = []

if "df" not in st.session_state:
    st.session_state["df"] = None
if "con" not in st.session_state:
    st.session_state["con"] = None
if "schema_summary" not in st.session_state:
    st.session_state["schema_summary"] = ""
if "uploaded_file_id" not in st.session_state:
    st.session_state["uploaded_file_id"] = None

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

# NEW: file uploader — this is the entry point for Phase 1's ingest()
uploaded_file = st.file_uploader("Upload a CSV or Excel file to analyze", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # NEW: only re-ingest if this is actually a different/new file, not just
    # a Streamlit rerun of the same one (reruns happen on every interaction)
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    if st.session_state["uploaded_file_id"] != file_id:
        try:
            file_bytes = uploaded_file.read()
            df, con, schema_summary = ingest(file_bytes, uploaded_file.name)
            st.session_state["df"] = df
            st.session_state["con"] = con
            st.session_state["schema_summary"] = schema_summary
            st.session_state["uploaded_file_id"] = file_id
            st.success(f"Loaded {uploaded_file.name} — {df.shape[0]} rows, {df.shape[1]} columns.")
        except Exception as e:

            st.error(f"Couldn't read this file: {e}. Try a different CSV/Excel file.")

    # NEW: small preview so the user can see what's loaded
    with st.expander("Preview data"):
        st.dataframe(st.session_state["df"].head(10))

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("figure"):
            fig = pio.from_json(message["figure"])
            st.plotly_chart(fig, use_container_width=True)


user_input = st.chat_input("Type here")

if user_input:

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
        st.write(user_input)

    current_thread_id = str(st.session_state["thread_id"])
    if st.session_state["df"] is not None:
        set_active_dataframe(st.session_state["df"], current_thread_id)
    if st.session_state["con"] is not None:
        set_active_connection(st.session_state["con"], current_thread_id)
    clear_last_figure(current_thread_id)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            graph_input = {"messages": [HumanMessage(content=user_input)]}
            if st.session_state["schema_summary"]:
                graph_input["schema_summary"] = st.session_state["schema_summary"]

            result = chatbot.invoke(graph_input, config=CONFIG)

        ai_content = get_text_content(result["messages"][-1].content)
        st.write(ai_content)

        fig_json = get_last_figure(current_thread_id)
        if fig_json:
            fig = pio.from_json(fig_json)
            st.plotly_chart(fig, use_container_width=True)

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": ai_content,
            "figure": fig_json,  # None if no chart was created this turn
        }
    )