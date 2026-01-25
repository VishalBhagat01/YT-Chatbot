import streamlit as st
from app import build_rag_chain

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    layout="centered",
)

st.title("YouTube RAG Chatbot")
st.markdown("Ask questions about any YouTube video using Gemini")

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("Configuration")
    video_id = st.text_input("YouTube Video ID", value="Gfr50f6ZBvo")
    k_results = st.slider("Retrieved Chunks (k)", 1, 8, 4)
    temperature = st.slider("Model Temperature", 0.0, 1.0, 0.2)
    rebuild_index = st.checkbox("Force rebuild knowledge base")

    build_button = st.button("Build Knowledge Base")

# -------------------------------
# Session State
# -------------------------------
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------------
# Build Pipeline
# -------------------------------
if build_button:
    with st.spinner("Building knowledge base..."):
        try:
            st.session_state.rag_chain = build_rag_chain(
                video_id=video_id,
                k_results=k_results,
                temperature=temperature,
                rebuild=rebuild_index,
            )
            st.session_state.chat_history = []
            st.success("Knowledge base ready. Start chatting!")
        except Exception as e:
            st.error(str(e))

# -------------------------------
# Chat UI
# -------------------------------
if st.session_state.rag_chain:
    st.subheader("Chat")

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Ask a question about the video")
        send_button = st.form_submit_button("Send")

    if send_button and user_input:
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.rag_chain.invoke(user_input)

                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("Bot", response))
            except Exception as e:
                st.error(str(e))

    for role, message in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**You:** {message}")
        else:
            st.markdown(f"**Bot:** {message}")

else:
    st.info("Enter a video ID and click 'Build Knowledge Base' to begin.")
