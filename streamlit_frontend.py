import traceback
import streamlit as st

# Import build_rag_chain but capture import errors so the UI can show them
build_rag_chain = None
build_import_error = None
try:
    from app import build_rag_chain
except Exception:
    build_import_error = traceback.format_exc()

# -------------------------------
# Page Setup
# -------------------------------
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    layout="centered",
)

st.title("YouTube RAG Chatbot")
st.markdown("Ask questions about any YouTube video using Gemini")

# If importing the backend failed, surface the error to the user and stop
if build_import_error:
    st.error("Failed to import backend (app.py). See traceback below:")
    st.code(build_import_error)
    st.stop()

# -------------------------------
# Sidebar
# -------------------------------
with st.sidebar:
    st.header("Configuration")
    video_id = st.text_input("YouTube Video ID", value="1aA1WGON49E")
    k_results = st.slider("Retrieved Chunks (k)", 1, 8, 4)
    temperature = st.slider("Model Temperature", 0.0, 1.0, 0.2)
    rebuild_index = st.checkbox("Force rebuild knowledge base")

    build_button = st.button("Build Knowledge Base")
    reset_chat = st.button("Reset Chat History")

# -------------------------------
# Session State
# -------------------------------
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = ""

# -------------------------------
# Reset Logic
# -------------------------------
if reset_chat:
    st.session_state.chat_history = []
    st.rerun()

# Auto-reset if video ID changes
if video_id != st.session_state.current_video_id:
    st.session_state.chat_history = []
    st.session_state.rag_chain = None  # Force rebuild for new video
    st.session_state.current_video_id = video_id

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

    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(message)

    # Chat input
    if user_input := st.chat_input("Ask a question about the video"):
        # Add user message to history
        st.session_state.chat_history.append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate response
        with st.chat_message("assistant"):
            try:
                rag = st.session_state.rag_chain

                # If the chain exposes a streaming iterator, render tokens as they arrive
                if hasattr(rag, "stream"):
                    placeholder = st.empty()
                    response_text = ""
                    for token in rag.stream(user_input):
                        response_text += str(token)
                        placeholder.markdown(response_text)
                else:
                    # Fallback: try common call patterns
                    try:
                        result = rag.run(user_input)
                    except Exception:
                        try:
                            result = rag(user_input)
                        except Exception as e:
                            raise
                    response_text = str(result)
                    st.markdown(response_text)

                st.session_state.chat_history.append(("assistant", response_text))
            except Exception as e:
                st.error(str(e))

else:
    st.info("Enter a video ID and click 'Build Knowledge Base' to begin.")
