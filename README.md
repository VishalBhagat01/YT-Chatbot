# YouTube RAG Chatbot — Local Setup Guide

A Retrieval-Augmented Generation (RAG) chatbot that lets you **ask questions about any YouTube video** using:
- **yt-dlp** for reliable subtitle extraction (no IP blocking)
- **FAISS** for fast vector search
- **Local embeddings (Sentence Transformers)** — no API quotas
- **Google Gemini** for chat responses
- **Streamlit** for a simple web UI

---

## Project Structure

Chatbot/
│
├── app.py
├── streamlit_frontend.py
├── .env
├── README.md
└── faiss_indexes/


# Step 2 — Create Virtual Environment
Windows
python -m venv venv
venv\Scripts\activate

macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Step 3 — Install Dependencies


pip install -r requirements.txt


# RUN
streamlit run streamlit_frontend.py



