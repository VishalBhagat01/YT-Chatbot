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


