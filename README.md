# 📺 YouTube RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)
![LangChain](https://img.shields.io/badge/LangChain-Latest-green.svg)
![Gemini](https://img.shields.io/badge/Google-Gemini-orange.svg)

A powerful **Retrieval-Augmented Generation (RAG)** chatbot that allows you to "talk" to YouTube videos. Just provide a YouTube Video ID, and the application will fetch the transcript, index it, and let you ask questions about the content using Google's Gemini LLM.

## 🚀 Features

- **Instant Transcript Processing**: Fetches transcripts using `youtube-transcript-api` with a fallback to `yt-dlp`.
- **Intelligent RAG Pipeline**: Uses LangChain for text splitting, HuggingFace for embeddings, and FAISS for efficient vector storage.
- **Streaming Responses**: Real-time chat experience with streaming output in Streamlit.
- **Local Persistence**: Vector indexes are cached locally per video ID to speed up repeated queries.
- **Interactive UI**: Clean and modern Streamlit interface.

## 🛠 Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini (via `langchain-google-genai`)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Database**: FAISS
- **Framework**: LangChain

## 📋 Prerequisites

- Python 3.8+
- A Google Cloud Project with the **Generative AI API** enabled.
- A Google API Key. [Get one here](https://aistudio.google.com/app/apikey).

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/VishalBhagat01/YT-Chatbot.git
   cd YT-Chatbot
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the root directory and add your Google API Key:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

## 🎯 Usage

1. **Run the Streamlit app**:
   ```bash
   streamlit run streamlit_frontend.py
   ```

2. **In the browser**:
   - Enter a **YouTube Video ID** (e.g., `Gfr50f6ZBvo`) in the sidebar.
   - Click **"Build Knowledge Base"**.
   - Start chatting with the video in the main window!

## 📁 Project Structure

- `app.py`: Contains the core RAG logic and chain building.
- `streamlit_frontend.py`: The Streamlit web application.
- `faiss_indexes/`: Directory where video embeddings are stored.
- `.env`: (Ignored) Environment variables.
- `requirements.txt`: Project dependencies.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
>>>>>>> 031b243 (Fixed)
