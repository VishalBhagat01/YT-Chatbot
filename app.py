import os
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
from langchain_postgres import PGVector
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser
import subprocess
import tempfile
import json
# -------------------------------
# Setup
# -------------------------------
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)


# -------------------------------
# Transcript
# -------------------------------

def fetch_transcript(video_id: str, languages=None) -> str:
    if languages is None:
        languages = ["en"]

    # Try YouTubeTranscriptApi first
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        return " ".join([t["text"] for t in transcript_list])
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable):
        pass
    except Exception as e:
        print(f"YouTubeTranscriptApi failed: {e}")

    # Fallback to yt-dlp
    with tempfile.TemporaryDirectory() as tmpdir:
        output_template = os.path.join(tmpdir, "%(id)s.%(ext)s")

        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-auto-subs",
            "--sub-format", "vtt",
            "--sub-lang", ",".join(languages),
            "-o", output_template,
            f"https://www.youtube.com/watch?v={video_id}",
        ]

        subprocess.run(cmd, capture_output=True)

        for file in os.listdir(tmpdir):
            if file.endswith(".vtt"):
                path = os.path.join(tmpdir, file)
                with open(path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                # Strip timestamps
                text_lines = [
                    line.strip()
                    for line in lines
                    if line.strip() and "-->" not in line and not line.startswith("WEBVTT")
                ]

                if text_lines:
                    return " ".join(text_lines)

    raise RuntimeError(f"Could not extract subtitles from video {video_id}")



# -------------------------------
# Helpers
# -------------------------------
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# Helper removed as we use PGVector directly


# -------------------------------
# Public API
# -------------------------------
def build_rag_chain(
    video_id: str,
    k_results: int = 4,
    temperature: float = 0.2,
    rebuild: bool = False,
):
    # -------------------------------
    # Embeddings
    # -------------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # -------------------------------
    # Load or Build Vector Store (PostgreSQL)
    # -------------------------------
    # We use pre_delete_collection=rebuild to handle the rebuild flag.
    # If not rebuilding, we still need to check if we should add documents.
    # To keep it simple and robust, we'll fetch the transcript if needed.
    
    collection_name = f"video_{video_id}"
    
    # We'll use a simple approach: if we can't find the collection or it's rebuild, we build it.
    # Note: PGVector.from_documents will create the collection if it doesn't exist.
    
    if rebuild:
        transcript_text = fetch_transcript(video_id)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.create_documents([transcript_text])
        
        vector_store = PGVector.from_documents(
            embedding=embeddings,
            documents=docs,
            collection_name=collection_name,
            connection=DATABASE_URL,
            use_jsonb=True,
            pre_delete_collection=True,
        )
    else:
        # Just initialize the store - it will connect to existing collection if it exists
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=DATABASE_URL,
            use_jsonb=True,
        )
        
        # Optional: Check if empty and build? 
        # For simplicity, we'll assume if not rebuild, the user expects it to be there 
        # or they will trigger a rebuild if chat fails.
        # However, to be user-friendly, let's auto-build if no docs found.
        # But checking count in PGVector is slightly involved.
        # Let's try to add documents ONLY if it's the first time for this video.
        # We can do this by trying to fetch transcript and adding if needed.
        # But for now, sticking to the standard pattern.

    # -------------------------------
    # Retriever
    # -------------------------------
    retriever = vector_store.as_retriever(
        search_kwargs={"k": k_results}
    )

    # -------------------------------
    # LLM
    # -------------------------------
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
    )

    # -------------------------------
    # Prompt
    # -------------------------------
    prompt = PromptTemplate(
        template="""
You are a YouTube video assistant.

Answer ONLY using the transcript context below.
If the answer is not found, say:
"Query is not available in the video."

Context:
{context}

Question:
{question}
""",
        input_variables=["context", "question"],
    )

    # -------------------------------
    # RAG Chain
    # -------------------------------
    rag_chain = (
        RunnableParallel(
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
        )
        | prompt
        | model
        | StrOutputParser()
    )

    return rag_chain
