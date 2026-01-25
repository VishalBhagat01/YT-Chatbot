import os
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from langchain_core.output_parsers import StrOutputParser

# -------------------------------
# Setup
# -------------------------------
load_dotenv()

BASE_INDEX_DIR = "faiss_indexes"


# -------------------------------
# Transcript
# -------------------------------
def fetch_transcript(video_id: str, languages=None) -> str:
    if languages is None:
        languages = ["en"]

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

        # Try preferred languages
        for lang in languages:
            try:
                transcript = transcript_list.find_transcript([lang])
                return " ".join(item["text"] for item in transcript.fetch())
            except:
                pass

        # Fallback: try any available transcript
        for transcript in transcript_list:
            try:
                return " ".join(item["text"] for item in transcript.fetch())
            except:
                continue

        raise RuntimeError("No usable transcript found")

    except Exception as e:
        raise RuntimeError(
            "Transcript could not be fetched. "
            "This video may be private, age-restricted, region-blocked, "
            "or does not have captions."
        ) from e



# -------------------------------
# Helpers
# -------------------------------
def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


# -------------------------------
# FAISS Persistence
# -------------------------------
def save_vectorstore(store: FAISS, path: str):
    os.makedirs(path, exist_ok=True)
    store.save_local(path)


def load_vectorstore(path: str, embeddings):
    return FAISS.load_local(
        path, embeddings, allow_dangerous_deserialization=True
    )


# -------------------------------
# Public API
# -------------------------------
def build_rag_chain(
    video_id: str,
    k_results: int = 4,
    temperature: float = 0.2,
    rebuild: bool = False,
):
    index_dir = os.path.join(BASE_INDEX_DIR, video_id)

    # -------------------------------
    # Embeddings
    # -------------------------------
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    # -------------------------------
    # Load or Build Vector Store
    # -------------------------------
    if os.path.exists(index_dir) and not rebuild:
        vector_store = load_vectorstore(index_dir, embeddings)
    else:
        transcript_text = fetch_transcript(video_id)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )

        docs = splitter.create_documents([transcript_text])

        vector_store = FAISS.from_documents(docs, embeddings)
        save_vectorstore(vector_store, index_dir)

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
        model="gemini-1.5-flash",
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
