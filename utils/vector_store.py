from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from utils.config import get_google_api_key


def get_embeddings():
    google_api_key = get_google_api_key()

    if not google_api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add it in Streamlit Cloud app secrets and reboot the app."
        )

    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=google_api_key
    )


def create_vector_store(text_chunks):
    """
    Creates FAISS vector store from text chunks
    using Gemini embeddings and saves locally.
    """

    embeddings = get_embeddings()

    vector_store = FAISS.from_texts(
    text_chunks,
    embedding=embeddings,
    metadatas=[{"chunk_id": i} for i in range(len(text_chunks))]
    )

    vector_store.save_local("faiss_index")


def load_vector_store():
    """
    Loads the saved FAISS vector store.
    """

    embeddings = get_embeddings()

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db
