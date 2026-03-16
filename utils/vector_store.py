from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(text_chunks):
    """
    Creates FAISS vector store from text chunks
    using Gemini embeddings and saves locally.
    """

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

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

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001"
    )

    db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return db