import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from utils.vector_store import load_vector_store

# Load environment variables
load_dotenv()


def get_google_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")

    if api_key:
        return api_key

    try:
        import streamlit as st

        return st.secrets.get("GOOGLE_API_KEY")
    except Exception:
        return None


def ask_question(question):

    if not os.path.exists(os.path.join("faiss_index", "index.faiss")):
        return "Please upload and process PDFs first."

    """
    Handles the complete RAG pipeline:
    question → retrieve docs → send to LLM → return answer
    """

    db = load_vector_store()

    docs = db.similarity_search(question)

    # Combine retrieved documents into context
    context = "\n\n".join([doc.page_content for doc in docs])

    google_api_key = get_google_api_key()

    if not google_api_key:
        return "GOOGLE_API_KEY is missing. Add it in Streamlit Cloud app secrets and reboot the app."

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=google_api_key
    )

    prompt = f"""
Answer the question as detailed as possible from the provided context.
If the answer is not available in the context, say "answer is not available in the context".

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = model.invoke(prompt)
    except Exception as exc:
        return (
            "Gemini could not generate an answer. Please check that your Streamlit "
            "secret contains a valid Gemini API key from Google AI Studio, then "
            f"reboot the app. Error type: {type(exc).__name__}"
        )

    return response.content
