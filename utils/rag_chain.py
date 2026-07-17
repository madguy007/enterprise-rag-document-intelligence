import os

from langchain_google_genai import ChatGoogleGenerativeAI
from utils.config import format_gemini_error, get_google_api_key
from utils.vector_store import load_vector_store


def extract_response_text(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))

        if text_parts:
            return "\n\n".join(text_parts)

    return str(content)


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
        model="gemini-3.5-flash",
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
        return format_gemini_error(exc)

    return extract_response_text(response.content)
