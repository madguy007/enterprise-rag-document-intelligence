import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from utils.vector_store import load_vector_store

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def ask_question(question):

    if not os.path.exists("faiss_index"):
        return "Please upload and process PDFs first."

    """
    Handles the complete RAG pipeline:
    question → retrieve docs → send to LLM → return answer
    """

    db = load_vector_store()

    docs = db.similarity_search(question)

    # Combine retrieved documents into context
    context = "\n\n".join([doc.page_content for doc in docs])

    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
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

    response = model.invoke(prompt)

    return response.content