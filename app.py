import streamlit as st
import os

from utils.pdf_loader import get_pdf_text
from utils.text_splitter import get_text_chunks
from utils.vector_store import create_vector_store
from utils.rag_chain import ask_question


# -----------------------------
# Streamlit App Configuration
# -----------------------------
st.set_page_config(page_title="Enterprise RAG System")

st.title("💬 Chat with Your PDFs (Gemini + RAG)")


# -----------------------------
# Sidebar : Upload PDFs
# -----------------------------
with st.sidebar:

    st.header("Upload PDF Documents")

    uploaded_files = st.file_uploader(
        "Upload your PDFs",
        type="pdf",
        accept_multiple_files=True
    )


    if st.button("Process Documents"):

        if uploaded_files:

            file_paths = []

            for file in uploaded_files:

                file_path = os.path.join("data", file.name)

                with open(file_path, "wb") as f:
                    f.write(file.read())

                file_paths.append(file_path)

                with st.spinner("Processing documents..."):

                    raw_text = get_pdf_text(file_paths)

                    text_chunks = get_text_chunks(raw_text)

                    create_vector_store(text_chunks)

            st.success("Documents processed successfully!")


# -----------------------------
# Chat History
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# -----------------------------
# Ask Questions
# -----------------------------
st.header("Ask a Question")

question = st.text_input("Enter your question")

if st.button("Submit"):

    if question:

        answer = ask_question(question)

        st.session_state.history.append((question, answer))


# -----------------------------
# Display Chat History
# -----------------------------
for q, a in st.session_state.history[::-1]:

    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")