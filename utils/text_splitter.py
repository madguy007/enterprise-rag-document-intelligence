from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_text_chunks(text):
    """
    Splits long document text into smaller chunks
    suitable for embedding and retrieval.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )

    chunks = splitter.split_text(text)

    return chunks