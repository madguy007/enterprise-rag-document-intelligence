from PyPDF2 import PdfReader


def get_pdf_text(pdf_paths):
    """
    Extracts text from multiple PDF files.
    """

    text = ""

    for path in pdf_paths:
        pdf_reader = PdfReader(path)

        for page in pdf_reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text