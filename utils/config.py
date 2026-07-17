import os

from dotenv import load_dotenv


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


def format_gemini_error(exc):
    message = str(exc)
    api_key = get_google_api_key()

    if api_key:
        message = message.replace(api_key, "[redacted]")

    return (
        "Gemini could not generate an answer.\n\n"
        f"Error type: {type(exc).__name__}\n\n"
        f"Details: {message[:1000]}"
    )
