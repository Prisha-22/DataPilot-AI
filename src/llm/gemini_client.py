
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from google import genai


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load the local .env file when running on your computer.
load_dotenv(PROJECT_ROOT / ".env")

# Prefer an environment variable; otherwise use Streamlit Secrets.
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = None

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is missing. Configure it in your local .env "
        "file or Streamlit Cloud Secrets."
    )

client = genai.Client(api_key=api_key)
