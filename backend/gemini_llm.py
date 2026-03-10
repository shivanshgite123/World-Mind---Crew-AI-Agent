"""
Gemini LLM wrapper for CrewAI integration.
Uses google-generativeai SDK directly via LangChain wrapper.
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def get_gemini_llm(model: str = "gemini-1.5-flash", temperature: float = 0.3):
    """
    Returns a LangChain-compatible Gemini LLM instance.

    Args:
        model: Gemini model name (default: gemini-1.5-flash for speed + free tier)
        temperature: Creativity level 0.0–1.0 (lower = more factual)

    Returns:
        ChatGoogleGenerativeAI instance
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not found in environment. "
            "Please add it to your .env file. "
            "Get a free key at https://aistudio.google.com"
        )

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        convert_system_message_to_human=True,  
    )
