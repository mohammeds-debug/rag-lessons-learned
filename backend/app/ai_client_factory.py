"""
AI Client Factory.

Selects the AI provider based on CLIENT_TO_BE_USED environment variable:
  "gemini"         -> GeminiClient  (default)
  "openai"/"azure" -> LLMClient     (OpenAI-compatible)

Singleton pattern: connection is reused across requests.
"""

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


class AIClientFactory:
    _instance = None
    _client = None
    _client_provider = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_client(cls) -> Any:
        provider = os.getenv("CLIENT_TO_BE_USED", "gemini").lower()

        if cls._client is None or cls._client_provider != provider:
            cls._client_provider = provider
            if provider == "gemini":
                try:
                    from app.gemini_client import GeminiClient
                except ImportError:
                    from gemini_client import GeminiClient
                cls._client = GeminiClient()
            else:
                try:
                    from app.llm_client import LLMClient
                except ImportError:
                    from llm_client import LLMClient
                cls._client = LLMClient()

        return cls._client

    @classmethod
    def get_client_type(cls) -> str:
        provider = os.getenv("CLIENT_TO_BE_USED", "gemini").lower()
        return "gemini" if provider == "gemini" else "openai-compatible"


def get_ai_client() -> Any:
    return AIClientFactory.get_client()


def get_client_info() -> dict:
    provider = os.getenv("CLIENT_TO_BE_USED", "gemini").lower()
    if provider == "gemini":
        return {
            "client_type": "gemini",
            "model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
            "provider": "Google Gemini",
            "required_env_vars": ["GEMINI_API_KEY"],
        }
    return {
        "client_type": "openai-compatible",
        "model": os.getenv("AZURE_AI_CHAT_MODEL", "gpt-4o-mini"),
        "provider": "OpenAI-compatible (Azure)",
        "required_env_vars": [
            "AZURE_AI_CHAT_ENDPOINT",
            "AZURE_AI_CHAT_API_KEY",
            "AZURE_AI_CHAT_MODEL",
            "AZURE_AI_CHAT_API_VERSION",
        ],
    }
