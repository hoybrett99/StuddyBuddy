"""
Configuration using Pydantic Settings.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache
from pydantic import Field
from typing import Optional, List

class Settings(BaseSettings):
    """
    Application settings.
    
    Pydantic will automatically:
    1. Look for environment variables
    2. Look in .env file
    3. Use defaults if not found
    """
    # API KEYS
    google_api_key: str = Field(
        ...,  # Required field
        min_length=1,
        description="Google Gemini API key"
    )
    
    claude_api_key: str = Field(
        ...,  # Required field
        min_length=1,
        description="Anthropic Claude API key"
    )

    # Application settings
    app_name: str = "Study Buddy"
    app_version: str = "0.0.1"
    debug: bool = False

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # File upload settings
    max_file_size_mb: int = 50
    allowed_file_type: List[str] = ["pdf", "txt", "docx"]
    upload_dir: str = "/app/uploads"

    # ChromaDB settings
    chroma_host: str = "chromadb"
    chroma_port: int = 8000
    chroma_collection: str = "study_buddy"

    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # LLM Model Selection
    llm_provider: str = "gemini"

    # RAG settings
    default_num_contexts: int = 4
    gemini_model: str = "gemini-2.0-flash"
    gemini_max_tokens: int = 8192

    claude_model: str = "claude-sonnet-4-20250514"
    claude_max_tokens: int = 8192
    
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "extra": "ignore"
    }

@lru_cache()  # Only create settings once, then reuse
def get_settings() -> Settings:
    """
    Get application settings.
    """
    return Settings()

if __name__ == "__main__":
    settings = get_settings()
    print(f"Claude Model: {settings.claude_model}")
    print(f"Gemini Model: {settings.gemini_model}")
    print(f"Claude API Key: {settings.claude_api_key[:10]}...")
    print(f"App Name: {settings.app_name}")