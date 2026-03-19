"""
config.py
---------
Centralised configuration for the Enterprise RAG Platform.

All sensitive values (API keys, endpoints) are loaded from environment
variables — never hardcoded. In production these are injected via
Azure Key Vault or environment secrets in your deployment pipeline.

Local development:
    Copy .env.example to .env and fill in your values.
    The python-dotenv library loads .env automatically.

    cp .env.example .env
    # edit .env with your Azure credentials
"""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env file if present (local development only)
# In production, environment variables are injected by the platform
load_dotenv()


def _require(var: str) -> str:
    """
    Get a required environment variable.
    Raises a clear error at startup if it's missing — better to
    fail fast here than get a cryptic error mid-request.
    """
    value = os.getenv(var)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{var}' is not set. "
            f"Copy .env.example to .env and fill in your values."
        )
    return value


def _optional(var: str, default: str = "") -> str:
    """Get an optional environment variable with a default."""
    return os.getenv(var, default)


@dataclass(frozen=True)
class Settings:
    """
    Immutable settings object built from environment variables.

    Frozen dataclass = no accidental mutation of config at runtime.
    All values are validated at instantiation time.
    """

    # ------------------------------------------------------------------
    # Azure OpenAI
    # ------------------------------------------------------------------
    AZURE_OPENAI_ENDPOINT: str
    AZURE_OPENAI_API_KEY: str
    AZURE_OPENAI_API_VERSION: str
    AZURE_OPENAI_DEPLOYMENT: str          # e.g. "gpt-4o"
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str  # e.g. "text-embedding-3-large"

    # ------------------------------------------------------------------
    # Azure AI Search (vector store)
    # ------------------------------------------------------------------
    AZURE_SEARCH_ENDPOINT: str
    AZURE_SEARCH_API_KEY: str
    AZURE_SEARCH_INDEX_NAME: str

    # ------------------------------------------------------------------
    # Azure Blob Storage (document store)
    # ------------------------------------------------------------------
    AZURE_STORAGE_CONNECTION_STRING: str
    AZURE_STORAGE_CONTAINER_NAME: str

    # ------------------------------------------------------------------
    # Chunking defaults (can be overridden per use case)
    # ------------------------------------------------------------------
    CHUNK_MAX_TOKENS: int
    CHUNK_OVERLAP_TOKENS: int

    # ------------------------------------------------------------------
    # RAG retrieval defaults
    # ------------------------------------------------------------------
    RETRIEVAL_TOP_K: int        # number of chunks to retrieve
    RETRIEVAL_MIN_SCORE: float  # minimum similarity score threshold

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    LOG_LEVEL: str
    ENVIRONMENT: str            # "development" | "staging" | "production"


def _load_settings() -> Settings:
    """Build Settings from environment variables."""
    return Settings(
        # Azure OpenAI
        AZURE_OPENAI_ENDPOINT=_require("AZURE_OPENAI_ENDPOINT"),
        AZURE_OPENAI_API_KEY=_require("AZURE_OPENAI_API_KEY"),
        AZURE_OPENAI_API_VERSION=_optional("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
        AZURE_OPENAI_DEPLOYMENT=_optional("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        AZURE_OPENAI_EMBEDDING_DEPLOYMENT=_optional(
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-large"
        ),

        # Azure AI Search
        AZURE_SEARCH_ENDPOINT=_require("AZURE_SEARCH_ENDPOINT"),
        AZURE_SEARCH_API_KEY=_require("AZURE_SEARCH_API_KEY"),
        AZURE_SEARCH_INDEX_NAME=_optional("AZURE_SEARCH_INDEX_NAME", "rag-index"),

        # Azure Blob Storage
        AZURE_STORAGE_CONNECTION_STRING=_require("AZURE_STORAGE_CONNECTION_STRING"),
        AZURE_STORAGE_CONTAINER_NAME=_optional(
            "AZURE_STORAGE_CONTAINER_NAME", "rag-documents"
        ),

        # Chunking
        CHUNK_MAX_TOKENS=int(_optional("CHUNK_MAX_TOKENS", "400")),
        CHUNK_OVERLAP_TOKENS=int(_optional("CHUNK_OVERLAP_TOKENS", "50")),

        # Retrieval
        RETRIEVAL_TOP_K=int(_optional("RETRIEVAL_TOP_K", "5")),
        RETRIEVAL_MIN_SCORE=float(_optional("RETRIEVAL_MIN_SCORE", "0.75")),

        # Application
        LOG_LEVEL=_optional("LOG_LEVEL", "INFO"),
        ENVIRONMENT=_optional("ENVIRONMENT", "development"),
    )


# Single settings instance imported by the rest of the codebase
# e.g.: from src.config import settings
settings = _load_settings()
