import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()


class Config:
    FLASK_ENV = os.getenv("FLASK_ENV", "development")
    PORT = int(os.getenv("PORT", 5000))

    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "odp_rag_db")
    MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "video_nodes")
    SOURCE_REGISTRY_COLLECTION = os.getenv("SOURCE_REGISTRY_COLLECTION", "source_registry")

    REDIS_URI = os.getenv("REDIS_URI", "redis://localhost:6379/0")

    # Optional. Discovery uses yt-dlp when this is empty.
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

    # Optional LLM providers. The analytics route falls back to a citation-preserving
    # template when none of these are configured.
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")
    OLLAMA_HOST = os.getenv("OLLAMA_HOST", "")
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

    KEYFRAME_FPS = float(os.getenv("KEYFRAME_FPS", "0.25"))

    # Validation
    if not MONGO_URI:
        raise ValueError("CRITICAL: MONGO_URI is missing from .env file.")
