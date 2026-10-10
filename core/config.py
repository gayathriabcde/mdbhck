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

    REDIS_URI = os.getenv("REDIS_URI", "redis://localhost:6379/0")

    # Validation
    if not MONGO_URI:
        raise ValueError("CRITICAL: MONGO_URI is missing from .env file.")