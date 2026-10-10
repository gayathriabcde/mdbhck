from pymongo import MongoClient
from pymongo.server_api import ServerApi
from core.config import Config

class DatabaseSetup:
    _client = None

    @classmethod
    def get_client(cls):
        """Returns a singleton MongoDB client using connection pooling."""
        if cls._client is None:
            try:
                # Initialize the client with strict API versioning for Atlas
                cls._client = MongoClient(
                    Config.MONGO_URI,
                    server_api=ServerApi('1'),
                    maxPoolSize=50, # Handles multiple Celery workers concurrently
                    connectTimeoutMS=5000
                )
                # Ping to verify connection
                cls._client.admin.command('ping')
                print("Successfully connected to MongoDB Atlas.")
            except Exception as e:
                print(f"Failed to connect to MongoDB: {e}")
                raise e
        return cls._client

    @classmethod
    def get_collection(cls):
        """Returns the specific collection for video nodes."""
        client = cls.get_client()
        db = client[Config.MONGO_DB_NAME]
        return db[Config.MONGO_COLLECTION_NAME]

# Expose a ready-to-use collection object
db_collection = DatabaseSetup.get_collection()