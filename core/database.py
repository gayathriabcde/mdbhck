from datetime import datetime, timezone

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from pymongo.server_api import ServerApi

from core.config import Config


def _utcnow():
    return datetime.now(timezone.utc)


class DatabaseSetup:
    _client = None
    _indexes_ready = False

    @classmethod
    def get_client(cls):
        """Returns a singleton MongoDB client using connection pooling."""
        if cls._client is None:
            try:
                cls._client = MongoClient(
                    Config.MONGO_URI,
                    server_api=ServerApi('1'),
                    maxPoolSize=50,
                    connectTimeoutMS=5000
                )
                cls._client.admin.command('ping')
                print("Successfully connected to MongoDB Atlas.")
            except Exception as e:
                print(f"Failed to connect to MongoDB: {e}")
                raise e
        return cls._client

    @classmethod
    def get_db(cls):
        return cls.get_client()[Config.MONGO_DB_NAME]

    @classmethod
    def get_collection(cls):
        """Returns the specific collection for video nodes."""
        collection = cls.get_db()[Config.MONGO_COLLECTION_NAME]
        cls.ensure_indexes()
        return collection

    @classmethod
    def get_source_registry(cls):
        registry = cls.get_db()[Config.SOURCE_REGISTRY_COLLECTION]
        cls.ensure_indexes()
        return registry

    @classmethod
    def ensure_indexes(cls):
        """B-tree indexes for graph edges, plus the source-registry unique key.

        The Atlas vector index is created when the cluster allows search indexes.
        Existing deployments that already have `vector_index` are left untouched.
        """
        if cls._indexes_ready:
            return
        nodes = cls.get_db()[Config.MONGO_COLLECTION_NAME]
        registry = cls.get_db()[Config.SOURCE_REGISTRY_COLLECTION]
        try:
            nodes.create_index([("video_id", 1), ("timestamp", 1)], name="video_timestamp")
            nodes.create_index("graph_edges.prev_node_id", name="prev_node")
            nodes.create_index("graph_edges.parent_segment_id", name="parent_segment")
            nodes.create_index("graph_edges.next_step_id", name="next_step")
            registry.create_index("video_id", unique=True, name="source_video_id")
            registry.create_index("status", name="source_status")
        except Exception as exc:
            print(f"Index setup skipped: {exc}")

        try:
            existing_names = set()
            try:
                existing_names = {index.get("name") for index in nodes.list_search_indexes()}
            except Exception:
                existing_names = set()
            if "vector_index" not in existing_names:
                nodes.create_search_index({
                    "name": "vector_index",
                    "type": "vectorSearch",
                    "definition": {
                        "fields": [{
                            "type": "vector",
                            "path": "vector_int8",
                            "numDimensions": 384,
                            "similarity": "cosine",
                        }]
                    },
                })
        except Exception as exc:
            print(f"Vector search index not created automatically: {exc}")

        cls._indexes_ready = True


db_collection = DatabaseSetup.get_collection()
source_registry = DatabaseSetup.get_source_registry()


def _public_source(document):
    if not document:
        return None
    document = dict(document)
    document.pop("_id", None)
    return document


def claim_for_ingest(video_id: str, url: str, title: str = None, origin_query: str = None, channel: str = None) -> dict:
    """Reserve a YouTube video in the source registry.

    Already indexed, queued, or processing videos are not dispatched again.
    A failed video can be claimed for another attempt. Concurrent claims lose
    the race on the unique video_id index and do not double-queue.
    """
    now = _utcnow()
    if db_collection.count_documents({"video_id": video_id}, limit=1):
        source_registry.update_one(
            {"video_id": video_id},
            {
                "$set": {
                    "url": url,
                    "title": title,
                    "channel": channel,
                    "status": "indexed",
                    "updated_at": now,
                    "error": None,
                },
                "$setOnInsert": {"video_id": video_id, "created_at": now},
            },
            upsert=True,
        )
        document = _public_source(source_registry.find_one({"video_id": video_id}))
        document["should_dispatch"] = False
        return document

    fields = {
        "url": url,
        "title": title,
        "channel": channel,
        "origin_query": origin_query,
        "status": "queued",
        "updated_at": now,
        "error": None,
        "task_id": None,
    }
    try:
        result = source_registry.update_one(
            {"video_id": video_id, "status": {"$nin": ["queued", "processing", "indexed"]}},
            {
                "$set": fields,
                "$setOnInsert": {"video_id": video_id, "created_at": now},
            },
            upsert=True,
        )
    except DuplicateKeyError:
        document = _public_source(source_registry.find_one({"video_id": video_id})) or {
            "video_id": video_id,
            "url": url,
            "status": "queued",
        }
        document["should_dispatch"] = False
        return document

    document = _public_source(source_registry.find_one({"video_id": video_id})) or {
        "video_id": video_id,
        "url": url,
        "title": title,
        "status": "queued",
    }
    document["should_dispatch"] = bool(result.upserted_id or result.modified_count)
    return document


def attach_task(video_id: str, task_id: str):
    source_registry.update_one(
        {"video_id": video_id},
        {"$set": {"task_id": task_id, "status": "queued", "updated_at": _utcnow()}},
    )


def mark_source_status(video_id: str, status: str, task_id: str = None, error: str = None, node_count: int = None):
    if not video_id:
        return
    update = {"status": status, "updated_at": _utcnow(), "error": error}
    if task_id:
        update["task_id"] = task_id
    if node_count is not None:
        update["node_count"] = node_count
    if status == "indexed":
        update["indexed_at"] = _utcnow()
    source_registry.update_one({"video_id": video_id}, {"$set": update}, upsert=True)
