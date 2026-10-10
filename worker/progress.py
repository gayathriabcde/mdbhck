"""Publish pipeline stage updates for the Server-Sent Events stream."""

import json

from core.config import Config

_redis_client = None


def progress_channel(task_id: str) -> str:
    return f"odp:progress:{task_id}"


def progress_key(task_id: str) -> str:
    return f"odp:progress:{task_id}"


def get_redis():
    global _redis_client
    if _redis_client is None:
        import redis
        _redis_client = redis.Redis.from_url(Config.REDIS_URI, decode_responses=True)
    return _redis_client


def publish_progress(task_id: str, stage: str, percent: int, detail: str = "", status: str = "running", video_id: str = None):
    """Store the latest stage and fan it out to live subscribers.

    Redis is optional at development time: a broker outage must not fail ingestion.
    """
    if not task_id:
        return
    payload = {
        "task_id": task_id,
        "stage": stage,
        "percent": int(max(0, min(100, percent))),
        "detail": detail or "",
        "status": status,
        "video_id": video_id,
    }
    try:
        client = get_redis()
        encoded = json.dumps(payload)
        client.set(progress_key(task_id), encoded, ex=60 * 60 * 24)
        client.publish(progress_channel(task_id), encoded)
    except Exception as exc:
        print(f"[Progress] Could not publish update for {task_id}: {exc}")
