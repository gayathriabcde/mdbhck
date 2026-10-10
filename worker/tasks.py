from celery import Celery

from app.utils import clamp_discover_limit, extract_youtube_video_id
from core.config import Config
from core.database import attach_task, claim_for_ingest, mark_source_status
from worker.ml_pipeline import process_video_pipeline
from worker.progress import publish_progress
from worker.youtube_discovery import discover_youtube_videos

celery_app = Celery(
    'odp_rag_worker',
    broker=Config.REDIS_URI,
    backend=Config.REDIS_URI
)

celery_app.conf.update(
    worker_concurrency=1,
    task_acks_late=True,
    worker_prefetch_multiplier=1
)


def enqueue_video(url: str, video_id: str = None, title: str = None, origin_query: str = None, channel: str = None) -> dict:
    """Dedup against the source registry, then dispatch ingestion for new videos."""
    resolved_id = video_id or extract_youtube_video_id(url)
    if not resolved_id:
        raise ValueError(f"Could not read a YouTube video id from {url}")

    claim = claim_for_ingest(
        video_id=resolved_id,
        url=url,
        title=title,
        origin_query=origin_query,
        channel=channel,
    )
    public = {
        "video_id": resolved_id,
        "url": claim.get("url") or url,
        "title": claim.get("title") or title or resolved_id,
        "channel": claim.get("channel") or channel or "",
        "status": claim.get("status"),
        "task_id": claim.get("task_id"),
    }
    if not claim.get("should_dispatch"):
        return public

    try:
        task = process_video_task.delay(url)
    except Exception as exc:
        mark_source_status(resolved_id, "failed", error=str(exc))
        raise
    attach_task(resolved_id, task.id)
    public["status"] = "queued"
    public["task_id"] = task.id
    return public


def discover_and_enqueue(query: str, limit: int = 5) -> dict:
    """Search YouTube and queue only videos the source registry has not indexed."""
    bounded = clamp_discover_limit(limit)
    videos = discover_youtube_videos(query, limit=bounded)
    results = []
    for video in videos:
        results.append(enqueue_video(
            url=video["url"],
            video_id=video["video_id"],
            title=video.get("title"),
            origin_query=query,
            channel=video.get("channel"),
        ))
    return {
        "query": query,
        "limit": bounded,
        "dispatched": sum(1 for item in results if item.get("status") == "queued"),
        "results": results,
    }


@celery_app.task(bind=True, name="process_video_task")
def process_video_task(self, youtube_url: str):
    """
    The main asynchronous task triggered by the API.
    """
    task_id = self.request.id or ""
    video_id = extract_youtube_video_id(youtube_url)
    print(f"[Worker] Starting ingestion for: {youtube_url}")

    def report(stage, percent, detail=""):
        status = "success" if percent >= 100 else "running"
        publish_progress(task_id, stage, percent, detail=detail, status=status, video_id=video_id)
        self.update_state(state="PROGRESS", meta={
            "stage": stage,
            "percent": percent,
            "detail": detail,
            "video_id": video_id,
        })

    try:
        if video_id:
            mark_source_status(video_id, "processing", task_id=task_id)
        result = process_video_pipeline(youtube_url, progress_callback=report)
        video_id = result.get("video_id") or video_id
        mark_source_status(
            video_id,
            "indexed",
            task_id=task_id,
            node_count=result.get("nodes_inserted", 0),
        )
        publish_progress(
            task_id,
            "Ingestion complete.",
            100,
            detail=f"{result.get('nodes_inserted', 0)} nodes, {result.get('step_count', 0)} steps",
            status="success",
            video_id=video_id,
        )
        print(f"[Worker] Successfully processed {youtube_url}")
        return {"status": "success", **result}
    except Exception as exc:
        print(f"[Worker] FAILED processing {youtube_url}: {exc}")
        if video_id:
            mark_source_status(video_id, "failed", task_id=task_id, error=str(exc))
        publish_progress(task_id, "Ingestion failed.", 100, detail=str(exc), status="error", video_id=video_id)
        raise exc
