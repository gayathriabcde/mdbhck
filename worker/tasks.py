import os
from celery import Celery
from worker.ml_pipeline import process_video_pipeline
from core.config import Config

# Initialize Celery connected to Redis
celery_app = Celery(
    'odp_rag_worker',
    broker=Config.REDIS_URI,
    backend=Config.REDIS_URI
)

# Celery settings optimized for local ML processing
celery_app.conf.update(
    worker_concurrency=1,  # Keep at 1 for your i5 CPU to avoid thermal throttling
    task_acks_late=True,   # Only acknowledge task completion when actually done
    worker_prefetch_multiplier=1 # Don't load multiple heavy video jobs into memory
)

@celery_app.task(bind=True, name="process_video_task")
def process_video_task(self, youtube_url: str):
    """
    The main asynchronous task triggered by the Flask API.
    """
    print(f"[Worker] Starting ingestion for: {youtube_url}")
    try:
        # Pass to our ML pipeline
        result = process_video_pipeline(youtube_url)
        print(f"[Worker] Successfully processed {youtube_url}")
        return {"status": "success", "nodes_inserted": result}
    except Exception as e:
        print(f"[Worker] FAILED processing {youtube_url}: {str(e)}")
        raise e