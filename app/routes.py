import json

from flask import Blueprint, Response, jsonify, request, stream_with_context

from app.retrieval import query_video_graph
from app.synthesis import run_analytics
from app.utils import extract_youtube_urls, is_valid_youtube_url
from core.database import db_collection, source_registry
from worker.progress import get_redis, progress_channel, progress_key
from worker.tasks import discover_and_enqueue, enqueue_video

api_bp = Blueprint('api', __name__)


def _ingest_message(queued: dict) -> str:
    status = queued.get("status")
    if status == "queued":
        return "Video ingestion started in background."
    if status == "indexed":
        return "Video is already in the library."
    if status == "processing":
        return "Video is already being processed."
    return "Video was not queued again."


@api_bp.route('/ingest', methods=['POST'])
def ingest_video():
    """Ingest a YouTube URL, or discover videos when the body carries a text query."""
    data = request.get_json(silent=True) or {}
    youtube_url = (data.get('url') or '').strip()
    query = (data.get('query') or '').strip()

    if youtube_url:
        if not is_valid_youtube_url(youtube_url):
            return jsonify({"error": "Invalid YouTube URL"}), 400
        try:
            queued = enqueue_video(youtube_url, origin_query=query or None)
        except Exception as exc:
            print(f"[API] Ingest error: {exc}")
            return jsonify({"error": "Could not queue that video."}), 500
        status_code = 202 if queued.get("status") == "queued" else 200
        return jsonify({"message": _ingest_message(queued), **queued}), status_code

    if query:
        urls = extract_youtube_urls(query)
        if urls:
            try:
                results = [enqueue_video(url, origin_query=query) for url in urls]
            except Exception as exc:
                print(f"[API] Ingest error: {exc}")
                return jsonify({"error": "Could not queue those videos."}), 500
            return jsonify({
                "message": "Video ingestion updated.",
                "query": query,
                "dispatched": sum(1 for item in results if item.get("status") == "queued"),
                "results": results,
            }), 202
        try:
            discovered = discover_and_enqueue(query, limit=data.get("limit", 5))
        except Exception as exc:
            print(f"[API] Discovery error: {exc}")
            return jsonify({"error": "YouTube discovery failed."}), 502
        return jsonify({"message": "Discovery complete.", **discovered}), 202

    return jsonify({"error": "Provide a YouTube 'url' or a text 'query'."}), 400


@api_bp.route('/discover', methods=['POST'])
def discover_videos():
    """Fetch the top 3–5 YouTube results and ingest anything not already registered."""
    data = request.get_json(silent=True) or {}
    query = (data.get('query') or '').strip()
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    urls = extract_youtube_urls(query)
    if urls:
        try:
            results = [enqueue_video(url, origin_query=query) for url in urls]
        except Exception as exc:
            print(f"[API] Discovery error: {exc}")
            return jsonify({"error": "Could not queue those videos."}), 500
        return jsonify({
            "query": query,
            "dispatched": sum(1 for item in results if item.get("status") == "queued"),
            "results": results,
        }), 202

    try:
        payload = discover_and_enqueue(query, limit=data.get("limit", 5))
    except Exception as exc:
        print(f"[API] Discovery error: {exc}")
        return jsonify({"error": "YouTube discovery failed."}), 502
    return jsonify(payload), 202


@api_bp.route('/analytics', methods=['POST'])
def analytics():
    """LLM (or citation template) synthesis over computed step differences."""
    data = request.get_json(silent=True) or {}
    user_query = (data.get('query') or '').strip()
    if not user_query:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    try:
        payload = run_analytics(user_query, limit=data.get('limit', 5))
        return jsonify(payload), 200
    except Exception as exc:
        print(f"[API] Analytics error: {exc}")
        return jsonify({"error": "Internal server error during analytics"}), 500


@api_bp.route('/progress/<task_id>', methods=['GET'])
def progress_stream(task_id):
    """Server-Sent Events for live ingestion stages."""

    def generate():
        try:
            client = get_redis()
            pubsub = client.pubsub(ignore_subscribe_messages=True)
            pubsub.subscribe(progress_channel(task_id))

            snapshot = client.get(progress_key(task_id))
            if snapshot:
                yield f"data: {snapshot}\n\n"
                payload = json.loads(snapshot)
                if payload.get("status") in ("success", "error"):
                    pubsub.close()
                    return

            while True:
                message = pubsub.get_message(timeout=15.0)
                if message and message.get("type") == "message":
                    data = message.get("data")
                    yield f"data: {data}\n\n"
                    payload = json.loads(data)
                    if payload.get("status") in ("success", "error"):
                        break
                else:
                    latest = client.get(progress_key(task_id))
                    if latest and latest != snapshot:
                        snapshot = latest
                        yield f"data: {latest}\n\n"
                        payload = json.loads(latest)
                        if payload.get("status") in ("success", "error"):
                            break
                    else:
                        yield ": keepalive\n\n"
            pubsub.close()
        except GeneratorExit:
            return
        except Exception as exc:
            error = json.dumps({
                "task_id": task_id,
                "stage": "Live progress unavailable.",
                "percent": 0,
                "detail": str(exc),
                "status": "error",
            })
            yield f"data: {error}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive',
        },
    )


@api_bp.route('/nodes/<video_id>', methods=['GET'])
def get_video_nodes(video_id):
    """Temporary endpoint to verify data made it to MongoDB."""
    nodes = list(db_collection.find({"video_id": video_id}, {"_id": 0, "vector_int8": 0}))
    return jsonify({"video_id": video_id, "node_count": len(nodes), "nodes": nodes}), 200


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Simple health check to ensure the API and DB are running."""
    try:
        doc_count = db_collection.count_documents({})
        registry_count = source_registry.count_documents({})
        return jsonify({
            "status": "healthy",
            "database_documents": doc_count,
            "source_registry": registry_count,
            "message": "ODP-RAG API is operational."
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@api_bp.route('/query', methods=['POST'])
def search_procedural_graph():
    """
    Endpoint for the frontend to query the video graph.
    Expected JSON: {"query": "how to tighten the bolt", "limit": 3}
    """
    data = request.get_json(silent=True) or {}
    user_query = data.get('query')
    limit = data.get('limit', 3)

    if not user_query:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    try:
        print(f"[API] Searching graph for: '{user_query}'")
        results = query_video_graph(user_query, limit=limit)

        return jsonify({
            "query": user_query,
            "results_count": len(results),
            "data": results
        }), 200

    except Exception as e:
        print(f"[API] Query Error: {str(e)}")
        return jsonify({"error": "Internal server error during retrieval"}), 500
