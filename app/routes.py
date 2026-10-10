from flask import Blueprint, jsonify, request

from app.retrieval import query_video_graph
from worker.tasks import process_video_task
from core.database import db_collection

api_bp = Blueprint('api', __name__)


@api_bp.route('/ingest', methods=['POST'])
def ingest_video():
    """Endpoint to trigger asynchronous video processing."""
    data = request.get_json()
    youtube_url = data.get('url')

    if not youtube_url:
        return jsonify({"error": "Missing 'url' parameter"}), 400

    # Send the job to Celery Redis Queue
    task = process_video_task.delay(youtube_url)

    return jsonify({
        "message": "Video ingestion started in background.",
        "task_id": task.id
    }), 202


@api_bp.route('/nodes/<video_id>', methods=['GET'])
def get_video_nodes(video_id):
    """Temporary endpoint to verify data made it to MongoDB."""
    nodes = list(db_collection.find({"video_id": video_id}, {"_id": 0, "vector_int8": 0}))  # Hide giant vectors for API
    return jsonify({"video_id": video_id, "node_count": len(nodes), "nodes": nodes}), 200


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Simple health check to ensure the API and DB are running."""
    try:
        # Check if DB is responsive by counting documents
        doc_count = db_collection.count_documents({})
        return jsonify({
            "status": "healthy",
            "database_documents": doc_count,
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
    data = request.get_json()
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