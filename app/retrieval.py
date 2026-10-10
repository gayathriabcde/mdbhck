from sentence_transformers import SentenceTransformer
from core.database import db_collection
from worker.ml_pipeline import quantize_to_int8 # Reuse our quantization logic

# Load the embedder locally for the web server
# (This is lightweight enough to run on the main Flask thread for text queries)
embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

def query_video_graph(user_query: str, limit: int = 3):
    """
    Embeds the user query, finds the target node via vector search,
    and traverses backward to get prerequisite steps.
    """
    # 1. Embed the user's text query
    float_query = embedder.encode(user_query)
    int8_query = quantize_to_int8(float_query)

    # 2. The ODP-RAG Aggregation Pipeline
    pipeline = [
        # Stage 1: Vector Search for the primary target node
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "vector_int8",
                "queryVector": int8_query,
                "numCandidates": 50,
                "limit": limit
            }
        },
        # Stage 2: The Topological Traversal (The Secret Sauce)
        # We start at the target node, and walk backward up the 'prev_node_id' chain
        {
            "$graphLookup": {
                "from": "video_nodes",
                "startWith": "$graph_edges.prev_node_id",
                "connectFromField": "graph_edges.prev_node_id",
                "connectToField": "_id",
                "as": "prerequisite_steps",
                "maxDepth": 2, # How many steps backward to fetch (0=1 step, 1=2 steps, etc.)
                "depthField": "steps_backward"
            }
        },
        # Stage 3: Clean up the output to send to the Frontend
        {
            "$project": {
                "_id": {"$toString": "$_id"},
                "video_id": 1,
                "timestamp": 1,
                "visual_caption": 1,
                "audio_transcript": 1,
                "dissonance_score": 1,
                "search_score": {"$meta": "vectorSearchScore"},
                # Sort prerequisites chronologically (deepest depth first)
                "prerequisite_steps": {
                    "$map": {
                        "input": "$prerequisite_steps",
                        "as": "step",
                        "in": {
                            "step_id": {"$toString": "$$step._id"},
                            "timestamp": "$$step.timestamp",
                            "visual_caption": "$$step.visual_caption",
                            "audio_transcript": "$$step.audio_transcript",
                            "steps_backward": "$$step.steps_backward"
                        }
                    }
                }
            }
        }
    ]

    # 3. Execute the pipeline
    results = list(db_collection.aggregate(pipeline))
    return results