from sentence_transformers import SentenceTransformer

from app.utils import citation_token
from core.config import Config
from core.database import db_collection
from worker.ml_pipeline import quantize_to_int8

# Lightweight enough to run on the API process for text queries.
embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

_DIVERGENCE_THRESHOLD = 0.62


def _cosine(vec_a, vec_b) -> float:
    if vec_a is None or vec_b is None:
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for left, right in zip(vec_a, vec_b):
        x = float(left)
        y = float(right)
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / ((norm_a ** 0.5) * (norm_b ** 0.5))


def query_video_graph(user_query: str, limit: int = 3):
    """
    Embeds the user query, finds the target node via vector search,
    and traverses backward through frame edges and step edges.
    """
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 3
    limit = max(1, min(limit, 20))
    # Atlas wants numCandidates >= limit. A wider candidate pool improves recall
    # without returning more documents than `limit`.
    num_candidates = min(max(limit * 20, 100), 200)

    float_query = embedder.encode(user_query)
    int8_query = quantize_to_int8(float_query)
    collection_name = Config.MONGO_COLLECTION_NAME

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "vector_int8",
                "queryVector": int8_query,
                "numCandidates": num_candidates,
                "limit": limit,
            }
        },
        # Frame-level prerequisites. Edges are ObjectIds, so the walk stays inside
        # the video that owns them. The prev_node index supports this traversal.
        {
            "$graphLookup": {
                "from": collection_name,
                "startWith": "$graph_edges.prev_node_id",
                "connectFromField": "graph_edges.prev_node_id",
                "connectToField": "_id",
                "as": "prerequisite_steps",
                "maxDepth": 8,
                "depthField": "steps_backward",
            }
        },
        # Step-level prerequisites via parent_segment_id. Depth here is steps, not frames.
        {
            "$graphLookup": {
                "from": collection_name,
                "startWith": "$graph_edges.parent_segment_id",
                "connectFromField": "graph_edges.parent_segment_id",
                "connectToField": "_id",
                "as": "step_chain",
                "maxDepth": 6,
                "depthField": "steps_backward",
            }
        },
        {
            "$lookup": {
                "from": collection_name,
                "localField": "graph_edges.next_step_id",
                "foreignField": "_id",
                "as": "following_step_docs",
            }
        },
        {
            "$project": {
                "_id": {"$toString": "$_id"},
                "video_id": 1,
                "timestamp": 1,
                "visual_caption": 1,
                "audio_transcript": 1,
                "ocr_text": 1,
                "dissonance_score": 1,
                "is_step_boundary": 1,
                "step_index": 1,
                "parent_segment_id": {"$toString": "$graph_edges.parent_segment_id"},
                "next_step_id": {"$toString": "$graph_edges.next_step_id"},
                "search_score": {"$meta": "vectorSearchScore"},
                "prerequisite_steps": {
                    "$map": {
                        "input": "$prerequisite_steps",
                        "as": "step",
                        "in": {
                            "step_id": {"$toString": "$$step._id"},
                            "video_id": "$$step.video_id",
                            "timestamp": "$$step.timestamp",
                            "visual_caption": "$$step.visual_caption",
                            "audio_transcript": "$$step.audio_transcript",
                            "ocr_text": "$$step.ocr_text",
                            "dissonance_score": "$$step.dissonance_score",
                            "step_index": "$$step.step_index",
                            "steps_backward": "$$step.steps_backward",
                        }
                    }
                },
                "step_chain": {
                    "$map": {
                        "input": "$step_chain",
                        "as": "step",
                        "in": {
                            "step_id": {"$toString": "$$step._id"},
                            "video_id": "$$step.video_id",
                            "timestamp": "$$step.timestamp",
                            "visual_caption": "$$step.visual_caption",
                            "audio_transcript": "$$step.audio_transcript",
                            "ocr_text": "$$step.ocr_text",
                            "dissonance_score": "$$step.dissonance_score",
                            "step_index": "$$step.step_index",
                            "is_step_boundary": "$$step.is_step_boundary",
                            "steps_backward": "$$step.steps_backward",
                        }
                    }
                },
                "following_step": {
                    "$let": {
                        "vars": {"step": {"$arrayElemAt": ["$following_step_docs", 0]}},
                        "in": {
                            "$cond": [
                                {"$ifNull": ["$$step._id", False]},
                                {
                                    "step_id": {"$toString": "$$step._id"},
                                    "video_id": "$$step.video_id",
                                    "timestamp": "$$step.timestamp",
                                    "visual_caption": "$$step.visual_caption",
                                    "audio_transcript": "$$step.audio_transcript",
                                    "step_index": "$$step.step_index",
                                },
                                None,
                            ]
                        }
                    }
                },
            }
        }
    ]

    results = list(db_collection.aggregate(pipeline))
    for result in results:
        result["prerequisite_steps"] = sorted(
            result.get("prerequisite_steps") or [],
            key=lambda step: step.get("steps_backward", 0),
            reverse=True,
        )
        result["step_chain"] = sorted(
            result.get("step_chain") or [],
            key=lambda step: step.get("steps_backward", 0),
            reverse=True,
        )
    return results


def _dedupe_steps(steps):
    ordered = []
    seen = set()
    for step in sorted(steps, key=lambda item: float(item.get("timestamp") or 0)):
        key = (step.get("video_id"), round(float(step.get("timestamp") or 0), 3))
        if key in seen:
            continue
        seen.add(key)
        ordered.append(step)
    return ordered[:8]


def _as_step(raw, video_id):
    return {
        "video_id": raw.get("video_id") or video_id,
        "timestamp": raw.get("timestamp") or 0,
        "visual_caption": raw.get("visual_caption") or "",
        "audio_transcript": raw.get("audio_transcript") or "",
        "ocr_text": raw.get("ocr_text"),
        "dissonance_score": raw.get("dissonance_score"),
        "step_index": raw.get("step_index"),
    }


def _collect_video_steps(hit):
    video_id = hit.get("video_id")
    collected = [_as_step(hit, video_id)]
    for step in hit.get("step_chain") or []:
        collected.append(_as_step(step, video_id))
    for step in hit.get("prerequisite_steps") or []:
        collected.append(_as_step(step, video_id))
    following = hit.get("following_step")
    if following:
        collected.append(_as_step(following, video_id))
    return _dedupe_steps(collected)


def _difference_summary(step_number, steps):
    pieces = []
    for step in steps:
        caption = (step.get("visual_caption") or step.get("audio_transcript") or "a different action").strip()
        pieces.append(f"{caption} {citation_token(step['video_id'], step.get('timestamp'))}")
    if len(pieces) == 1:
        return f"At step {step_number}, only one video still shows the procedure: {pieces[0]}."
    if len(pieces) == 2:
        return f"At step {step_number}, the videos diverge: {pieces[0]} while another shows {pieces[1]}."
    return f"At step {step_number}, the videos diverge: {'; '.join(pieces)}."


def build_comparison(user_query: str, limit: int = 5):
    """Align retrieved steps across videos and describe where the procedures differ."""
    hits = query_video_graph(user_query, limit=max(limit, 5))
    grouped = {}
    for hit in hits:
        video_id = hit.get("video_id") or "unknown"
        bucket = grouped.setdefault(video_id, {
            "video_id": video_id,
            "title": video_id,
            "score": 0.0,
            "steps": [],
        })
        score = float(hit.get("search_score") or 0)
        if score >= bucket["score"]:
            bucket["score"] = score
        bucket["steps"].extend(_collect_video_steps(hit))

    videos = []
    for bucket in grouped.values():
        bucket["steps"] = _dedupe_steps(bucket["steps"])
        if bucket["steps"]:
            videos.append(bucket)
    videos.sort(key=lambda video: video["score"], reverse=True)
    videos = videos[:5]

    flat_steps = [step for video in videos for step in video["steps"]]
    vectors = []
    if flat_steps:
        texts = [
            (step.get("visual_caption") or step.get("audio_transcript") or "silence")
            for step in flat_steps
        ]
        vectors = list(embedder.encode(texts))
    vector_at = {id(step): vectors[index] for index, step in enumerate(flat_steps)}

    differences = []
    citations = []
    seen_citations = set()

    def remember(step):
        key = (step.get("video_id"), round(float(step.get("timestamp") or 0), 3))
        if key in seen_citations:
            return
        seen_citations.add(key)
        citations.append({
            "video_id": step.get("video_id"),
            "timestamp": step.get("timestamp") or 0,
            "label": step.get("visual_caption") or step.get("audio_transcript") or "",
            "token": citation_token(step.get("video_id"), step.get("timestamp")),
        })

    for video in videos:
        if video["steps"]:
            remember(video["steps"][0])

    if len(videos) >= 2:
        longest = max(len(video["steps"]) for video in videos)
        for index in range(longest):
            present = [video["steps"][index] for video in videos if index < len(video["steps"])]
            missing = len(present) != len(videos)
            diverged = False
            if len(present) >= 2:
                base = vector_at.get(id(present[0]))
                diverged = any(
                    _cosine(base, vector_at.get(id(other))) < _DIVERGENCE_THRESHOLD
                    for other in present[1:]
                )
            if not (missing or diverged):
                continue
            for step in present:
                remember(step)
            differences.append({
                "step_index": index + 1,
                "summary": _difference_summary(index + 1, present),
                "videos": present,
            })
    elif len(videos) == 1:
        for index, step in enumerate(videos[0]["steps"]):
            remember(step)
            differences.append({
                "step_index": index + 1,
                "summary": _difference_summary(index + 1, [step]),
                "videos": [step],
            })

    if videos and not differences:
        opening = [video["steps"][0] for video in videos if video["steps"]]
        for step in opening:
            remember(step)
        differences.append({
            "step_index": 1,
            "summary": "The opening steps describe the same action. " + " ".join(
                citation_token(step["video_id"], step.get("timestamp")) for step in opening
            ),
            "videos": opening,
        })

    return {
        "query": user_query,
        "videos": videos,
        "differences": differences,
        "citations": citations,
    }
