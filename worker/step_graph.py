"""Infer procedural step edges from raw video signals.

No pre-labeled CSV is required. A new step starts when the frame sequence
shows a visual feature shift, a VLM action change, or a transcript pause.
"""

import re

STEP_MIN_FRAMES = 2
VISUAL_SHIFT_THRESHOLD = 0.70
STRONG_SHIFT_THRESHOLD = 0.55
CAPTION_JACCARD_SHIFT = 0.28
TRANSCRIPT_GAP_SECONDS = 0.8

_ACTION_VERBS = {
    "add", "align", "attach", "bend", "clamp", "close", "connect", "cover",
    "cut", "drill", "drop", "fasten", "fill", "flip", "fold", "glue", "hold",
    "insert", "install", "lift", "loosen", "measure", "mix", "mount", "open",
    "peel", "place", "pour", "press", "pull", "push", "remove", "rotate",
    "sand", "screw", "seal", "set", "solder", "spread", "stick", "tighten",
    "turn", "twist", "unwrap", "wire", "wrap",
}

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def _tokens(text):
    return set(_TOKEN_RE.findall((text or "").lower()))


def _jaccard(left, right):
    a = _tokens(left)
    b = _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _cosine(vec_a, vec_b):
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


def _actions(text):
    return _tokens(text) & _ACTION_VERBS


def _action_changed(previous_caption, caption):
    previous = _actions(previous_caption)
    current = _actions(caption)
    if not current:
        return False
    return current != previous and not (previous & current)


def _segment_starts_in_window(segments, t_prev, t_curr, min_gap=TRANSCRIPT_GAP_SECONDS):
    """True when speech starts inside the frame gap after a noticeable pause."""
    if not segments:
        return False
    for segment in segments:
        start = float(segment.get("start", 0.0))
        if not (t_prev < start <= t_curr):
            continue
        previous_ends = [
            float(item.get("end", 0.0))
            for item in segments
            if float(item.get("end", 0.0)) <= start + 1e-3
        ]
        if not previous_ends:
            return True
        if start - max(previous_ends) >= min_gap:
            return True
    return False


def detect_step_boundaries(
    nodes,
    embeddings,
    speech_segments=None,
    transcript_segments=None,
    min_frames=STEP_MIN_FRAMES,
    visual_shift=VISUAL_SHIFT_THRESHOLD,
    strong_shift=STRONG_SHIFT_THRESHOLD,
    caption_jaccard=CAPTION_JACCARD_SHIFT,
):
    """Return indexes of frames that begin a procedural step."""
    if not nodes:
        return []

    boundaries = [0]
    speech_segments = speech_segments or []
    transcript_segments = transcript_segments or []

    for index in range(1, len(nodes)):
        if index - boundaries[-1] < min_frames:
            continue

        previous = nodes[index - 1]
        current = nodes[index]
        similarity = _cosine(
            embeddings[index] if index < len(embeddings) else None,
            embeddings[index - 1] if index - 1 < len(embeddings) else None,
        )
        overlap = _jaccard(previous.get("visual_caption"), current.get("visual_caption"))
        actions_changed = _action_changed(previous.get("visual_caption"), current.get("visual_caption"))
        transcript_break = _segment_starts_in_window(
            speech_segments, previous.get("timestamp", 0.0), current.get("timestamp", 0.0)
        ) or _segment_starts_in_window(
            transcript_segments, previous.get("timestamp", 0.0), current.get("timestamp", 0.0)
        )

        strong_visual_shift = similarity < strong_shift
        moderate_visual_shift = similarity < visual_shift
        caption_shift = overlap < caption_jaccard

        if (
            strong_visual_shift
            or (moderate_visual_shift and actions_changed)
            or (moderate_visual_shift and caption_shift)
            or (transcript_break and overlap < 0.45)
        ):
            boundaries.append(index)

    return boundaries


def assign_step_edges(nodes, embeddings, speech_segments=None, transcript_segments=None, **kwargs):
    """Write parent_segment_id and next_step_id onto each node's graph_edges.

    parent_segment_id:
      - step-start frames point at the previous step (None for the first step)
      - later frames in the step point at that step's start frame
    next_step_id:
      - every frame points at the next step's start frame, or None on the last step
    """
    boundaries = detect_step_boundaries(
        nodes,
        embeddings,
        speech_segments=speech_segments,
        transcript_segments=transcript_segments,
        **kwargs,
    )
    if not boundaries:
        return 0

    boundary_ids = [nodes[index]["_id"] for index in boundaries]
    boundary_set = set(boundaries)
    segment_of = [0] * len(nodes)
    pointer = 0
    for index in range(len(nodes)):
        while pointer + 1 < len(boundaries) and index >= boundaries[pointer + 1]:
            pointer += 1
        segment_of[index] = pointer

    for index, node in enumerate(nodes):
        segment = segment_of[index]
        step_start_id = boundary_ids[segment]
        previous_step_id = boundary_ids[segment - 1] if segment > 0 else None
        next_step_id = boundary_ids[segment + 1] if segment + 1 < len(boundary_ids) else None
        is_boundary = index in boundary_set
        edges = node.setdefault("graph_edges", {})
        edges["parent_segment_id"] = previous_step_id if is_boundary else step_start_id
        edges["next_step_id"] = next_step_id
        node["is_step_boundary"] = is_boundary
        node["step_index"] = segment

    return len(boundaries)
