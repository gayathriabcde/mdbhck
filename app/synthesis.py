"""Turn computed step differences into comparative prose with citation tokens intact."""

import json
import urllib.error
import urllib.request

from app.utils import citation_token


def template_narrative(comparison: dict) -> str:
    """Fluent comparison used on its own, and as the fallback when no LLM is configured."""
    videos = comparison.get("videos") or []
    query = comparison.get("query") or "this request"
    if not videos:
        return (
            "No indexed steps matched this query yet. "
            "Paste a video URL or run a YouTube search, then ask again once ingestion finishes."
        )

    if len(videos) == 1:
        video = videos[0]
        sentences = [f"One indexed video covers {query}."]
        for index, step in enumerate(video.get("steps") or [], start=1):
            caption = (step.get("visual_caption") or step.get("audio_transcript") or "an unlabeled moment").strip()
            token = citation_token(step["video_id"], step.get("timestamp"))
            sentences.append(f"Step {index} shows {caption} {token}.")
        return " ".join(sentences)

    sentences = [f"Comparing {len(videos)} indexed videos for {query}."]
    for video in videos:
        steps = video.get("steps") or []
        if not steps:
            continue
        opening = steps[0]
        caption = (opening.get("visual_caption") or opening.get("audio_transcript") or "the opening step").strip()
        token = citation_token(opening["video_id"], opening.get("timestamp"))
        sentences.append(f"Video {opening['video_id']} begins with {caption} {token}.")

    differences = comparison.get("differences") or []
    if differences:
        for difference in differences:
            summary = (difference.get("summary") or "").strip()
            if summary:
                sentences.append(summary)
    else:
        sentences.append("The aligned steps describe the same actions, so there is no material procedural difference.")
    return " ".join(sentences)


def preserve_citations(text: str, citations: list) -> str:
    """Append any citation token the model dropped so timestamps stay clickable."""
    narrative = text or ""
    missing = []
    for citation in citations or []:
        token = citation.get("token") or citation_token(citation.get("video_id"), citation.get("timestamp"))
        if token not in narrative:
            missing.append(token)
    if not missing:
        return narrative.strip()
    suffix = " Sources: " + " ".join(missing) + "."
    return (narrative.strip() + suffix).strip()


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 45) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _build_prompt(comparison: dict) -> str:
    citations = comparison.get("citations") or []
    token_lines = "\n".join(
        f"- {item.get('token')} — {item.get('label') or 'moment'}"
        for item in citations
    ) or "- (no citations)"
    differences = comparison.get("differences") or []
    difference_lines = "\n".join(
        f"- {item.get('summary')}" for item in differences if item.get("summary")
    ) or "- No material differences were computed."
    return (
        "Write a short comparative explanation of these how-to videos.\n"
        "Use the computed differences below. Do not invent steps, timestamps, or video ids.\n"
        "Whenever you mention a moment, paste its citation token exactly, braces included.\n\n"
        f"Query: {comparison.get('query')}\n\n"
        f"Citation tokens you may use:\n{token_lines}\n\n"
        f"Computed differences:\n{difference_lines}\n"
    )


def _call_openai(prompt: str) -> str:
    from core.config import Config

    payload = _post_json(
        "https://api.openai.com/v1/chat/completions",
        {
            "model": Config.OPENAI_MODEL,
            "temperature": 0.2,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You write fluent procedural comparisons. "
                        "Keep every {{cite:VIDEO_ID:SECONDS}} token unchanged."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        },
        {
            "Authorization": f"Bearer {Config.OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    return payload["choices"][0]["message"]["content"]


def _call_anthropic(prompt: str) -> str:
    from core.config import Config

    payload = _post_json(
        "https://api.anthropic.com/v1/messages",
        {
            "model": Config.ANTHROPIC_MODEL,
            "max_tokens": 800,
            "temperature": 0.2,
            "system": "You write fluent procedural comparisons. Keep every {{cite:VIDEO_ID:SECONDS}} token unchanged.",
            "messages": [{"role": "user", "content": prompt}],
        },
        {
            "x-api-key": Config.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
    )
    blocks = payload.get("content") or []
    return "".join(block.get("text", "") for block in blocks if block.get("type") == "text")


def _call_ollama(prompt: str) -> str:
    from core.config import Config

    host = Config.OLLAMA_HOST.rstrip("/")
    payload = _post_json(
        f"{host}/api/chat",
        {
            "model": Config.OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": "You write fluent procedural comparisons. Keep every {{cite:VIDEO_ID:SECONDS}} token unchanged.",
                },
                {"role": "user", "content": prompt},
            ],
        },
        {"Content-Type": "application/json"},
    )
    message = payload.get("message") or {}
    return message.get("content") or ""


def synthesize_comparison(comparison: dict) -> dict:
    """Prefer a configured LLM, and always keep the computed citation tokens."""
    from core.config import Config

    narrative = ""
    provider = "template"
    try:
        if Config.OPENAI_API_KEY:
            narrative = _call_openai(_build_prompt(comparison))
            provider = "openai"
        elif Config.ANTHROPIC_API_KEY:
            narrative = _call_anthropic(_build_prompt(comparison))
            provider = "anthropic"
        elif Config.OLLAMA_HOST:
            narrative = _call_ollama(_build_prompt(comparison))
            provider = "ollama"
    except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError, ValueError) as exc:
        print(f"[Analytics] LLM synthesis failed, using the citation template: {exc}")
        narrative = ""
        provider = "template"

    if not (narrative or "").strip():
        narrative = template_narrative(comparison)
        provider = "template"

    return {
        "query": comparison.get("query"),
        "narrative": preserve_citations(narrative, comparison.get("citations") or []),
        "provider": provider,
        "differences": comparison.get("differences") or [],
        "citations": comparison.get("citations") or [],
        "videos": [
            {
                "video_id": video.get("video_id"),
                "score": video.get("score"),
                "step_count": len(video.get("steps") or []),
            }
            for video in comparison.get("videos") or []
        ],
    }


def run_analytics(user_query: str, limit: int = 5) -> dict:
    from app.retrieval import build_comparison

    comparison = build_comparison(user_query, limit=limit)
    return synthesize_comparison(comparison)
