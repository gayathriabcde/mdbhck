import glob
import os
import subprocess

import numpy as np
from bson.objectid import ObjectId
from faster_whisper import WhisperModel
from PIL import Image
from sentence_transformers import SentenceTransformer
import yt_dlp

from core.config import Config
from core.database import db_collection
from worker.dissonance_inference import get_dissonance_score, warm_dissonance_model
from worker.step_graph import assign_step_edges

# Low-rate sampling keeps newly ingested videos moving on CPU.
# 0.25 FPS is one keyframe every 4 seconds. Override with KEYFRAME_FPS.
KEYFRAME_FPS = Config.KEYFRAME_FPS
BLIP_MAX_NEW_TOKENS = 30
OCR_MAX_SIDE = 480
VECTOR_DEDUP_SIMILARITY = 0.85

_runtime_models = {}


def _report(progress_callback, stage, percent, detail=""):
    print(f"-> [{percent}%] {stage} {detail}".rstrip())
    if not progress_callback:
        return
    try:
        progress_callback(stage, percent, detail)
    except Exception as exc:
        print(f"[Pipeline] Progress callback failed: {exc}")


def download_video(youtube_url, output_dir="temp_vids"):
    """Downloads video at lowest acceptable quality."""
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'format': 'worstvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
        'quiet': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=True)
        return os.path.join(output_dir, f"{info['id']}.mp4"), info['id']


def extract_frames_fast(video_path, output_dir="temp_frames", fps=KEYFRAME_FPS):
    """Bypasses OpenCV. Uses FFmpeg for low-rate keyframe extraction."""
    os.makedirs(output_dir, exist_ok=True)
    for frame_path in glob.glob(f"{output_dir}/*.jpg"):
        os.remove(frame_path)

    print(f"-> [FFmpeg] Extracting keyframes at {fps} FPS...")
    command = [
        'ffmpeg', '-i', video_path,
        '-vf', f'fps={fps}',
        '-q:v', '5',
        f'{output_dir}/frame_%04d.jpg',
        '-loglevel', 'error', '-y'
    ]
    subprocess.run(command, check=True)
    return sorted(glob.glob(f"{output_dir}/*.jpg"))


def get_temporal_audio(whisper_segments, frame_time, window=2.0):
    """Extracts only the words spoken within +/- 'window' seconds of the frame."""
    relevant_text = []
    for seg in whisper_segments:
        if seg['start'] <= (frame_time + window) and seg['end'] >= (frame_time - window):
            relevant_text.append(seg['text'].strip())

    return " ".join(relevant_text) if relevant_text else "[SILENCE]"


def quantize_to_int8(embedding: np.ndarray) -> list:
    """Storage Hack #1: Cast float32 vectors to int8."""
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding.astype(np.int8).tolist()
    normalized_vec = embedding / norm
    int8_vec = np.round(normalized_vec * 127).astype(np.int8)
    return int8_vec.tolist()


def cosine_similarity(vec_a, vec_b):
    """Calculates semantic distance between two vectors."""
    if vec_a is None or vec_b is None:
        return 0.0
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def run_silero_vad(media_path: str) -> list:
    """Return speech spans in seconds using Silero VAD via faster-whisper."""
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    sampling_rate = 16000
    audio = decode_audio(media_path, sampling_rate=sampling_rate)
    raw_spans = get_speech_timestamps(
        audio,
        vad_options=VadOptions(min_silence_duration_ms=500, speech_pad_ms=200),
        sampling_rate=sampling_rate,
    )
    duration_s = len(audio) / float(sampling_rate) if len(audio) else 0.0
    spans = []
    for span in raw_spans:
        start = float(span["start"])
        end = float(span["end"])
        # faster-whisper returns sample indexes. Seconds are much smaller than the sample count.
        if duration_s and end > duration_s * 1.5:
            start /= sampling_rate
            end /= sampling_rate
        spans.append({"start": round(start, 3), "end": round(end, 3)})
    return spans


def _get_whisper():
    if "whisper" not in _runtime_models:
        _runtime_models["whisper"] = WhisperModel("tiny", device="cpu", compute_type="int8")
    return _runtime_models["whisper"]


def _get_embedder():
    if "embedder" not in _runtime_models:
        _runtime_models["embedder"] = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
    return _runtime_models["embedder"]


def _get_blip():
    if "blip" not in _runtime_models:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor

        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        model.eval()
        _runtime_models["blip"] = (processor, model, torch)
    return _runtime_models["blip"]


def _get_easyocr():
    if "easyocr" not in _runtime_models:
        import easyocr
        _runtime_models["easyocr"] = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _runtime_models["easyocr"]


def caption_frame_blip(image: Image.Image) -> str:
    """Fast BLIP caption. Kept short so CPU ingestion stays responsive."""
    processor, model, torch = _get_blip()
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=BLIP_MAX_NEW_TOKENS)
    caption = processor.decode(generated[0], skip_special_tokens=True).strip()
    return caption or "[no visual caption]"


def _downscale(image: Image.Image, max_side=OCR_MAX_SIDE) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.BILINEAR)


def read_onscreen_text(image: Image.Image) -> str:
    """EasyOCR on a downscaled frame. Empty string when no text is present."""
    reader = _get_easyocr()
    small = _downscale(image)
    lines = reader.readtext(np.array(small), detail=0, paragraph=True)
    text = " ".join(line.strip() for line in lines if line and line.strip())
    return text


def _apply_vector_dedup(nodes):
    """Drop near-duplicate vectors and point those frames at the parent visual node."""
    previous_embedding = None
    previous_reference_id = None
    for index, node in enumerate(nodes):
        embedding = node.get("_float_embedding")
        similarity = cosine_similarity(embedding, previous_embedding)
        if index > 0 and similarity > VECTOR_DEDUP_SIMILARITY:
            node["vector_int8"] = None
            node["graph_edges"]["visual_reference_id"] = previous_reference_id
        else:
            node["vector_int8"] = quantize_to_int8(np.asarray(embedding)) if embedding is not None else None
            node["graph_edges"]["visual_reference_id"] = None
            previous_embedding = embedding
            previous_reference_id = node["_id"]


def process_video_pipeline(youtube_url: str, progress_callback=None):
    """Dynamic ingestion pass: keyframes, Silero VAD, BLIP, EasyOCR, dissonance, step edges."""
    video_path = None
    frame_files = []
    frames_dir = "temp_frames"
    video_id = None

    try:
        _report(progress_callback, "Downloading video...", 5)
        video_path, video_id = download_video(youtube_url)
        _report(progress_callback, "Sampling keyframes...", 15, f"{KEYFRAME_FPS} FPS")
        frame_files = extract_frames_fast(video_path, frames_dir, fps=KEYFRAME_FPS)
        num_frames = len(frame_files)
        if num_frames == 0:
            raise RuntimeError(f"No keyframes extracted for {video_id}")

        _report(progress_callback, "Loading speech, caption, and dissonance models...", 22)
        whisper_model = _get_whisper()
        embedder = _get_embedder()
        _get_blip()
        warm_dissonance_model()

        _report(progress_callback, "Running Silero VAD...", 30)
        try:
            speech_segments = run_silero_vad(video_path)
        except Exception as exc:
            print(f"[Pipeline] Silero VAD failed, continuing with Whisper segments only: {exc}")
            speech_segments = []
        _report(progress_callback, "Running Silero VAD...", 36, f"{len(speech_segments)} speech spans")

        _report(progress_callback, "Transcribing audio...", 40)
        segments_generator, _ = whisper_model.transcribe(video_path, vad_filter=True)
        audio_segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments_generator]

        node_ids = [ObjectId() for _ in range(num_frames)]
        nodes_to_insert = []
        seconds_per_frame = 1.0 / float(KEYFRAME_FPS)

        _report(progress_callback, "Generating BLIP Captions...", 45)
        for index, frame_file in enumerate(frame_files):
            percent = 45 + int(20 * (index + 1) / num_frames)
            timestamp = round(index * seconds_per_frame, 3)
            frame_audio = get_temporal_audio(audio_segments, timestamp, window=max(2.0, seconds_per_frame))
            image = Image.open(frame_file).convert("RGB")
            visual_caption = caption_frame_blip(image)
            unified_text = f"[Visual: {visual_caption}] [Audio Context: {frame_audio}]"
            unified_embedding = embedder.encode(unified_text)
            audio_embedding = embedder.encode(frame_audio)
            visual_embedding = embedder.encode(visual_caption)
            dissonance_score = get_dissonance_score(audio_embedding, visual_embedding)

            _report(
                progress_callback,
                "Generating BLIP Captions...",
                min(percent, 70),
                f"frame {index + 1}/{num_frames}, dissonance {dissonance_score}",
            )

            nodes_to_insert.append({
                "_id": node_ids[index],
                "video_id": video_id,
                "node_index": index,
                "timestamp": timestamp,
                "visual_caption": visual_caption,
                "ocr_text": None,
                "audio_transcript": frame_audio,
                "unified_text": unified_text,
                "vector_int8": None,
                "dissonance_score": dissonance_score,
                "is_step_boundary": False,
                "step_index": 0,
                "graph_edges": {
                    "prev_node_id": node_ids[index - 1] if index > 0 else None,
                    "next_node_id": node_ids[index + 1] if index < num_frames - 1 else None,
                    "visual_reference_id": None,
                    "parent_segment_id": None,
                    "next_step_id": None,
                },
                "_float_embedding": unified_embedding,
                "_frame_file": frame_file,
            })

        _report(progress_callback, "Running EasyOCR...", 72)
        ocr_hits = 0
        try:
            for index, node in enumerate(nodes_to_insert):
                percent = 72 + int(10 * (index + 1) / num_frames)
                image = Image.open(node["_frame_file"]).convert("RGB")
                ocr_text = read_onscreen_text(image) or None
                if ocr_text:
                    ocr_hits += 1
                    node["ocr_text"] = ocr_text
                    node["unified_text"] += f" [OCR: {ocr_text}]"
                    node["_float_embedding"] = embedder.encode(node["unified_text"])
                _report(progress_callback, "Running EasyOCR...", percent, f"frame {index + 1}/{num_frames}")
        except Exception as exc:
            print(f"[Pipeline] EasyOCR unavailable, continuing without on-screen text: {exc}")
            _report(progress_callback, "Running EasyOCR...", 82, "reader unavailable, captions kept")
        else:
            _report(progress_callback, "Running EasyOCR...", 82, f"{ocr_hits} frames with on-screen text")

        for node in nodes_to_insert:
            node.pop("_frame_file", None)

        _report(progress_callback, "Attaching dissonance scores...", 84)
        _apply_vector_dedup(nodes_to_insert)

        _report(progress_callback, "Building step-boundary graph...", 90)
        embeddings = [node.get("_float_embedding") for node in nodes_to_insert]
        step_count = assign_step_edges(
            nodes_to_insert,
            embeddings,
            speech_segments=speech_segments,
            transcript_segments=audio_segments,
        )
        for node in nodes_to_insert:
            node.pop("_float_embedding", None)

        _report(progress_callback, "Writing nodes to MongoDB...", 96, f"{step_count} procedural steps")
        db_collection.delete_many({"video_id": video_id})
        if nodes_to_insert:
            db_collection.insert_many(nodes_to_insert)

        _report(progress_callback, "Ingestion complete.", 100, f"{len(nodes_to_insert)} nodes")
        return {
            "nodes_inserted": len(nodes_to_insert),
            "video_id": video_id,
            "step_count": step_count,
        }

    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        for frame_path in frame_files:
            if os.path.exists(frame_path):
                os.remove(frame_path)
