import os
import subprocess
import glob
import numpy as np
from PIL import Image
from bson.objectid import ObjectId
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, AutoModelForCausalLM
from core.database import db_collection
import yt_dlp

from worker.dissonance_inference import get_dissonance_score

# --- Helper Functions ---

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


def extract_frames_fast(video_path, output_dir="temp_frames", fps=0.5):
    """Bypasses OpenCV. Uses FFmpeg C-binaries for blazingly fast extraction."""
    os.makedirs(output_dir, exist_ok=True)
    # Clear existing frames if any
    for f in glob.glob(f"{output_dir}/*.jpg"):
        os.remove(f)

    print(f"-> [FFmpeg] Extracting frames at {fps} FPS...")
    command = [
        'ffmpeg', '-i', video_path,
        '-vf', f'fps={fps}',
        '-q:v', '2',  # High quality jpeg
        f'{output_dir}/frame_%04d.jpg',
        '-loglevel', 'error', '-y'
    ]
    subprocess.run(command, check=True)

    # Return sorted list of frame file paths
    return sorted(glob.glob(f"{output_dir}/*.jpg"))


def get_temporal_audio(whisper_segments, frame_time, window=2.0):
    """Extracts only the words spoken within +/- 'window' seconds of the frame."""
    relevant_text = []
    for seg in whisper_segments:
        # If the audio segment overlaps with our frame's time window
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
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


# --- The Main Pipeline ---

def process_video_pipeline(youtube_url: str):
    video_path = None
    frames_dir = "temp_frames"

    try:
        # 1. Ingestion
        video_path, video_id = download_video(youtube_url)
        frame_files = extract_frames_fast(video_path, frames_dir, fps=0.5)
        num_frames = len(frame_files)

        # 2. Load Models
        print("-> Loading Models...")
        whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
        embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
        vlm_processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True)
        vlm_model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', trust_remote_code=True).eval()

        # 3. Audio Extraction & Segmentation
        print("-> Transcribing Audio...")
        segments_generator, _ = whisper_model.transcribe(video_path, vad_filter=True)
        # Convert generator to a list of dicts so we can loop over it multiple times
        audio_segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segments_generator]

        # 4. Generate ObjectIds upfront to establish Graph Edges
        node_ids = [ObjectId() for _ in range(num_frames)]
        nodes_to_insert = []

        # Tracking variables for Visual Delta Deduplication
        prev_float_embedding = None
        prev_reference_id = None

        # 5. The State Graph Construction Loop
        print("-> Building Procedural State Graph...")
        for i, frame_file in enumerate(frame_files):
            current_id = node_ids[i]
            timestamp = i * 2.0  # Because fps=0.5 (1 frame every 2 secs)

            # --- Temporal Audio ---
            frame_audio = get_temporal_audio(audio_segments, timestamp)

            # --- Visual Ingestion ---
            image = Image.open(frame_file).convert("RGB")
            inputs = vlm_processor(text="<MORE_DETAILED_CAPTION>", images=image, return_tensors="pt")
            generated_ids = vlm_model.generate(
                input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                max_new_tokens=1024, do_sample=False, num_beams=3
            )
            generated_text = vlm_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            caption = vlm_processor.post_process_generation(generated_text, task="<MORE_DETAILED_CAPTION>",
                                                            image_size=image.size)
            visual_caption = caption['<MORE_DETAILED_CAPTION>']

            # --- OCR Logic (Placeholder for PaddleOCR) ---
            # If Florence-2 detects text (you can tune this logic later), run PaddleOCR
            # For now, we strictly define the schema field.
            ocr_text = None

            # --- State Unification ---
            unified_text = f"[Visual: {visual_caption}] [Audio Context: {frame_audio}]"
            if ocr_text:
                unified_text += f" [OCR: {ocr_text}]"

            current_float_embedding = embedder.encode(unified_text)

            # --- Tri-Modal Dissonance Calculation ---
            # We must embed the audio and visual separately to calculate the semantic gap
            audio_float_vec = embedder.encode(frame_audio)
            visual_float_vec = embedder.encode(visual_caption)

            # Pass to your newly trained PyTorch model
            calculated_dissonance = get_dissonance_score(audio_float_vec, visual_float_vec)

            # --- Visual Delta Thresholding & Graph Deduplication ---
            vector_to_store = None
            reference_id = None

            similarity = cosine_similarity(current_float_embedding, prev_float_embedding)

            # If similarity > 0.85, the scene hasn't changed. Don't store a new vector.
            if i > 0 and similarity > 0.85:
                vector_to_store = None
                reference_id = prev_reference_id  # Point to the parent visual node
            else:
                vector_to_store = quantize_to_int8(current_float_embedding)
                reference_id = current_id  # This node is now the new parent reference

                # Update tracking variables
                prev_float_embedding = current_float_embedding
                prev_reference_id = current_id

            # --- Construct the Document Schema ---
            node = {
                "_id": current_id,
                "video_id": video_id,
                "node_index": i,
                "timestamp": timestamp,
                "visual_caption": visual_caption,
                "ocr_text": ocr_text,
                "audio_transcript": frame_audio,
                "unified_text": unified_text,
                "vector_int8": vector_to_store,
                "dissonance_score": calculated_dissonance,
                "graph_edges": {
                    "prev_node_id": node_ids[i - 1] if i > 0 else None,
                    "next_node_id": node_ids[i + 1] if i < num_frames - 1 else None,
                    "visual_reference_id": reference_id if reference_id != current_id else None
                }
            }

            nodes_to_insert.append(node)

        # 6. Database Persistence
        print(f"-> Inserting {len(nodes_to_insert)} linked nodes into MongoDB Atlas...")
        if nodes_to_insert:
            db_collection.insert_many(nodes_to_insert)

        # Cleanup
        os.remove(video_path)
        for f in frame_files:
            os.remove(f)

        return len(nodes_to_insert)

    except Exception as e:
        # Emergency Cleanup
        if video_path and os.path.exists(video_path):
            os.remove(video_path)
        raise e