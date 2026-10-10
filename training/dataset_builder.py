import os
import cv2
import pandas as pd
import torch
from faster_whisper import WhisperModel
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
VIDEO_DIR = "./raw_videos"  # Put your 10-15 .mp4 files here
OUTPUT_CSV = "./dissonance_dataset_raw.csv"

# Hardware configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# ---------------------------------------------------------
# Model Initialization
# ---------------------------------------------------------
print(f"Loading models on {DEVICE}...")

# 1. Load Faster-Whisper (ASR)
# 'small' or 'base' is usually fine for clear tutorial audio
whisper_model = WhisperModel("small", device=DEVICE, compute_type="float16" if DEVICE == "cuda" else "int8")

# 2. Load Florence-2 (VLM)
model_id = "microsoft/Florence-2-base"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
vlm_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=TORCH_DTYPE,
    trust_remote_code=True
).to(DEVICE)


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def extract_frame_at_time(video_path, time_in_seconds):
    """Extracts a single PIL Image from a video at a specific timestamp."""
    vidcap = cv2.VideoCapture(video_path)

    # Calculate exact frame number based on fps
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_no = int(time_in_seconds * fps)

    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    success, image = vidcap.read()
    vidcap.release()

    if success:
        # Convert BGR (OpenCV) to RGB (PIL)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(image_rgb)
    return None


def generate_visual_caption(image):
    """Passes the image to Florence-2 to get a dense caption."""
    # Using the detailed caption prompt for better physical state descriptions
    prompt = "<MORE_DETAILED_CAPTION>"

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE, TORCH_DTYPE)

    generated_ids = vlm_model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3
    )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    # Clean up the output string
    parsed_answer = processor.post_process_generation(generated_text, task=prompt,
                                                      image_size=(image.width, image.height))
    return parsed_answer[prompt]


# ---------------------------------------------------------
# Main Execution Loop
# ---------------------------------------------------------
def main():
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR)
        print(f"Created directory {VIDEO_DIR}. Please add .mp4 files and rerun.")
        return

    dataset_rows = []
    video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]

    print(f"Found {len(video_files)} videos. Starting pipeline...")

    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        video_id = os.path.splitext(video_file)[0]
        print(f"\nProcessing: {video_id}")

        # Step 1: Transcribe the audio track
        # vad_filter=True prevents Whisper from hallucinating during silent parts
        segments, info = whisper_model.transcribe(video_path, beam_size=5, vad_filter=True)

        for segment in segments:
            start_time = segment.start
            end_time = segment.end
            audio_text = segment.text.strip()

            # Skip extremely short utterances like "um" or "so"
            if len(audio_text.split()) < 3:
                continue

            # Step 2: Find the midpoint of the speech
            midpoint_time = start_time + ((end_time - start_time) / 2)

            # Step 3: Extract the visual frame at that midpoint
            frame = extract_frame_at_time(video_path, midpoint_time)

            if frame is None:
                continue

            # Step 4: Generate the visual caption
            visual_text = generate_visual_caption(frame)

            print(f"[{start_time:.2f}s - {end_time:.2f}s] Audio: {audio_text[:30]}... | Visual: {visual_text[:30]}...")

            # Add to our data collection
            dataset_rows.append({
                "video_id": video_id,
                "start_time": round(start_time, 2),
                "end_time": round(end_time, 2),
                "audio_transcript": audio_text,
                "visual_caption": visual_text,
                "is_dissonant": ""  # LEAVE BLANK FOR MANUAL ANNOTATION
            })

    # Step 5: Export to CSV
    df = pd.DataFrame(dataset_rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Pipeline complete. Saved {len(df)} rows to {OUTPUT_CSV}.")
    print("Next Step: Open the CSV and fill the 'is_dissonant' column with 1 or 0.")


if __name__ == "__main__":
    main()
