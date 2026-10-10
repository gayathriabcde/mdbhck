import os
import yt_dlp

VIDEO_DIR = "./raw_videos"

# A curated list of specific procedural videos (Cooking, IT, DIY, Automotive)
# Replace these URLs with videos that have clear physical steps
VIDEO_URLS = [
    "https://www.youtube.com/watch?v=__izua1kKeI",  # Programming basics Python/C# (variables, loops, functions demo)
    "https://www.youtube.com/watch?v=4JzDttgdILQ",  # Creative coding full course (Processing sketches, mouse interaction)
    "https://www.youtube.com/watch?v=W8KRzm-HUcc",  # Python Django tutorial (models, views, templates step-by-step)
    "https://www.youtube.com/watch?v=TRCQus73lAc",  # JavaScript project build (DOM manipulation, event listeners)
    "https://www.youtube.com/watch?v=3OtM64iWvsI",  # React hooks tutorial (useState, useEffect screen demo),

    # Physical DIY Projects (5) - Hands-on tools, assembly shown
    "https://www.youtube.com/watch?v=r1vdNNc-drQ",  # Simple shelf build (wood cutting, screwing, wall mount)
    "https://www.youtube.com/watch?v=eWNDNeaHyMM",  # Corner shelf under $30 (angle cuts, brackets install)
    "https://www.youtube.com/watch?v=EOGQZDV5TW0",  # Floating shelves (leveling, anchors, wood finish)
    "https://www.youtube.com/watch?v=L2R0qKAxdzc",  # Paint room complete (taping, rolling, cleanup)
    "https://www.youtube.com/watch?v=bLbUIevOxzY",  # Room painting beginners (priming, edging technique),

    # Cooking Step-by-Step (5) - Physical chopping, mixing, cooking
    "https://www.youtube.com/watch?v=wh3cs85ow1A",  # Bread from scratch (kneading 8 mins, proofing, baking)
    "https://www.youtube.com/watch?v=mVXvoQOMS2E",  # Pro onion chopping (dice, mince, no tears method)
    "https://www.youtube.com/watch?v=dCGS067s0zo",  # Gordon Ramsay onion dicing (sharp knife, board stability)
    "https://www.youtube.com/watch?v=tmIkZkT52Fo",  # 5-step dinners (prep, sear, sauce, plate)
    "https://www.youtube.com/watch?v=oCTZJ4YoAb0"  # Easy home recipes (ingredient measure, cook sequence)
]

def download_youtube_videos(urls, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    ydl_opts = {
        # Force mp4 format for cv2 compatibility in your dataset_builder
        'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4][height<=720]',
        'outtmpl': f'{output_path}/%(id)s.%(ext)s',
        # Ignore age-restricted or unavailable videos gracefully
        'ignoreerrors': True,
        'quiet': False,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        print(f"Downloading {len(urls)} videos to {output_path}...")
        ydl.download(urls)
        print("✅ Download complete.")

if __name__ == "__main__":
    download_youtube_videos(VIDEO_URLS, VIDEO_DIR)