"""
Preprocess videos into ordered face-frame sequences for CNN+LSTM training.

Output structure:
  dataset_sequences/
    train/real/video001/  -> 0.jpg, 1.jpg, ..., 19.jpg
    train/fake/video050/  -> 0.jpg, 1.jpg, ...
    val/real/...
    test/fake/...

Each video gets its own folder with sequentially numbered face crops.
Split is done at the VIDEO level (70/15/15) to avoid data leakage.
"""

import os
import gc
import cv2
import random
from tqdm import tqdm

# ─── Configuration ───────────────────────────────────────────────────────────
REAL_VIDEO_DIR = "DFD_original sequences"
FAKE_VIDEO_DIR = "DFD_manipulated_sequences"
OUTPUT_DIR = "dataset_sequences"
IMG_SIZE = 224          # ResNet-18 default input size
SEQ_LEN = 20           # Frames per video sequence
MAX_VIDEOS = 150       # 150 per class, balanced
SPLIT_RATIO = (0.8, 0.1, 0.1)   # train / val / test
SEED = 42

# Haar Cascade face detector
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def collect_videos(directory):
    """Recursively collect all video files from a directory."""
    videos = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith((".mp4", ".avi", ".mov")):
                videos.append(os.path.join(root, f))
    return sorted(videos)


def extract_face_sequence(video_path, output_folder, seq_len=SEQ_LEN):
    """
    Extract a sequence of `seq_len` face crops from a single video.
    Uses frame seeking (cap.set) instead of reading every frame to save memory.
    Returns True if at least 1 face frame was saved.
    """
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_count <= 0:
            cap.release()
            return False

        # Compute evenly spaced frame indices to sample
        step = max(1, frame_count // seq_len)
        target_indices = [i * step for i in range(seq_len) if i * step < frame_count]

        os.makedirs(output_folder, exist_ok=True)
        saved = 0

        for target_idx in target_indices:
            # Seek directly to the target frame (avoids reading all frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                # Take the largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

                # Add 20 % margin
                margin = int(w * 0.2)
                x_m = max(0, x - margin)
                y_m = max(0, y - margin)
                w_m = min(frame.shape[1] - x_m, w + 2 * margin)
                h_m = min(frame.shape[0] - y_m, h + 2 * margin)

                if w_m > 0 and h_m > 0:
                    face = frame[y_m : y_m + h_m, x_m : x_m + w_m]
                    face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
                    cv2.imwrite(
                        os.path.join(output_folder, f"{saved}.jpg"), face
                    )
                    saved += 1

            # Free frame memory immediately
            del frame, gray
            if len(faces) > 0:
                del face

        cap.release()
        return saved > 0

    except Exception as e:
        print(f"\n  [WARN] Skipping {os.path.basename(video_path)}: {e}")
        return False


def split_videos(video_paths):
    """Split a list of video paths into train/val/test lists."""
    random.seed(SEED)
    shuffled = list(video_paths)
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * SPLIT_RATIO[0])
    n_val = int(n * SPLIT_RATIO[1])

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


def process_class(video_dir, label):
    """Process all videos of one class (real/fake) through the full pipeline."""
    all_videos = collect_videos(video_dir)
    if MAX_VIDEOS is not None:
        all_videos = all_videos[:MAX_VIDEOS]

    print(f"\n[{label.upper()}] Found {len(all_videos)} videos")
    splits = split_videos(all_videos)

    for split_name, videos in splits.items():
        print(f"  {split_name}: {len(videos)} videos")
        for vid_path in tqdm(videos, desc=f"{split_name}/{label}"):
            vid_name = os.path.splitext(os.path.basename(vid_path))[0]
            out_folder = os.path.join(OUTPUT_DIR, split_name, label, vid_name)
            extract_face_sequence(vid_path, out_folder)
            gc.collect()  # Free memory after each video


def main():
    print("=" * 60)
    print("  CNN+LSTM Sequence Preprocessing")
    print("=" * 60)

    process_class(REAL_VIDEO_DIR, "real")
    process_class(FAKE_VIDEO_DIR, "fake")

    # Print summary
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            d = os.path.join(OUTPUT_DIR, split, label)
            if os.path.exists(d):
                count = len(os.listdir(d))
                print(f"  {split}/{label}: {count} video sequences")
    print("=" * 60)
    print("Done! Sequences saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
