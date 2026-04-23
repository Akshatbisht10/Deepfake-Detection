"""
ST-ViT Single-Video Prediction Script.

Usage:
    python predict_video_st_vit.py path/to/video.mp4
    python predict_video_st_vit.py path/to/video.mp4 --model best_st_vit_model.pth
"""

import argparse
import os

import cv2
import torch
from torchvision import transforms
from PIL import Image

from src.models.st_vit_model import ST_ViT

# ─── Configuration ───────────────────────────────────────────────────────────
SEQ_LEN = 20
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def extract_faces_from_video(video_path, seq_len=SEQ_LEN):
    """
    Extract exactly `seq_len` evenly spaced face crops from the video.
    Returns a list of PIL Images.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise ValueError("Video has no frames.")

    # Evenly spaced frame indices
    step = max(1, frame_count // seq_len)
    indices = set(i * step for i in range(seq_len) if i * step < frame_count)

    extracted_faces = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in indices:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

                margin = int(w * 0.2)
                x_m = max(0, x - margin)
                y_m = max(0, y - margin)
                w_m = min(frame.shape[1] - x_m, w + 2 * margin)
                h_m = min(frame.shape[0] - y_m, h + 2 * margin)

                if w_m > 0 and h_m > 0:
                    face = frame[y_m : y_m + h_m, x_m : x_m + w_m]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    extracted_faces.append(Image.fromarray(face))

        frame_idx += 1

        if len(extracted_faces) >= seq_len:
            break

    cap.release()
    return extracted_faces


def predict_video(video_path, model_path):
    """Load the ST-ViT model and predict on a single video."""
    print(f"Loading ST-ViT model from {model_path}...")
    model = ST_ViT(seq_len=SEQ_LEN, freeze_cnn=False).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    print(f"Extracting {SEQ_LEN} face frames from video...")
    faces = extract_faces_from_video(video_path)

    if len(faces) == 0:
        print("Error: Could not detect any faces in the video.")
        return

    print(f"Successfully extracted {len(faces)} faces.")

    # Preprocess frames
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    processed_frames = [transform(face) for face in faces]

    # Pad with zeros if not enough faces
    if len(processed_frames) < SEQ_LEN:
        pad = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        while len(processed_frames) < SEQ_LEN:
            processed_frames.append(pad)

    # Stack: (1, T, C, H, W)
    sequence = torch.stack(processed_frames).unsqueeze(0).to(DEVICE)

    print("Running ST-ViT inference...")
    with torch.no_grad():
        logits = model(sequence)
        probability = torch.sigmoid(logits).item()

    print("\n" + "=" * 50)
    print(f"  ST-ViT Deepfake Detection Result")
    print(f"  Video : {os.path.basename(video_path)}")
    print(f"  Fake Probability : {probability:.4f} ({probability * 100:.2f}%)")

    if probability > 0.5:
        print(f"  Prediction : FAKE 🛑")
    else:
        print(f"  Prediction : REAL ✅")

    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test a video with the ST-ViT Deepfake Detection model."
    )
    parser.add_argument("video_path", help="Path to the video file to analyze")
    parser.add_argument(
        "--model",
        default="best_st_vit_model.pth",
        help="Path to the trained .pth model file (default: best_st_vit_model.pth)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found.")
    elif not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
    else:
        predict_video(args.video_path, args.model)
