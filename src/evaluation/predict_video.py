import argparse
import os
import cv2
import torch
from torchvision import transforms
from PIL import Image

# Import the model architecture from the training script
from src.training.train_cnn_lstm import CNN_LSTM, SEQ_LEN, IMG_SIZE

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def extract_faces_from_video(video_path, seq_len=SEQ_LEN):
    """
    Extracts exactly `seq_len` evenly spaced face crops from the video.
    Returns a list of PIL Images.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise ValueError("Video has no frames.")

    # Calculate frame indices to sample evenly
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
                # Get the largest face
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

                # Add 20% margin
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
        
        # Stop early if we have enough frames
        if len(extracted_faces) >= seq_len:
            break

    cap.release()
    return extracted_faces

def predict_video(video_path, model_path):
    print(f"Loading model from {model_path}...")
    model = CNN_LSTM().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
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
    
    # Pad with zeros if we didn't find enough faces
    if len(processed_frames) < SEQ_LEN:
        pad = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        while len(processed_frames) < SEQ_LEN:
            processed_frames.append(pad)
            
    # Stack into sequence (1, T, C, H, W)
    sequence = torch.stack(processed_frames).unsqueeze(0).to(DEVICE)
    
    print("Running inference...")
    with torch.no_grad():
        logits = model(sequence)
        probability = torch.sigmoid(logits).item()
        
    print("\n" + "="*40)
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Fake Probability: {probability:.4f} ({probability*100:.2f}%)")
    
    if probability > 0.5:
        print("Prediction: FAKE 🛑")
    else:
        print("Prediction: REAL ✅")

    print("\n[WARNING] Model trained on only 35 videos/class and is heavily biased toward predicting FAKE. Treat this as a proof-of-concept!")
    print("="*40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a video with the CNN+LSTM Deepfake model.")
    parser.add_argument("video_path", help="Path to the video file to analyze")
    parser.add_argument("--model", default="best_cnn_lstm_model.pth", help="Path to the trained .pth model file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' not found.")
    elif not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found.")
    else:
        predict_video(args.video_path, args.model)
