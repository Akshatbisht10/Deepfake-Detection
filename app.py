import gradio as gr
import os
import torch
import cv2
from torchvision import transforms
from PIL import Image

from src.training.train_cnn_lstm import SEQ_LEN, IMG_SIZE
from src.models.st_vit_model import ST_ViT

# Configuration Let's use CPU for inference as it's typically safer for web apps 
# unless specifically running on a dedicated GPU server.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_st_vit_model.pth"

# Load Haar Cascade face detector
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Global model variable to hold the loaded model
model = None

def load_model():
    global model
    if model is None:
        print(f"Loading ST-ViT model from {MODEL_PATH} onto {DEVICE}...")
        model = ST_ViT(
            seq_len=SEQ_LEN,
            lstm_hidden=256,
            lstm_layers=2,
            vit_dim=512,
            vit_depth=4,
            vit_heads=8,
            freeze_cnn=False
        ).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
    return model

def extract_faces_from_video(video_path, seq_len=SEQ_LEN):
    """Extracts exactly `seq_len` evenly spaced face crops from the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []
        
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        return []

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

def predict(video_file):
    if video_file is None:
        return "Please upload a video file."
        
    loaded_model = load_model()
    video_path = video_file
    
    faces = extract_faces_from_video(video_path)
    
    if len(faces) == 0:
        return "Error: Could not detect any clear human faces in the video."
        
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
    
    # Run inference
    with torch.no_grad():
        logits = loaded_model(sequence)
        probability = torch.sigmoid(logits).item()
        
    # Revert to standard threshold
    is_fake = probability > 0.5
    
    # Formatting the output
    prediction_text = "🚨 FAKE 🚨" if is_fake else "✅ REAL ✅"
    confidence = probability if is_fake else (1.0 - probability)
    
    details_text = f"**Prediction:** {prediction_text}\n\n**Confidence:** {confidence*100:.1f}%\n*(Fake Probability Score: {probability:.4f})*\n\n---\n*Powered by Spatiotemporal Vision Transformer (ST-ViT) V2*"
    
    return details_text

# Define Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# 🕵️ Deepfake Detection Pipeline")
    gr.Markdown("Upload a video of a person talking, and the highly-accurate **Spatiotemporal Vision Transformer (ST-ViT)** model will analyze sequential frames to determine if it is a deepfake or an authentic video.")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="Upload Video")
            submit_btn = gr.Button("Analyze Video", variant="primary")
            
        with gr.Column():
            output_markdown = gr.Markdown("Waiting for video upload...")
            
    submit_btn.click(fn=predict, inputs=video_input, outputs=output_markdown)

if __name__ == "__main__":
    # Launch on local host
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, theme=gr.themes.Soft())
