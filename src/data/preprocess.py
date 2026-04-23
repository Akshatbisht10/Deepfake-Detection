import os
import cv2
import glob
from tqdm import tqdm
import math
import numpy as np

# Configuration
REAL_VIDEO_DIR = "DFD_original sequences"
FAKE_VIDEO_DIR = "DFD_manipulated_sequences"
OUTPUT_DIR = "dataset_faces"
IMG_SIZE = 299
FRAMES_PER_VIDEO = 20
MAX_VIDEOS = 50 # Initial limit

# Haar Cascade
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_faces(video_path, output_folder, label):
    try:
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_name = os.path.basename(video_path).split('.')[0]
        
        if frame_count > 0:
            intervals = max(1, frame_count // FRAMES_PER_VIDEO)
        else:
            intervals = 10
            
        count = 0
        frames_saved = 0
        
        while cap.isOpened() and frames_saved < FRAMES_PER_VIDEO:
            ret, frame = cap.read()
            if not ret:
                break
                
            if count % intervals == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
                
                for i, (x, y, w, h) in enumerate(faces):
                    # Add margin if possible
                    margin = int(w * 0.2)
                    x_m = max(0, x - margin)
                    y_m = max(0, y - margin)
                    w_m = min(frame.shape[1] - x_m, w + 2 * margin)
                    h_m = min(frame.shape[0] - y_m, h + 2 * margin)
                    
                    if w_m > 0 and h_m > 0:
                        face_img = frame[y_m:y_m+h_m, x_m:x_m+w_m]
                        try:
                            face_img = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))
                            # Save image
                            save_name = f"{video_name}_frame{count}_face{i}.jpg"
                            create_dir(output_folder)
                            cv2.imwrite(os.path.join(output_folder, save_name), face_img)
                            frames_saved += 1
                            if frames_saved >= FRAMES_PER_VIDEO:
                                break
                        except Exception:
                            pass
            count += 1
        cap.release()
    except Exception as e:
        print(f"Error processing {video_path}: {e}")

def main():
    create_dir(os.path.join(OUTPUT_DIR, "real"))
    create_dir(os.path.join(OUTPUT_DIR, "fake"))
    
    # Process Real Videos
    print("Processing Real Videos...")
    real_videos = [os.path.join(REAL_VIDEO_DIR, f) for f in os.listdir(REAL_VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]
    for video in tqdm(real_videos[:MAX_VIDEOS]):
        extract_faces(video, os.path.join(OUTPUT_DIR, "real"), "real")
        
    # Process Fake Videos
    print("Processing Fake Videos...")
    fake_videos = []
    for root, dirs, files in os.walk(FAKE_VIDEO_DIR):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov')):
                fake_videos.append(os.path.join(root, file))
    
    for video in tqdm(fake_videos[:MAX_VIDEOS]):
        extract_faces(video, os.path.join(OUTPUT_DIR, "fake"), "fake")

if __name__ == "__main__":
    main()
