import os
import cv2
import shutil

# Configuration
REAL_VIDEO_DIR = "DFD_original sequences"
# Haar Cascade path
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
OUTPUT_DIR = "debug_faces_cv2"

def extract_faces_test_cv2(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    cap = cv2.VideoCapture(video_path)
    
    frames_extracted = 0
    while cap.isOpened() and frames_extracted < 3:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)
        
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Save crop
            face_img = frame[y:y+h, x:x+w]
            if face_img.size > 0:
                cv2.imwrite(os.path.join(output_folder, f"face_{frames_extracted}_{i}.jpg"), face_img)
        
        # Save full frame
        cv2.imwrite(os.path.join(output_folder, f"frame_{frames_extracted}.jpg"), frame)
        frames_extracted += 1
        
    cap.release()
    print(f"Processed {video_path}, saved to {output_folder}")

def main():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)
    
    # Test 1 Real Video
    real_videos = [f for f in os.listdir(REAL_VIDEO_DIR) if f.endswith('.mp4')]
    if real_videos:
        print(f"Testing Real Video: {real_videos[0]}")
        extract_faces_test_cv2(os.path.join(REAL_VIDEO_DIR, real_videos[0]), os.path.join(OUTPUT_DIR, "real_test"))

if __name__ == "__main__":
    main()
