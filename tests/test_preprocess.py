import os
import cv2
import mediapipe as mp
import shutil

# Configuration
REAL_VIDEO_DIR = "DFD_original sequences"
FAKE_VIDEO_DIR = "DFD_manipulated_sequences"
OUTPUT_DIR = "debug_faces"

def extract_faces_test(video_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    cap = cv2.VideoCapture(video_path)
    mp_face_detection = mp.solutions.face_detection
    
    frames_extracted = 0
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened() and frames_extracted < 3: # Just extract 3 frames
            ret, frame = cap.read()
            if not ret:
                break
                
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)
            
            if results.detections:
                for i, detection in enumerate(results.detections):
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x = int(bboxC.xmin * iw)
                    y = int(bboxC.ymin * ih)
                    w = int(bboxC.width * iw)
                    h = int(bboxC.height * ih)
                    
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Save crop
                    face_img = frame[max(0,y):y+h, max(0,x):x+w]
                    if face_img.size > 0:
                        cv2.imwrite(os.path.join(output_folder, f"face_{frames_extracted}_{i}.jpg"), face_img)
            
            # Save full frame with bbox
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
        extract_faces_test(os.path.join(REAL_VIDEO_DIR, real_videos[0]), os.path.join(OUTPUT_DIR, "real_test"))
        
    # Test 1 Fake Video
    fake_videos = [f for f in os.listdir(FAKE_VIDEO_DIR) if f.endswith('.mp4')]
    if fake_videos:
        print(f"Testing Fake Video: {fake_videos[0]}")
        extract_faces_test(os.path.join(FAKE_VIDEO_DIR, fake_videos[0]), os.path.join(OUTPUT_DIR, "fake_test"))

if __name__ == "__main__":
    main()
