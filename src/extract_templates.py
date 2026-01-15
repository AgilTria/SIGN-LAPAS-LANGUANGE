import cv2
import numpy as np
import mediapipe as mp
import os

VIDEO_DIR = "data/videos"
OUT_DIR = "data/templates"

os.makedirs(OUT_DIR, exist_ok=True)

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

def extract_pose(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            landmarks = [[lm.x, lm.y, lm.z] for lm in hand.landmark]

            # NORMALISASI KE WRIST
            wrist = landmarks[0]
            landmarks = [[x-wrist[0], y-wrist[1], z-wrist[2]] for x,y,z in landmarks]

            frames.append(landmarks)

    cap.release()

    if len(frames) == 0:
        return None

    # AMBIL FRAME TENGAH (POSE INTI)
    return np.array(frames[len(frames)//2])

for file in os.listdir(VIDEO_DIR):
    if file.endswith(".mp4"):
        word = os.path.splitext(file)[0]
        pose = extract_pose(os.path.join(VIDEO_DIR, file))

        if pose is not None:
            np.save(os.path.join(OUT_DIR, f"{word}.npy"), pose)
            print(f"✅ Template dibuat: {word}")
        else:
            print(f"❌ Gagal: {word}")
