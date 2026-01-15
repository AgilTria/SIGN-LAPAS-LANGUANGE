import cv2
import numpy as np
import mediapipe as mp
import time
from collections import deque

from src.similarity import load_templates, predict
from src.translate import translate
from src.tts import speak_id_jawa

# =========================
# KONFIGURASI SISTEM
# =========================
POSE_BUFFER_SIZE = 7
SCORE_THRESHOLD = 0.90
MARGIN_THRESHOLD = 0.05
COOLDOWN = 1.5

# =========================
# UKURAN LAYOUT
# =========================
VIDEO_W, VIDEO_H = 640, 480
PANEL_W = 420
CANVAS_W = VIDEO_W + PANEL_W
CANVAS_H = VIDEO_H

# =========================
# INIT
# =========================
templates = load_templates()
pose_buffer = deque(maxlen=POSE_BUFFER_SIZE)

last_spoken_time = 0
has_spoken = False

last_word = ""
last_id = ""
last_jv = ""
last_score = 0.0
status_text = "DETECTING"

mp_hands = mp.solutions.hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_H)

# =========================
# UI HELPER
# =========================
def draw_confidence_bar(panel, score):
    x, y = 30, 300
    w, h = 320, 25

    filled = int(w * score)
    color = (0, 200, 0) if score >= SCORE_THRESHOLD else (0, 200, 200)

    cv2.rectangle(panel, (x, y), (x + w, y + h), (255, 255, 255), 2)
    cv2.rectangle(panel, (x, y), (x + filled, y + h), color, -1)

    cv2.putText(panel, f"{score:.2f}", (x + w - 60, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# =========================
# MAIN LOOP
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    panel = np.zeros((CANVAS_H, PANEL_W, 3), dtype=np.uint8)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mp_hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        landmarks = [[lm.x, lm.y, lm.z] for lm in hand.landmark]

        wrist = landmarks[0]
        landmarks = [[x-wrist[0], y-wrist[1], z-wrist[2]] for x,y,z in landmarks]

        pose_buffer.append(np.array(landmarks))

        if len(pose_buffer) == POSE_BUFFER_SIZE:
            pose = np.mean(pose_buffer, axis=0)
            word, score, margin = predict(pose, templates)

            last_word = word
            last_score = score

            if score < 0.85:
                status_text = "DETECTING"
            elif score < SCORE_THRESHOLD:
                status_text = "HOLD POSE"
            else:
                status_text = "STABLE"

            now = time.time()
            if (
                score >= SCORE_THRESHOLD and
                not has_spoken and
                now - last_spoken_time > COOLDOWN
            ):

                res_id = translate(word, "ngoko")
                res_jv = translate(word, "krama")

                if res_id and res_jv:
                    last_id = res_id["indonesia"]
                    last_jv = res_jv["jawa"]

                    speak_id_jawa(last_id, last_jv)
                    has_spoken = True
                    last_spoken_time = now
                    status_text = "SPEAKING"
    else:
        pose_buffer.clear()
        has_spoken = False
        status_text = "DETECTING"

    # =========================
    # PANEL UI (KANAN)
    # =========================
    cv2.putText(panel, "SIGN LANGUAGE", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(panel, "TRANSLATOR", (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(panel, f"Gesture:", (30, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    cv2.putText(panel, last_word, (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(panel, "Bahasa Indonesia:", (30, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(panel, last_id, (30, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.putText(panel, "Bahasa Jawa:", (30, 270),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.putText(panel, last_jv, (30, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    draw_confidence_bar(panel, last_score)

    cv2.putText(panel, f"Status: {status_text}", (30, 360),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # =========================
    # GABUNGKAN KE CANVAS
    # =========================
    canvas[:, :VIDEO_W] = frame
    canvas[:, VIDEO_W:] = panel

    cv2.imshow("Sign Language Translator", canvas)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
