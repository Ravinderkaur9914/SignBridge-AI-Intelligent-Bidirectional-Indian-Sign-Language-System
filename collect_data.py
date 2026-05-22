"""
collect_data.py
--------------
Step 1: Data Collection
Captures hand landmarks via webcam using MediaPipe and saves them
as numerical feature vectors (42 values: x, y for 21 hand landmarks).

NOW WITH: Reference image shown on screen for each gesture!

Usage:
    python collect_data.py
"""

import cv2
import mediapipe as mp
import csv
import os
import time

# ── MediaPipe setup ────────────────────────────────────────────────
mp_hands    = mp.solutions.hands
mp_draw     = mp.solutions.drawing_utils
mp_style    = mp.solutions.drawing_styles

# ── Settings ───────────────────────────────────────────────────────
SAMPLES_PER_GESTURE = 300
DATASET_DIR         = "dataset"
LANDMARKS_CSV       = os.path.join(DATASET_DIR, "landmarks.csv")   # ← single combined CSV
SIGNS_ALPHA_DIR     = "signs/alphabets"   # your alphabet images folder
SIGNS_WORDS_DIR     = "signs/words"       # your words images folder

# ── All gestures (words + alphabets) ──────────────────────────────
# Words first, then alphabets A-Z
GESTURES = [
    # Common words
    "Hello", "Yes", "No", "Good", "Bad",
    "Please", "Sorry", "Thank You", "Help", "Stop",
    "I", "You", "We", "What", "Where",
    # Alphabets
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
]

# ── Helper: find reference image for a gesture ─────────────────────
def find_reference_image(gesture_name):
    """
    Looks for a reference image in signs/alphabets or signs/words.
    Tries common extensions: jpg, jpeg, png, bmp
    """
    extensions = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"]
    
    # Check both folders
    search_dirs = [SIGNS_ALPHA_DIR, SIGNS_WORDS_DIR]
    
    for folder in search_dirs:
        for ext in extensions:
            path = os.path.join(folder, gesture_name + ext)
            if os.path.exists(path):
                return path
            # Also try lowercase
            path = os.path.join(folder, gesture_name.lower() + ext)
            if os.path.exists(path):
                return path
    return None  # No image found


# ── Helper: overlay reference image on frame ───────────────────────
def overlay_reference_image(frame, ref_img_path, position="top-right"):
    """
    Loads reference image and places it as a small box on the camera frame.
    Shows a placeholder box if no image found.
    """
    h, w = frame.shape[:2]
    box_size = 180  # size of the reference image box

    # Position: top-right corner
    x_start = w - box_size - 10
    y_start = 10

    if ref_img_path and os.path.exists(ref_img_path):
        ref_img = cv2.imread(ref_img_path)
        if ref_img is not None:
            ref_img = cv2.resize(ref_img, (box_size, box_size))
            # Draw white border around reference image
            cv2.rectangle(frame,
                          (x_start - 3, y_start - 3),
                          (x_start + box_size + 3, y_start + box_size + 3),
                          (255, 255, 255), 3)
            frame[y_start:y_start+box_size, x_start:x_start+box_size] = ref_img
            # Label above image
            cv2.putText(frame, "Reference", (x_start, y_start - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            return frame

    # ── No image found: draw placeholder box ──────────────────────
    cv2.rectangle(frame,
                  (x_start, y_start),
                  (x_start + box_size, y_start + box_size),
                  (80, 80, 80), -1)
    cv2.rectangle(frame,
                  (x_start, y_start),
                  (x_start + box_size, y_start + box_size),
                  (200, 200, 200), 2)
    cv2.putText(frame, "No Image", (x_start + 35, y_start + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    cv2.putText(frame, "Found", (x_start + 50, y_start + 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
    return frame


# ── Helper: draw HUD (progress bar, labels) on frame ──────────────
def draw_hud(frame, gesture_name, count, total, collecting, hand_detected):
    h, w = frame.shape[:2]

    # ── Top bar: gesture name ──────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
    cv2.putText(frame, f"Gesture: {gesture_name}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # ── Hand detected status ───────────────────────────────────────
    if hand_detected:
        status_color = (0, 220, 0)      # Green
        status_text  = "Hand Detected"
    else:
        status_color = (0, 0, 220)      # Red
        status_text  = "No Hand - Show Your Hand!"

    cv2.rectangle(frame, (0, 55), (w, 90), (20, 20, 20), -1)
    cv2.putText(frame, status_text, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, status_color, 2)

    # ── Progress bar ───────────────────────────────────────────────
    bar_x, bar_y, bar_w, bar_h = 10, h - 60, w - 20, 28
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (60, 60, 60), -1)
    filled = int(bar_w * (count / total))
    bar_color = (0, 200, 0) if collecting else (100, 100, 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h),
                  bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                  (200, 200, 200), 2)
    cv2.putText(frame, f"{count}/{total}", (bar_x + bar_w//2 - 30, bar_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # ── Bottom instructions ────────────────────────────────────────
    if not collecting:
        cv2.putText(frame, "Press S to START  |  Press Q to SKIP",
                    (10, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
    else:
        cv2.putText(frame, "Recording... Hold your hand STILL",
                    (10, h - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


# ── Main collection loop ───────────────────────────────────────────
def collect_data():
    os.makedirs(DATASET_DIR, exist_ok=True)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        return

    print("\n" + "="*55)
    print("  SIGN LANGUAGE DATA COLLECTION")
    print("="*55)
    print(f"  Total gestures : {len(GESTURES)}")
    print(f"  Samples each   : {SAMPLES_PER_GESTURE}")
    print(f"  Estimated time : {len(GESTURES) * 2} – {len(GESTURES) * 3} minutes")
    print("="*55)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        for gesture_idx, gesture in enumerate(GESTURES):
            gesture_dir = os.path.join(DATASET_DIR, gesture)
            os.makedirs(gesture_dir, exist_ok=True)

            # ── Check how many samples already exist ───────────────
            existing = len([f for f in os.listdir(gesture_dir)
                            if f.endswith('.jpg')])
            if existing >= SAMPLES_PER_GESTURE:
                print(f"\n[{gesture_idx+1}/{len(GESTURES)}] '{gesture}' "
                      f"already has {existing} samples — SKIPPING")
                continue

            # ── Find reference image ───────────────────────────────
            ref_img_path = find_reference_image(gesture)
            if ref_img_path:
                print(f"\n[{gesture_idx+1}/{len(GESTURES)}] '{gesture}' "
                      f"→ Reference image found: {ref_img_path}")
            else:
                print(f"\n[{gesture_idx+1}/{len(GESTURES)}] '{gesture}' "
                      f"→ No reference image found (will show placeholder)")

            print(f"  Existing samples : {existing}/{SAMPLES_PER_GESTURE}")
            print("  Look at the camera window!")
            print("  Press S to START  |  Press Q to SKIP")

            count      = existing
            collecting = False

            # ── Ensure landmarks.csv exists with header ────────────
            if not os.path.exists(LANDMARKS_CSV):
                with open(LANDMARKS_CSV, "w", newline="") as hf:
                    writer_h = csv.writer(hf)
                    header = ["label"] + [f"{ax}{i}"
                              for i in range(21) for ax in ("x", "y")]
                    writer_h.writerow(header)
                    print(f"  Created {LANDMARKS_CSV} with header")

            # ── Wait screen (show reference before starting) ───────
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                # Show reference image
                frame = overlay_reference_image(frame, ref_img_path)

                # Big instruction text
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (0, 0), (w, 50), (30, 30, 30), -1)
                cv2.putText(frame,
                            f"[{gesture_idx+1}/{len(GESTURES)}] Next: {gesture}",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 255), 2)
                cv2.putText(frame,
                            "Make this hand shape  →  then press S",
                            (10, h//2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                            (255, 255, 0), 2)
                cv2.putText(frame, "Press S = Start  |  Q = Skip",
                            (10, h//2 + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (200, 200, 200), 1)

                cv2.imshow("Sign Language Data Collection", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('s') or key == ord('S'):
                    collecting = True
                    break
                elif key == ord('q') or key == ord('Q'):
                    print(f"  Skipped: {gesture}")
                    collecting = False
                    break

            if not collecting:
                continue

            # ── Recording loop ─────────────────────────────────────
            with open(LANDMARKS_CSV, "a", newline="") as f:
                writer = csv.writer(f)

                while count < SAMPLES_PER_GESTURE:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)

                    rgb         = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result      = hands.process(rgb)
                    hand_detected = False

                    if result.multi_hand_landmarks:
                        hand_detected = True
                        for hand_lm in result.multi_hand_landmarks:
                            # Draw landmarks on frame
                            mp_draw.draw_landmarks(
                                frame, hand_lm,
                                mp_hands.HAND_CONNECTIONS,
                                mp_style.get_default_hand_landmarks_style(),
                                mp_style.get_default_hand_connections_style()
                            )

                            # Extract 42 values (x, y per landmark)
                            row = []
                            for lm in hand_lm.landmark:
                                row.extend([lm.x, lm.y])

                            writer.writerow([gesture] + row)

                            # Also save image
                            img_path = os.path.join(gesture_dir,
                                                     f"{count}.jpg")
                            cv2.imwrite(img_path, frame)
                            count += 1

                    # Draw HUD and reference image
                    frame = overlay_reference_image(frame, ref_img_path)
                    frame = draw_hud(frame, gesture, count,
                                     SAMPLES_PER_GESTURE, True, hand_detected)

                    cv2.imshow("Sign Language Data Collection", frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("  Stopped early by user.")
                        break

            print(f"  ✅ Done! {count} samples saved for '{gesture}'")

        print("\n" + "="*55)
        print("  ALL GESTURES COMPLETE!")
        print(f"  Data saved in: {os.path.abspath(DATASET_DIR)}")
        print("  Next step: python train_model.py")
        print("="*55 + "\n")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    collect_data()