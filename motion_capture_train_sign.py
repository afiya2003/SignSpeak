import cv2
import mediapipe as mp
import numpy as np
import os
import time
import winsound  # Windows beep

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ===================== CONFIG =====================
SEQ_LEN = 30              # frames per sample
FPS_DELAY = 0.03          # ~30 FPS
COUNTDOWN_SEC = 2         # countdown before recording starts
REST_SEC = 1.0            # rest time between samples
TARGET_SAMPLES = 30       # auto stop after this many new samples in AUTO mode (set None to never stop)

# Beep settings
BEEP_START_FREQ = 1200
BEEP_END_FREQ = 800
BEEP_MS = 120

# ===================== USER INPUT =====================
sign_name = input("Enter sign name (example: wakeup): ").strip().lower()

os.makedirs("signs_seq", exist_ok=True)
save_path = f"signs_seq/sign_{sign_name}.npy"

existing_data = None
if os.path.exists(save_path):
    print("⚠ This sign already exists.")
    print("1️⃣ Retrain from scratch")
    print("2️⃣ Add more samples to existing data")
    choice = input("Choose option (1 or 2): ").strip()

    if choice == "2":
        existing_data = np.load(save_path, allow_pickle=True)
        # Ensure shape is (N, SEQ_LEN, 126)
        if existing_data.ndim == 1 and len(existing_data) > 0 and hasattr(existing_data[0], "shape"):
            existing_data = np.stack(existing_data, axis=0)
        print(f"📊 Existing samples: {len(existing_data)}")
    elif choice == "1":
        print("🔁 Retraining from scratch...")
    else:
        print("❌ Invalid choice")
        raise SystemExit

def extract_features(result):
    features = []
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks[:2]:
            for lm in handLms.landmark:
                features.extend([lm.x, lm.y, lm.z])

    if len(features) == 0:
        features = [0.0] * 126
    elif len(features) == 63:
        features.extend([0.0] * 63)
    elif len(features) < 126:
        features.extend([0.0] * (126 - len(features)))

    return features[:126]

def beep_start():
    try:
        winsound.Beep(BEEP_START_FREQ, BEEP_MS)
    except Exception:
        pass

def beep_end():
    try:
        winsound.Beep(BEEP_END_FREQ, BEEP_MS)
    except Exception:
        pass

def draw_status(frame, text, y=40):
    cv2.putText(frame, text, (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

# ===================== TRAINING =====================
data_seq = []

with mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
) as hands:

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not opened")
        raise SystemExit

    print("🎥 Camera started")
    print("Controls:")
    print("  S = record ONE sample (manual)")
    print("  A = toggle AUTO mode (hands-free)")
    print("  Q = quit")

    recording = False
    auto_mode = False
    seq_buffer = []

    phase = "IDLE"         # IDLE / COUNTDOWN / RECORD / REST
    phase_end_time = 0.0
    countdown_end_time = 0.0
    rest_end_time = 0.0
    current_count_num = COUNTDOWN_SEC

    auto_started_samples = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Draw landmarks
        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        key = cv2.waitKey(1) & 0xFF
        now = time.time()

        # ---------- Key controls ----------
        if key == ord('q'):
            break

        if key == ord('a'):
            auto_mode = not auto_mode
            if auto_mode:
                print("✅ AUTO mode ON")
                phase = "COUNTDOWN"
                countdown_end_time = now + COUNTDOWN_SEC
                current_count_num = COUNTDOWN_SEC
            else:
                print("🛑 AUTO mode OFF")
                phase = "IDLE"
                seq_buffer = []

        if key == ord('s') and not recording:
            # Manual single sample recording
            phase = "COUNTDOWN"
            countdown_end_time = now + COUNTDOWN_SEC
            current_count_num = COUNTDOWN_SEC
            auto_mode = False  # manual overrides auto

        # ---------- Phase machine ----------
        if phase == "IDLE":
            draw_status(frame, f"IDLE | saved: {len(data_seq)} | press A for AUTO, S for manual")

            # If auto_mode is ON, it should not be idle
            if auto_mode:
                phase = "COUNTDOWN"
                countdown_end_time = now + COUNTDOWN_SEC
                current_count_num = COUNTDOWN_SEC

        elif phase == "COUNTDOWN":
            # countdown display
            remaining = int(max(0, countdown_end_time - now))
            # show 2..1..0 nicely
            draw_status(frame, f"GET READY... {remaining+1}", y=40)
            draw_status(frame, f"Next sample in {remaining+1}s", y=80)

            if now >= countdown_end_time:
                phase = "RECORD"
                recording = True
                seq_buffer = []
                beep_start()
                print("⏺ Recording started... perform the sign!")

        elif phase == "RECORD":
            feats = extract_features(result)
            seq_buffer.append(feats)

            draw_status(frame, f"RECORDING {len(seq_buffer)}/{SEQ_LEN}", y=40)

            if len(seq_buffer) >= SEQ_LEN:
                data_seq.append(np.array(seq_buffer, dtype=np.float32))
                beep_end()
                print(f"✅ Sample saved: {len(data_seq)}")

                recording = False
                seq_buffer = []

                # Move to rest
                phase = "REST"
                rest_end_time = now + REST_SEC

                # Auto-stop logic
                if auto_mode:
                    auto_started_samples += 1
                    if TARGET_SAMPLES is not None and auto_started_samples >= TARGET_SAMPLES:
                        print(f"🏁 Reached target of {TARGET_SAMPLES} samples. AUTO mode OFF.")
                        auto_mode = False
                        phase = "IDLE"

        elif phase == "REST":
            remaining_rest = max(0.0, rest_end_time - now)
            draw_status(frame, f"REST... {remaining_rest:.1f}s | saved: {len(data_seq)}", y=40)

            if now >= rest_end_time:
                if auto_mode:
                    phase = "COUNTDOWN"
                    countdown_end_time = now + COUNTDOWN_SEC
                else:
                    phase = "IDLE"

        cv2.imshow("Motion Training (Hands-Free)", frame)

        # Keep consistent timing (best-effort)
        time.sleep(FPS_DELAY)

    cap.release()
    cv2.destroyAllWindows()

# ===================== SAVE DATA =====================
if len(data_seq) == 0:
    print("❌ No samples collected")
    raise SystemExit

new_data = np.array(data_seq, dtype=np.float32)  # (samples, SEQ_LEN, 126)

if existing_data is not None:
    final_data = np.concatenate([existing_data, new_data], axis=0)
    print(f"📈 Total samples after merge: {len(final_data)}")
else:
    final_data = new_data
    print("🆕 Training from scratch")

np.save(save_path, final_data)
print("🎉 Training completed successfully!")
print(f"📁 Saved at: {save_path}")
print(f"📐 Saved shape: {final_data.shape}  (samples, SEQ_LEN, 126)")