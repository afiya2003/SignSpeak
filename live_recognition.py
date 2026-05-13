import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import threading
import time
import pyttsx3
import os
import ctypes
import re
from collections import deque, Counter

# =============================
# STABILITY / SMOOTHING SETTINGS
# =============================
SEQ_LEN = 30
FEATS = 126

PRED_HISTORY = 9
MIN_AGREE = 6
LOCK_SECONDS = 0.6
CONF_THRESH = 0.75          
PRED_EVERY_N_FRAMES = 2

# speak settings
SPEAK_COOLDOWN = 0.6

# =============================
# ADDED: SENTENCE BUILD SETTINGS
# =============================
TOKEN_COOLDOWN = 1.2        # min seconds before accepting same token again
PAUSE_TO_FINALIZE = 2.0     # if no new token for this time => finalize sentence and speak
MAX_TOKENS = 12             # max tokens kept in sentence buffer
IGNORE_TOKENS = {"WAITING...", "NO HAND", "COLLECTING...", "UNSURE"}

# Optional: if you have a "DONE" sign in labels, it can finalize immediately:
FINALIZE_TOKENS = {"DONE", "STOP", "FINISH"}  # include only if these exist in your dataset

# =============================
# 1. System & Camera Management
# =============================
def force_close_camera_apps():
    try:
        os.system("taskkill /IM WindowsCamera.exe /F >nul 2>&1")
    except Exception:
        pass

def show_camera_error_popup():
    ctypes.windll.user32.MessageBoxW(
        0,
        "The Camera is currently in use by another application (Zoom, Teams, etc.).\n\nPlease close other apps and restart this program.",
        "Camera Error - Access Denied",
        0x10 | 0x0
    )

# =============================
# 2. Audio Engine Setup
# =============================
print("🔊 Initializing Audio Engine...")
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

def speak_worker(text: str):
    try:
        engine.say(text)
        engine.runAndWait()
    except RuntimeError:
        pass

def speak_interrupt(text: str):
    try:
        if getattr(engine, "_inLoop", False):
            engine.stop()
    except Exception:
        pass
    t = threading.Thread(target=speak_worker, args=(text,), daemon=True)
    t.start()

# =============================
# 3. Startup
# =============================
print("⚙️ System Check...")
force_close_camera_apps()
time.sleep(1)
speak_interrupt("Motion sign recognition started")

# =============================
# 4. Model & MediaPipe Setup
# =============================
MODEL_PATH = "model_seq/motion_model_final.h5"
LABEL_PATH = "model_seq/labels.npy"

print("⏳ Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    labels = np.load(LABEL_PATH, allow_pickle=True)
    print(f"✅ Model loaded! Classes: {labels}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise SystemExit

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

frame_buffer = deque(maxlen=SEQ_LEN)
pred_history = deque(maxlen=PRED_HISTORY)

stable_label = "Waiting..."
stable_conf = 0.0
stable_until = 0.0

last_spoken_text = ""
last_spoken_time = 0.0

# =============================
# ADDED: SENTENCE BUFFER STATE
# =============================
token_buffer = deque(maxlen=MAX_TOKENS)
last_token_time = 0.0
last_token = ""
last_sentence_spoken = ""
last_sentence_time = 0.0

# =============================
# ADDED: NLP / RULE LAYER
# =============================
QUESTION_WORDS = {"WHAT", "WHERE", "WHEN", "WHY", "HOW", "WHO"}

# Set-based expansions (fast and reliable)
EXPANSION_RULES = [
    ({"WHAT", "NAME"}, "What is your name?"),
    ({"YOUR", "NAME"}, "What is your name?"),
    ({"WHO", "YOU"}, "Who are you?"),
    ({"WHERE", "TOILET"}, "Where is the toilet?"),
    ({"I", "HUNGRY"}, "I am hungry."),
    ({"I", "THIRSTY"}, "I am thirsty."),
    ({"I", "SICK"}, "I feel sick."),
    ({"NEED", "HELP"}, "I need help."),
    ({"CALL", "HELP"}, "Please call for help."),
]

PATTERN_RULES = [
    (re.compile(r"\bWHAT\b.*\bNAME\b", re.I), "What is your name?"),
    (re.compile(r"\bYOUR\b.*\bNAME\b", re.I), "What is your name?"),
    (re.compile(r"\bMY\b.*\bNAME\b.*\b([A-Z]+)\b", re.I), None),  # handled in slot fill
]

def normalize_tokens(tokens):
    out = []
    for t in tokens:
        t = t.strip().upper()
        # map variants here if you want
        if t in {"THANKYOU", "THANKS"}:
            t = "THANK YOU"
        if t in {"WHATS", "WHAT'S"}:
            t = "WHAT"
        out.extend(t.split())
    return out

def apply_expansion_rules(tokens):
    tokset = set(tokens)
    for needed, out in EXPANSION_RULES:
        if needed.issubset(tokset):
            return out
    return None

def slot_fill_my_name(tokens):
    # If tokens look like: MY NAME ASHRAF  -> My name is Ashraf.
    tokens_u = [t.upper() for t in tokens]
    if len(tokens_u) >= 3 and tokens_u[0] in {"MY", "ME", "I"} and tokens_u[1] == "NAME":
        name = tokens_u[2]
        # Basic cleanup: if name is a known sign token, keep it
        return f"My name is {name.title()}."
    return None

def basic_grammar(tokens):
    if not tokens:
        return ""

    is_question = any(t in QUESTION_WORDS for t in tokens)

    sent = " ".join(t.lower() for t in tokens)
    sent = sent[:1].upper() + sent[1:] if sent else sent

    if is_question and not sent.endswith("?"):
        sent += "?"
    elif not is_question and not sent.endswith("."):
        sent += "."

    sent = sent.replace(" i ", " I ").replace("i ", "I ")
    sent = sent.replace(" your ", " your ").replace(" you ", " you ")
    return sent

def tokens_to_sentence(tokens):
    tokens = normalize_tokens(tokens)
    if not tokens:
        return ""

    # Slot filling first
    slot = slot_fill_my_name(tokens)
    if slot:
        return slot

    # Rule expansions
    out = apply_expansion_rules(tokens)
    if out:
        return out

    # Pattern rules
    joined = " ".join(tokens)
    for pat, out2 in PATTERN_RULES:
        if pat.search(joined) and out2:
            return out2

    # Fallback
    return basic_grammar(tokens)

def add_token_if_valid(token: str):
    """
    Add stable token to sentence buffer with cooldown filtering.
    """
    global last_token, last_token_time

    tok = token.strip().upper()
    if not tok or tok in IGNORE_TOKENS:
        return

    now = time.time()

    # finalize token triggers
    if tok in FINALIZE_TOKENS:
        token_buffer.clear()
        return

    # prevent rapid repeats
    if tok == last_token and (now - last_token_time) < TOKEN_COOLDOWN:
        return

    token_buffer.append(tok)
    last_token = tok
    last_token_time = now

# =============================
# Feature extraction
# =============================
def extract_frame_features(result) -> np.ndarray:
    features = []

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks[:2]:
            for lm in hand.landmark:
                features.extend([lm.x, lm.y, lm.z])

    if len(features) == 0:
        features = [0.0] * 126
    elif len(features) == 63:
        features.extend([0.0] * 63)
    elif len(features) < 126:
        features.extend([0.0] * (126 - len(features)))

    features = np.array(features[:126], dtype=np.float32)

    reshaped = features.reshape(-1, 3)
    base_x, base_y, base_z = reshaped[0]
    reshaped[:, 0] -= base_x
    reshaped[:, 1] -= base_y
    reshaped[:, 2] -= base_z

    return reshaped.flatten()

def update_stable_label(class_id: int, conf: float):
    global stable_label, stable_conf, stable_until
    now = time.time()

    if conf >= CONF_THRESH:
        pred_history.append(class_id)

    if now < stable_until and stable_label not in ["Waiting...", "No hand"]:
        return stable_label, stable_conf, "LOCKED"

    if len(pred_history) < MIN_AGREE:
        return "Collecting...", 0.0, "COLLECTING"

    counts = Counter(pred_history)
    best_id, best_count = counts.most_common(1)[0]

    if best_count >= MIN_AGREE:
        stable_label = str(labels[best_id])
        stable_conf = float(conf)
        stable_until = now + LOCK_SECONDS
        return stable_label, stable_conf, "STABLE"

    return "Unsure", 0.0, "UNSURE"

# =============================
# 5. Camera Initialization
# =============================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera is busy.")
    show_camera_error_popup()
    raise SystemExit

print("📷 Camera Acquired. Starting Loop...")

# =============================
# 6. Main Loop
# =============================
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Frame drop.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    frame_count += 1
    prediction_text = stable_label

    # -----------------------
    # Hand detected
    # -----------------------
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        feat126 = extract_frame_features(result)
        frame_buffer.append(feat126)

        if len(frame_buffer) == SEQ_LEN and (frame_count % PRED_EVERY_N_FRAMES == 0):
            seq = np.array(frame_buffer, dtype=np.float32).reshape(1, SEQ_LEN, FEATS)
            pred = model.predict(seq, verbose=0)

            class_id = int(np.argmax(pred))
            conf = float(np.max(pred))

            label, label_conf, state = update_stable_label(class_id, conf)

            if label in ["Collecting...", "Unsure"]:
                prediction_text = label
            else:
                prediction_text = f"{label} ({label_conf:.2f})"

            # =============================
            # ADDED: Token -> Sentence flow
            # =============================
            if label not in ["Collecting...", "Unsure", "Waiting...", "No hand"]:
                add_token_if_valid(label)

    # -----------------------
    # No hand detected
    # -----------------------
    else:
        frame_buffer.clear()
        pred_history.clear()
        stable_label = "No hand"
        stable_conf = 0.0
        stable_until = 0.0
        prediction_text = "No hand"

    # =============================
    # ADDED: Build sentence from tokens
    # =============================
    tokens_list = list(token_buffer)
    sentence_text = tokens_to_sentence(tokens_list)

    # Finalize and speak sentence when user pauses
    now = time.time()
    if tokens_list and (now - last_token_time) > PAUSE_TO_FINALIZE:
        if sentence_text and sentence_text != last_sentence_spoken and (now - last_sentence_time) > 0.8:
            speak_interrupt(sentence_text)
            last_sentence_spoken = sentence_text
            last_sentence_time = now
        token_buffer.clear()

    # =============================
    # UI bottom bar
    # =============================
    h, w, _ = frame.shape
    cv2.rectangle(frame, (0, h - 90), (w, h), (0, 0, 0), -1)

    # line 1: stable word
    cv2.putText(frame, f"Word: {prediction_text}", (20, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # line 2: sentence preview
    preview = sentence_text if sentence_text else "Sentence: ..."
    cv2.putText(frame, f"{preview}", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("ISL Motion Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # Optional: press 'c' to clear sentence buffer manually
    if key == ord('c'):
        token_buffer.clear()

cap.release()
cv2.destroyAllWindows()
print("❌ Program closed.")