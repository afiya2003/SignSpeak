import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Configuration ---
DATA_DIR = "signs_seq"   # <-- motion sequences folder
MODEL_DIR = "model_seq"
os.makedirs(MODEL_DIR, exist_ok=True)

SEQ_LEN = 30   # must match what you recorded
FEATS = 126

def normalize_frame(frame_126):
    """Wrist-relative normalization per frame (126 -> 126)."""
    frame = np.array(frame_126, dtype=np.float32)

    if frame.shape[0] != FEATS:
        return None

    pts = frame.reshape(-1, 3)     # (42, 3)
    base = pts[0].copy()           # wrist (x,y,z)
    pts -= base                    # relative
    return pts.flatten()

print("📂 Loading sequence data...")

X, y = [], []

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Directory '{DATA_DIR}' not found.")

for file in os.listdir(DATA_DIR):
    if file.endswith(".npy"):
        label = file.replace("sign_", "").replace(".npy", "")
        file_path = os.path.join(DATA_DIR, file)

        try:
            data = np.load(file_path, allow_pickle=True)  # (samples, SEQ_LEN, 126)

            if data.ndim != 3:
                print(f"Skipping {file}: expected 3D (samples, SEQ_LEN, 126), got {data.shape}")
                continue

            for seq in data:
                if seq.shape[0] != SEQ_LEN:
                    # if some file has different seq length, skip or pad/truncate
                    continue

                seq_norm = []
                ok = True
                for frame in seq:
                    nf = normalize_frame(frame)
                    if nf is None:
                        ok = False
                        break
                    seq_norm.append(nf)

                if ok:
                    X.append(np.stack(seq_norm, axis=0))   # (SEQ_LEN, 126)
                    y.append(label)

        except Exception as e:
            print(f"Skipping {file}: {e}")

X = np.array(X, dtype=np.float32)   # (N, SEQ_LEN, 126)
print(f"✅ Loaded {len(X)} sequences. Input shape: {X.shape}")

# --- Label Encoding ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = tf.keras.utils.to_categorical(y_encoded)
np.save(os.path.join(MODEL_DIR, "labels.npy"), le.classes_)

# --- Stratified Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, stratify=y_encoded, random_state=42
)

# --- LSTM Model ---
model = Sequential([
    tf.keras.layers.Input(shape=(SEQ_LEN, FEATS)),

    LSTM(128, return_sequences=True),
    Dropout(0.3),

    LSTM(64),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    Dense(len(le.classes_), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --- Callbacks ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_model.h5"),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1)
]

print("🚀 Training motion model (LSTM)...")
history = model.fit(
    X_train, y_train,
    epochs=150,
    batch_size=16,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# --- Evaluation ---
print("\n📊 Evaluating Model...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_classes, target_names=le.classes_, zero_division=0))

model.save(os.path.join(MODEL_DIR, "motion_model_final.h5"))
print("✅ Motion model saved.")