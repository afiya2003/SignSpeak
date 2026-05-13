import numpy as np
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --- Configuration ---
DATA_DIR = "signs"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. Data Loading & Normalization ---
print("📂 Loading and Normalizing data...")
X, y = [], []

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"Directory '{DATA_DIR}' not found.")

for file in os.listdir(DATA_DIR):
    if file.endswith(".npy"):
        label = file.replace("sign_", "").replace(".npy", "")
        file_path = os.path.join(DATA_DIR, file)
        try:
            data = np.load(file_path)
            for sample in data:
                # Pad if single hand (63 points) to double hand size (126 points)
                if len(sample) == 63:
                    sample = np.concatenate([sample, np.zeros(63)])
                
                if len(sample) == 126:
                    # --- CRITICAL FIX: NORMALIZATION ---
                    # 1. Reshape to (42 points, 3 coords) to access x,y,z
                    temp_reshaped = sample.reshape(-1, 3) 
                    
                    # 2. Find the wrist coordinates (Point 0)
                    base_x, base_y, base_z = temp_reshaped[0]
                    
                    # 3. Subtract wrist from all points (Relative Position)
                    temp_reshaped[:, 0] -= base_x
                    temp_reshaped[:, 1] -= base_y
                    temp_reshaped[:, 2] -= base_z
                    
                    # 4. Flatten back to 1D array
                    normalized_sample = temp_reshaped.flatten()
                    
                    X.append(normalized_sample)
                    y.append(label)
        except Exception as e:
            print(f"Skipping {file}: {e}")

X = np.array(X)
print(f"✅ Loaded {len(X)} samples. Input shape: {X.shape}")

# --- 2. Label Encoding ---
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_cat = tf.keras.utils.to_categorical(y_encoded)
np.save(os.path.join(MODEL_DIR, "labels.npy"), le.classes_)

# --- 3. Stratified Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat, test_size=0.2, stratify=y_encoded, random_state=42
)

# --- 4. The Dense Model (Best for Relative Coordinates) ---
model = Sequential([
    tf.keras.layers.Input(shape=(126,)), # Expecting 126 features
    
    # Layer 1: Learn broad structures
    Dense(128, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    
    # Layer 2: Learn fine details
    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.3),
    
    # Layer 3: Refine
    Dense(32, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),
    
    # Output Layer
    Dense(len(le.classes_), activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# --- 5. Callbacks ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "best_model.h5"),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
]

# --- 6. Training ---
print("🚀 Starting training with Normalization...")
history = model.fit(
    X_train, y_train,
    epochs=150,             
    batch_size=16,            
    validation_data=(X_test, y_test),
    callbacks=callbacks
)

# --- 7. Evaluation ---
print("\n📊 Evaluating Model...")
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(classification_report(y_true, y_pred_classes, target_names=le.classes_, zero_division=0))
model.save(os.path.join(MODEL_DIR, "cnn_model.keras"))
print("✅ Final normalized model saved.")