from flask import Flask, render_template, request, redirect, url_for, jsonify
import subprocess
import sys
import base64
import numpy as np
import cv2
from keras.models import load_model
import mediapipe as mp
from collections import deque
import threading
import os
from sentence_to_video import generate_sign_video_from_sentence
import time

app = Flask(__name__, static_folder="static")
os.makedirs(os.path.join(app.static_folder, "generated"), exist_ok=True)

model = load_model("model/cnn_model.keras", compile=False)
labels = np.load("model/labels.npy")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
sentence = deque(maxlen=3)
lock = threading.Lock()
sentence_proc = None

def extract_features(result):
    features = []
    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks[:2]:
            for lm in hand.landmark:
                features.extend([lm.x, lm.y, lm.z])
    while len(features) < 126:
        features.extend([0.0] * 63)
    return np.array(features).reshape(1, 126, 1)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form["name"]
        age = request.form["age"]
        return redirect(url_for("menu", name=name, age=age))
    return render_template("index.html")

@app.route("/menu")
def menu():
    name = request.args.get("name")
    age = request.args.get("age")
    return render_template("menu.html", name=name, age=age)

@app.route("/sentence")
def sentence_start():
    global sentence_proc
    sentence_proc = subprocess.Popen([sys.executable, "live_cnn_recognition.py"])
    return "<h2>Sentence recognition started. Check webcam window. <a href='/stop_sentence'>Stop</a></h2>"

@app.route("/stop_sentence")
def stop_sentence():
    global sentence_proc
    if sentence_proc and sentence_proc.poll() is None:
        sentence_proc.terminate()
        sentence_proc = None
        return "<h2>Sentence recognition stopped.</h2>"
    return "<h2>No active sentence recognition process.</h2>"

@app.route("/webcam")
def webcam():
    return render_template("webcam.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data_url = request.json.get("image")
    if not data_url:
        return jsonify({"error": "no_image"}), 400
    header, b64 = data_url.split(",", 1)
    img_bytes = base64.b64decode(b64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    with lock:
        result = hands.process(rgb)
        word = ""
        conf = 0.0
        if result.multi_hand_landmarks:
            features = extract_features(result)
            pred = model.predict(features, verbose=0)
            idx = int(np.argmax(pred))
            word = str(labels[idx])
            conf = float(pred[0][idx])
            if conf > 0.7:
                if not sentence or sentence[-1] != word:
                    sentence.append(word)
        return jsonify({"word": word, "confidence": conf, "sentence": " ".join(sentence)})

@app.route("/video", methods=["GET", "POST"])
def video():
    output_rel = "sentence_sign_video.mp4"
    output_abs = os.path.join(app.static_folder, output_rel)
    if request.method == "POST":
        text = request.form.get("text", "").strip()
        if text:
            generate_sign_video_from_sentence(text, output_path=output_abs)
        cache_bust = f"?t={int(time.time())}"
        return render_template("video.html", video_path=f"/static/{output_rel}{cache_bust}", text=text)
    return render_template("video.html", video_path=None, text="")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)