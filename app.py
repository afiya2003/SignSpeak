from flask import Flask, render_template, Response, request, jsonify, send_file
from flask_cors import CORS
import os

# --- IMPORT YOUR CUSTOM LOGIC ---
# Ensure live_recognition.py DOES NOT contain cv2.imshow()
from sentence_to_video import generate_sign_video_from_sentence

app = Flask(__name__)
CORS(app)

# 1. LANDING & DASHBOARD
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# 2. LIVE RECOGNITION (WEBCAM)
@app.route("/webcam")
def webcam():
    return render_template("webcam.html")

@app.route('/video_feed')
def video_feed():
    """Streams the camera feed with CNN overlays."""
    # This calls the generator from your live_recognition.py
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/get_recognition_data")
def get_recognition_data():
    """Provides the current word and sentence as JSON for the UI."""
    try:
        # Convert the current token buffer into a readable sentence
        sentence = tokens_to_sentence(list(token_buffer))
        return jsonify({
            "letter": stable_label,
            "sentence": sentence if sentence else "Waiting for signs..."
        })
    except Exception as e:
        return jsonify({"letter": "Error", "sentence": str(e)})

# 3. TEXT TO SIGN (VIDEO GENERATION)
@app.route("/text-to-sign")
def text_to_sign():
    return render_template("text_to_sign.html")

@app.route("/generate-video", methods=["POST"])
def generate_video_api():
    try:
        data = request.get_json()
        sentence = data.get("sentence")
        output_path = "sentence_sign_video.mp4"

        result_metadata = generate_sign_video_from_sentence(
            sentence,
            sign_video_dir="sign_videos",
            output_path=output_path
        )

        if result_metadata:
            return jsonify({
                "video_url": "/video-output",
                "word_timings": result_metadata 
            })
        else:
            return jsonify({"error": "No video generated"}), 400
    except Exception as e:
        print("Generate error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/video-output")
def video_output():
    try:
        video_path = os.path.join(os.getcwd(), "sentence_sign_video.mp4")
        if not os.path.exists(video_path):
            return "Video not found", 404
            
        response = send_file(video_path, mimetype="video/mp4")
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response
    except Exception as e:
        return "Video error", 404

# 4. UTILITIES
@app.route("/suggest")
def suggest():
    try:
        query = request.args.get("q", "").lower()
        words = []
        if not os.path.exists("sign_videos"):
            return jsonify([])
        for file in os.listdir("sign_videos"):
            if file.endswith(".mp4"):
                word = file.replace(".mp4", "")
                if word.startswith(query):
                    words.append(word)
        return jsonify(words[:5])
    except Exception as e:
        return jsonify([])

@app.route("/history")
def history():
    return render_template("history.html")

@app.route("/settings")
def settings():
    return render_template("settings.html")

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)