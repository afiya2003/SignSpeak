import cv2
import os
import numpy as np

LETTERS_FOLDER = "alphabets"
OUTPUT_FOLDER = "generated"

KEY_SIZE = 100
COLS = 10
FPS = 2
FRAME_REPEAT = 12
OUTPUT_VIDEO = os.path.join(OUTPUT_FOLDER, "output.mp4")

typed_word = ""

# Create generated folder if not exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------- Utility ----------

def blank_key():
    return np.ones((KEY_SIZE, KEY_SIZE, 3), dtype=np.uint8) * 255

def load_letter(letter):
    path = os.path.join(LETTERS_FOLDER, f"{letter}.png")
    img = cv2.imread(path)

    if img is None:
        return blank_key()

    return cv2.resize(img, (KEY_SIZE, KEY_SIZE))

# ---------- Video Generator ----------

def generate_video(word):
    frames = []

    for ch in word:
        img_path = os.path.join(LETTERS_FOLDER, f"{ch}.png")
        img = cv2.imread(img_path)

        if img is None:
            continue

        img = cv2.resize(img, (400, 400))

        for _ in range(FRAME_REPEAT):
            frames.append(img)

    if not frames:
        print("❌ No frames to write")
        return None

    h, w, _ = frames[0].shape

    out = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (w, h)
    )

    for f in frames:
        out.write(f)

    out.release()

    print("✅ Video generated:", OUTPUT_VIDEO)
    return OUTPUT_VIDEO

# ---------- Draw Keyboard ----------

def draw_keyboard():
    layout = [
        list("abcdefghij"),
        [""] + list("klmnopqrs"),
        ["", ""] + list("tuvwxyz")
    ]

    keyboard_rows = []

    for row in layout:
        row_images = []

        for key in row:
            if key == "":
                row_images.append(blank_key())
            else:
                row_images.append(load_letter(key))

        while len(row_images) < COLS:
            row_images.append(blank_key())

        keyboard_rows.append(np.hstack(row_images))

    return np.vstack(keyboard_rows)

# ---------- Main Keyboard Function ----------

def run_keyboard():
    global typed_word
    typed_word = ""

    def mouse_callback(event, x, y, flags, param):
        global typed_word

        if event == cv2.EVENT_LBUTTONDOWN:
            col = x // KEY_SIZE
            row = y // KEY_SIZE

            layout = [
                list("abcdefghij"),
                [""] + list("klmnopqrs"),
                ["", ""] + list("tuvwxyz")
            ]

            if row < len(layout) and col < len(layout[row]):
                key = layout[row][col]
                if key != "":
                    typed_word += key
                    print("Typed:", typed_word)

    cv2.namedWindow("Sign Keyboard")
    cv2.setMouseCallback("Sign Keyboard", mouse_callback)

    while True:
        keyboard_img = draw_keyboard()

        # Word display area
        cv2.rectangle(keyboard_img, (0, 300), (1000, 360), (40, 40, 40), -1)

        cv2.putText(
            keyboard_img,
            f"Word: {typed_word}",
            (10, 340),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        cv2.imshow("Sign Keyboard", keyboard_img)
        key = cv2.waitKey(1)

        if key == 13:  # ENTER
            video_path = generate_video(typed_word)
            typed_word = ""
            cv2.destroyAllWindows()
            return video_path

        elif key == 8:  # BACKSPACE
            typed_word = typed_word[:-1]

        elif key == 27:  # ESC
            break

    cv2.destroyAllWindows()