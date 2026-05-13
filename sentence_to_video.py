import os
import subprocess


def generate_sign_video_from_sentence(sentence, sign_video_dir="sign_videos", output_path="sentence_sign_video.mp4"):

    if not os.path.exists(sign_video_dir):
        print(f"❌ Folder '{sign_video_dir}' not found.")
        return None

    words = [w for w in sentence.lower().split() if w]

    if len(words) > 1:
        words = [w for w in words if w != "a"]

    file_list_path = "file_list.txt"

    existing = []
    missing = []

    with open(file_list_path, "w", encoding="utf-8") as f:
        for word in words:
            video_path = os.path.join(sign_video_dir, f"{word}.mp4")

            if os.path.exists(video_path):
                abs_path = os.path.abspath(video_path)
                f.write(f"file '{abs_path}'\n")
                existing.append(word)
            else:
                missing.append(word)

    if not existing:
        print("⚠ No matching video clips found.")
        os.remove(file_list_path)
        return None

    # 🔥 Use FULL PATH to ffmpeg.exe
    ffmpeg_path = r"C:\ffmpeg\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"

    command = [
        ffmpeg_path,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", file_list_path,
        "-c", "copy",
        output_path
    ]

    print("⚡ Generating video instantly...")
    subprocess.run(command)

    os.remove(file_list_path)

    if os.path.exists(output_path):
        print(f"✅ Video saved at: {output_path}")

        if missing:
            print(f"⚠ Missing clips for: {', '.join(missing)}")

        return output_path

    print("❌ Failed to generate video.")
    return None


if __name__ == "__main__":

    sentence_input = input("Enter the sentence: ").strip()

    if sentence_input:
        output = generate_sign_video_from_sentence(sentence_input)

        if output:
            print("📺 Playing video...")
            os.startfile(output)
    else:
        print("Empty input.")