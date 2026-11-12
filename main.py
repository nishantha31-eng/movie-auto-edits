import os
import glob
import random
import subprocess
import requests
import numpy as np
import librosa
from PIL import Image, ImageEnhance
from scipy.io.wavfile import write as wav_write
from icrawler.builtin import BingImageCrawler
from moviepy.editor import ImageClip, AudioFileClip, ColorClip, CompositeVideoClip, concatenate_videoclips

MOVIE_TOPICS = [
    {"query": "Interstellar Christopher Nolan cinematic scene wallpaper 4k", "caption": "Time doesn't wait for anyone. 🌌⏳ #interstellar #christophernolan #cinema #movieedits"},
    {"query": "Batman The Dark Knight Joker cinematic dark aesthetic wallpaper", "caption": "Some men just want to watch the world burn. 🦇🃏 #thedarkknight #batman #joker #movieedits"},
    {"query": "John Wick Keanu Reeves neon rain action scene wallpaper", "caption": "Everything's got a price. 🕴️🔫 #johnwick #keanureeves #actionmovies #edits"},
    {"query": "Oppenheimer Cillian Murphy cinematic dramatic scene wallpaper", "caption": "Now I am become Death. 💥🎬 #oppenheimer #cillianmurphy #cinema #reels"},
    {"query": "Blade Runner 2049 Ryan Gosling neon cyberpunk aesthetic wallpaper", "caption": "All the best memories are hers. 🌧️🌆 #bladerunner2049 #ryangosling #aesthetic #movieedits"},
    {"query": "Dune Part Two Paul Atreides desert cinematic wallpaper 4k", "caption": "Long live the fighters. 🏜️⚔️ #dune #timotheechalamet #scifi #cinema"}
]

CINEMATIC_MUSIC_QUERIES = [
    "dark cinematic ambient phonk slowed",
    "interstellar theme phonk remix",
    "hans zimmer epic dark synth edit",
    "slowed reverb dark atmospheric beat"
]

def generate_fallback_beat(output_audio="bg_track.wav", duration=16, bpm=95):
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    beat_interval = 60.0 / bpm
    beat_pos = (t % beat_interval) / beat_interval
    # Deep cinematic sub-bass pulse
    kick = np.sin(2 * np.pi * (45 + 75 * np.exp(-beat_pos * 14)) * t) * np.exp(-beat_pos * 5)
    drone = 0.25 * np.sin(2 * np.pi * 65.41 * t) * np.sin(2 * np.pi * 0.25 * t)
    audio = np.clip(kick * 0.85 + drone, -1.0, 1.0)
    wav_write(output_audio, sr, (audio * 32767).astype(np.int16))
    return output_audio

def fetch_auto_audio(output_audio="bg_track.mp3"):
    # 1. SoundCloud slowed/cinematic edit tracks
    try:
        query = random.choice(CINEMATIC_MUSIC_QUERIES)
        pick = random.randint(1, 8)
        cmd = [
            "yt-dlp",
            f"scsearch10:{query}",
            "--playlist-items", str(pick),
            "-x", "--audio-format", "mp3",
            "--match-filter", "duration >= 25 & duration <= 300",
            "-o", output_audio
        ]
        subprocess.run(cmd, check=True, timeout=45)
        if os.path.exists(output_audio) and os.path.getsize(output_audio) > 100_000:
            print("Loaded SoundCloud cinematic track!")
            return output_audio
    except Exception as e:
        print("SoundCloud fetch skipped:", e)

    # 2. Backup: Audius free ambient/cinematic tracks
    try:
        res = requests.get(
            "https://audius.co",
            params={"query": "cinematic dark ambient", "app_name": "MovieEditsBot"},
            timeout=15
        ).json()
        tracks = res.get("data", [])
        if tracks:
            chosen = random.choice(tracks[:10])
            stream_url = f"https://audius.co{chosen['id']}/stream?app_name=MovieEditsBot"
            dl = requests.get(stream_url, timeout=25)
            if dl.status_code == 200 and len(dl.content) > 100_000:
                with open(output_audio, "wb") as f:
                    f.write(dl.content)
                return output_audio
    except Exception as e:
        print("Audius fetch skipped:", e)

    return generate_fallback_beat("bg_track.wav")

def download_images(query, count=8):
    os.makedirs("raw_imgs", exist_ok=True)
    os.makedirs("imgs", exist_ok=True)
    crawler = BingImageCrawler(storage={"root_dir": "raw_imgs"})
    crawler.crawl(keyword=query, max_num=25)

    paths = []
    for raw_path in glob.glob("raw_imgs/*"):
        if len(paths) >= count:
            break
        try:
            img = Image.open(raw_path).convert("RGB")
            if img.width >= 500 and img.height >= 500:
                target_ratio = 1080 / 1920
                img_ratio = img.width / img.height
                if img_ratio > target_ratio:
                    new_w = int(img.height * target_ratio)
                    left = (img.width - new_w) // 2
                    img = img.crop((left, 0, left + new_w, img.height))
                else:
                    new_h = int(img.width / target_ratio)
                    top = (img.height - new_h) // 2
                    img = img.crop((0, top, img.width, top + new_h))
                img = img.resize((1080, 1920), Image.Resampling.LANCZOS)
                # Moody cinematic color grade: higher contrast, slightly deeper shadows
                img = ImageEnhance.Contrast(img).enhance(1.22)
                img = ImageEnhance.Brightness(img).enhance(0.92)
                img = ImageEnhance.Color(img).enhance(1.10)
                out_path = f"imgs/img_{len(paths)}.jpg"
                img.save(out_path, quality=95)
                paths.append(out_path)
        except Exception:
            continue
    return paths

def build_cinematic_edit(image_paths, audio_path, output_path="latest_movie_edit.mp4"):
    total_dur = AudioFileClip(audio_path).duration
    start_offset = 20.0 if total_dur > 40.0 else 0.0

    y, sr = librosa.load(audio_path, offset=start_offset, duration=16)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    # Dramatic pacing: cuts every 3rd beat so scenes breathe
    cuts = [0.0]
    for b in beat_times[2::3]:
        if b <= 14.5 and len(cuts) < len(image_paths):
            cuts.append(float(b))
    cuts.append(min(16.0, cuts[-1] + 2.2))

    clips = []
    for idx, img_path in enumerate(image_paths[:len(cuts)-1]):
        duration = max(0.9, cuts[idx+1] - cuts[idx])
        if idx % 2 == 0:
            base_clip = (ImageClip(img_path)
                         .set_duration(duration)
                         .resize(lambda t, d=duration: 1.0 + 0.09 * (t / d))
                         .set_position("center"))
        else:
            base_clip = (ImageClip(img_path)
                         .set_duration(duration)
                         .resize(lambda t, d=duration: 1.09 - 0.09 * (t / d))
                         .set_position("center"))

        # Subtle dark/white cinematic flash on scene transition
        flash = (ColorClip(size=(1080, 1920), color=(255, 255, 255))
                 .set_duration(min(0.06, duration * 0.2))
                 .set_opacity(0.35))
        comp = CompositeVideoClip([base_clip, flash], size=(1080, 1920)).set_duration(duration)
        clips.append(comp)

    video = concatenate_videoclips(clips, method="compose")
    audio = AudioFileClip(audio_path).subclip(start_offset, start_offset + video.duration)
    video = video.set_audio(audio)
    video.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac")
    return output_path

if __name__ == "__main__":
    choice = random.choice(MOVIE_TOPICS)
    print("Selected movie:", choice["query"])
    print("Suggested caption:", choice["caption"])
    images = download_images(choice["query"], count=8)
    chosen_audio = fetch_auto_audio("bg_track.mp3")
    build_cinematic_edit(images, chosen_audio, "latest_movie_edit.mp4")
