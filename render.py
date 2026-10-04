"""
Offcut shorts renderer.

Reads a shorts package (JSON from the Make "Offcut — Shorts Writer" scenario),
and for each short:
  1. records the voiceover per scene with Kokoro (free, open-source TTS)
  2. finds matching stock footage on Pixabay (free API), cropped to vertical
  3. builds 1080x1920 scene clips with FFmpeg
  4. burns in big, bold captions
  5. writes out/<slug>.mp4

Input comes from (in order):
  - a repository_dispatch event (client_payload.shorts_json), or
  - the file given as the first command-line argument (for manual tests).

Environment variables:
  PIXABAY_API_KEY  required
  VOICE            optional, Kokoro voice (default: am_michael)
  SPEED            optional, speech speed (default: 1.05)
"""

import json
import math
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

SAMPLE_RATE = 24000
W, H, FPS = 1080, 1920, 30
SCENE_GAP = 0.15  # seconds of silence between scenes
OUT = Path("out")
WORK = Path("work")
FALLBACK_SEARCHES = ["construction worker", "tradesman working", "tools workshop", "building site"]


# ---------------------------------------------------------------- input

def load_package() -> dict:
    raw = None
    event_path = os.environ.get("GITHUB_EVENT_PATH")
    if os.environ.get("GITHUB_EVENT_NAME") == "repository_dispatch" and event_path:
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
        raw = event.get("client_payload", {}).get("shorts_json")
    if raw is None:
        if len(sys.argv) < 2:
            sys.exit("No input: pass a JSON file path or trigger via repository_dispatch.")
        raw = Path(sys.argv[1]).read_text(encoding="utf-8")
    if isinstance(raw, dict):
        return raw
    # The AI sometimes wraps JSON in ``` fences; strip them.
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    return json.loads(raw)


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:60] or "short"


# ---------------------------------------------------------------- voice

_pipeline = None


def tts(text: str) -> np.ndarray:
    global _pipeline
    if _pipeline is None:
        from kokoro import KPipeline
        voice = os.environ.get("VOICE", "am_michael")
        _pipeline = KPipeline(lang_code=voice[0])  # 'a' = American, 'b' = British
    voice = os.environ.get("VOICE", "am_michael")
    speed = float(os.environ.get("SPEED", "1.05"))
    chunks = []
    for _, _, audio in _pipeline(text, voice=voice, speed=speed):
        if audio is None:
            continue
        a = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
        chunks.append(a.astype(np.float32))
    if not chunks:
        return np.zeros(int(SAMPLE_RATE * 0.5), dtype=np.float32)
    return np.concatenate(chunks)


# ---------------------------------------------------------------- footage

def pixabay_search(query: str, used: set) -> str | None:
    key = os.environ["PIXABAY_API_KEY"]
    r = requests.get(
        "https://pixabay.com/api/videos/",
        params={"key": key, "q": query[:100], "per_page": 20, "safesearch": "true"},
        timeout=30,
    )
    r.raise_for_status()
    hits = [h for h in r.json().get("hits", []) if h["id"] not in used]
    random.shuffle(hits)
    # prefer portrait clips, then the rest (landscape gets centre-cropped)
    hits.sort(key=lambda h: 0 if h["videos"].get("medium", {}).get("height", 0)
              >= h["videos"].get("medium", {}).get("width", 1) else 1)
    for h in hits:
        for size in ("large", "medium", "small"):
            f = h.get("videos", {}).get(size) or {}
            if f.get("url") and min(f.get("width", 0), f.get("height", 0)) >= 720:
                used.add(h["id"])
                return f["url"]
    return None


def get_footage(query: str, used: set, dest: Path) -> Path:
    for q in [query] + FALLBACK_SEARCHES:
        try:
            link = pixabay_search(q, used)
        except requests.RequestException as e:
            print(f"  Pixabay error for '{q}': {e}")
            link = None
        if link:
            with requests.get(link, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as fh:
                    for part in resp.iter_content(1 << 20):
                        fh.write(part)
            return dest
        print(f"  no footage for '{q}', trying fallback")
    raise RuntimeError(f"No footage found for '{query}' or fallbacks")


# ---------------------------------------------------------------- video

def run(cmd: list):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def make_scene_clip(src: Path, duration: float, dest: Path):
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=increase,"
          f"crop={W}:{H},fps={FPS},setsar=1,format=yuv420p")
    run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(src), "-t", f"{duration:.3f}",
         "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", str(dest)])


def ass_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_captions(scenes: list, path: Path):
    """Captions in 2-3 word bursts, timed by character share within each scene."""
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,DejaVu Sans,86,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,3,2,80,80,620,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = []
    for sc in scenes:
        words = sc["text"].split()
        groups = [words[i:i + 3] for i in range(0, len(words), 3)]
        if len(groups) > 1 and len(groups[-1]) == 1:  # no lonely last word
            groups[-2] += groups.pop()
        groups = [" ".join(g) for g in groups]
        total_chars = sum(len(g) for g in groups) or 1
        t = sc["start"]
        for g in groups:
            d = sc["duration"] * len(g) / total_chars
            text = g.upper().replace("{", "(").replace("}", ")")
            lines.append(f"Dialogue: 0,{ass_time(t)},{ass_time(t + d)},Cap,,0,0,0,,{text}")
            t += d
    path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")


def render_short(short: dict, idx: int) -> Path:
    slug = slugify(short.get("slug") or short.get("title") or f"short-{idx}")
    print(f"\n=== Short {idx}: {slug}")
    wdir = WORK / slug
    wdir.mkdir(parents=True, exist_ok=True)

    gap = np.zeros(int(SAMPLE_RATE * SCENE_GAP), dtype=np.float32)
    audio_parts, scenes, t = [], [], 0.0
    used_footage: set = set()
    clip_paths = []

    for n, sc in enumerate(short["scenes"]):
        text = sc["text"].strip()
        print(f"  scene {n}: {text[:60]}...")
        audio = np.concatenate([tts(text), gap])
        dur = len(audio) / SAMPLE_RATE
        audio_parts.append(audio)
        scenes.append({"text": text, "start": t, "duration": dur - SCENE_GAP})
        t += dur

        raw = get_footage(sc.get("search") or "tradesman working", used_footage, wdir / f"raw{n}.mp4")
        clip = wdir / f"clip{n}.mp4"
        make_scene_clip(raw, dur, clip)
        clip_paths.append(clip)

    wav = wdir / "voice.wav"
    sf.write(wav, np.concatenate(audio_parts), SAMPLE_RATE)

    concat_list = wdir / "list.txt"
    concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in clip_paths), encoding="utf-8")
    silent = wdir / "silent.mp4"
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(silent)])

    subs = wdir / "subs.ass"
    build_captions(scenes, subs)

    OUT.mkdir(exist_ok=True)
    final = OUT / f"{slug}.mp4"
    run(["ffmpeg", "-y", "-i", str(silent), "-i", str(wav),
         "-vf", f"ass={subs}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart", str(final)])
    print(f"  done: {final} ({t:.1f}s, {final.stat().st_size / 1e6:.1f} MB)")
    short["_slug"] = slug
    short["_seconds"] = round(t, 1)
    return final


# ---------------------------------------------------------------- main

def main():
    package = load_package()
    shorts = package.get("shorts", [])
    if not shorts:
        sys.exit("Package has no shorts.")
    for i, short in enumerate(shorts, 1):
        render_short(short, i)

    manifest = {
        "topic": package.get("topic", ""),
        "shorts": [
            {
                "slug": s["_slug"],
                "file": f"{s['_slug']}.mp4",
                "title": s.get("title", ""),
                "caption": s.get("caption", ""),
                "hashtags": s.get("hashtags", []),
                "seconds": s["_seconds"],
            }
            for s in shorts if "_slug" in s
        ],
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nManifest written to out/manifest.json")


if __name__ == "__main__":
    main()
