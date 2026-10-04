"""
Offcut shorts renderer.

Reads a shorts package (JSON from the Make "Offcut — Shorts Writer" scenario),
and for each short:
  1. records the voiceover per scene with Kokoro (free, open-source TTS)
  2. builds an animated, on-topic graphic for each scene (visuals.py): titles,
     checklists, counting numbers, do/don't cards; about one scene in three uses
     real stock footage from Pixabay (free API), cropped to vertical
  3. joins the 1080x1920 scene clips with FFmpeg, adds the OFFCUT tag and a progress bar
  4. burns in big, bold captions
  5. writes out/<slug>.mp4

Input comes from (in order):
  - a repository_dispatch event (client_payload.shorts_json), or
  - the file given as the first command-line argument (for manual tests).

Environment variables:
  PIXABAY_API_KEY  optional (without it every scene is animated)
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

import visuals as V

SAMPLE_RATE = 24000
W, H, FPS = 1080, 1920, 30
SCENE_GAP = 0.15  # seconds of silence between scenes
OUT = Path("out")
WORK = Path("work")
FALLBACK_SEARCHES = ["construction worker", "carpenter working", "power tools workshop", "building construction site"]


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

OFF_TOPIC_TAGS = {
    "bird", "birds", "animal", "animals", "wildlife", "dog", "dogs", "cat", "cats", "horse", "cow",
    "insect", "bee", "butterfly", "flower", "flowers", "nature", "forest", "tree", "trees", "sea",
    "ocean", "beach", "fish", "woodpecker", "squirrel", "sunset", "sunrise", "sky", "clouds",
    "waterfall", "mountain", "mountains", "lake", "river", "wedding", "party", "dance", "fashion",
    "model", "christmas", "abstract", "background", "particles", "space", "galaxy", "cartoon",
}
QUERY_STOP = {"the", "and", "for", "with", "man", "men", "woman", "person", "people", "closeup", "close", "up"}


def _words(text: str) -> list:
    return re.findall(r"[a-z]+", text.lower())


def _relevance(query_words: list, tags: set) -> int:
    """How many search words show up in the clip's tags (prefix match, so 'measuring' ~ 'measure')."""
    score = 0
    for q in query_words:
        if any(t.startswith(q[:5]) or q.startswith(t[:5]) for t in tags if len(t) > 2):
            score += 1
    return score


def pixabay_search(query: str, used: set) -> str | None:
    key = os.environ["PIXABAY_API_KEY"]
    r = requests.get(
        "https://pixabay.com/api/videos/",
        params={"key": key, "q": query[:100], "per_page": 50, "safesearch": "true"},
        timeout=30,
    )
    r.raise_for_status()
    q_words = [w for w in _words(query) if len(w) > 2 and w not in QUERY_STOP]
    need = 2 if len(q_words) >= 3 else 1  # longer searches must match at least two words
    ranked = []
    for rank, h in enumerate(r.json().get("hits", [])):
        if h["id"] in used:
            continue
        tags = set(_words(h.get("tags", "")))
        if tags & OFF_TOPIC_TAGS:
            continue  # birds, scenery and other off-topic clips
        score = _relevance(q_words, tags)
        if score < need:
            continue
        med = h.get("videos", {}).get("medium", {}) or {}
        portrait = med.get("height", 0) >= med.get("width", 1)
        ranked.append((-score, 0 if portrait else 1, rank, h))
    ranked.sort(key=lambda x: x[:3])
    for _, _, _, h in ranked:
        for size in ("large", "medium", "small"):
            f = h.get("videos", {}).get(size) or {}
            if f.get("url") and min(f.get("width", 0), f.get("height", 0)) >= 720:
                used.add(h["id"])
                print(f"    footage for '{query}': pixabay {h['id']} [{h.get('tags', '')}]")
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
          f"crop={W}:{H},fps={FPS},setsar=1,eq=brightness=-0.05:saturation=1.08,format=yuv420p")
    run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", str(src), "-t", f"{duration:.3f}",
         "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(dest)])


def ass_time(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def build_captions(scenes: list, path: Path):
    """Captions in 2-3 word bursts, timed by character share within each scene."""
    font = V.caption_font_name()
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},78,&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,7,3,2,80,80,450,1

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
    for n, sc in enumerate(short["scenes"]):
        text = sc["text"].strip()
        print(f"  voice {n}: {text[:60]}...")
        audio = np.concatenate([tts(text), gap])
        dur = len(audio) / SAMPLE_RATE
        audio_parts.append(audio)
        scenes.append({"text": text, "start": t, "duration": dur - SCENE_GAP, "clip": dur,
                       "search": sc.get("search"), "visual": sc.get("visual")})
        t += dur

    plans = V.plan_visuals(scenes)
    used_footage: set = set()
    clip_paths = []
    for n, (sc, plan) in enumerate(zip(scenes, plans)):
        clip = wdir / f"clip{n}.mp4"
        print(f"  scene {n}: {plan['type']}")
        if plan["type"] == "footage":
            try:
                if not os.environ.get("PIXABAY_API_KEY"):
                    raise RuntimeError("no PIXABAY_API_KEY")
                raw = get_footage(plan.get("search") or "tradesman working", used_footage, wdir / f"raw{n}.mp4")
                make_scene_clip(raw, sc["clip"], clip)
                clip_paths.append(clip)
                continue
            except Exception as e:  # never let footage break a render
                print(f"    footage unavailable ({e}); animating instead")
                plan = {"type": "statement", "text": V.headline(sc["text"])}
        V.render_animated(plan, sc, sc["clip"], clip)
        clip_paths.append(clip)

    wav = wdir / "voice.wav"
    sf.write(wav, np.concatenate(audio_parts), SAMPLE_RATE)

    concat_list = wdir / "list.txt"
    concat_list.write_text("".join(f"file '{p.resolve()}'\n" for p in clip_paths), encoding="utf-8")
    subs = wdir / "subs.ass"
    build_captions(scenes, subs)
    brand = wdir / "brand.png"
    V.brand_overlay(brand)

    OUT.mkdir(exist_ok=True)
    final = OUT / f"{slug}.mp4"
    fonts_dir = Path(V.__file__).parent / "fonts"
    ass = f"ass={subs}" + (f":fontsdir={fonts_dir}" if fonts_dir.exists() else "")
    graph = (f"[0:v][2:v]overlay=0:0[b];"
             f"[b][3:v]overlay=x='-w+w*t/{t:.3f}':y=0:shortest=1[p];"
             f"[p]{ass}[v]")
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-i", str(wav),
         "-i", str(brand), "-f", "lavfi", "-i", f"color=c=0xFF7A1A:s={W}x12:r={FPS}",
         "-filter_complex", graph, "-map", "[v]", "-map", "1:a",
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
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
