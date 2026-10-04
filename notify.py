"""
After rendering + publishing the GitHub release, tell Make the shorts are ready.

Adds a public download URL to each short in out/manifest.json and POSTs the
manifest to the Make webhook (MAKE_WEBHOOK_URL secret). If the secret is not
set, it just prints the manifest.
"""

import json
import os
from pathlib import Path

import requests

repo = os.environ["GITHUB_REPOSITORY"]  # e.g. youruser/offcut-renderer
tag = os.environ["RELEASE_TAG"]
manifest = json.loads(Path("out/manifest.json").read_text(encoding="utf-8"))

for s in manifest["shorts"]:
    s["video_url"] = f"https://github.com/{repo}/releases/download/{tag}/{s['file']}"
manifest["release_url"] = f"https://github.com/{repo}/releases/tag/{tag}"

print(json.dumps(manifest, indent=2))

hook = os.environ.get("MAKE_WEBHOOK_URL", "").strip()
if hook:
    r = requests.post(hook, json=manifest, timeout=60)
    print(f"Make webhook responded {r.status_code}")
    r.raise_for_status()
else:
    print("MAKE_WEBHOOK_URL not set, skipping webhook.")
