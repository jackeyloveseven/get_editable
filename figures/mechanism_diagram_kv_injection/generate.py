#!/usr/bin/env python3
"""
generate.py -- calls Kaon Router's GPT-Image-2 endpoint to produce
mechanism_diagram_kv_injection.png from the prompt in PROMPT.md.

Not OpenRouter: this project's available credential is a Kaon Router key
(sourced the same way probe/38_gemini_judge.py sources KAON_KEY, from
/home/ubuntu/smoke_anime/flash_caption_smoke.py), and the working
base_url/model pair was verified with a throwaway test call before this
script was written:
  base_url = https://kaon-router.kaonai.com/v1
  model    = huskyi/gpt-image-2   (also available: baihaii/gpt-image-2-i)
(the openrouter.ai-hardcoded script bundled with the nature-figure skill,
scripts/generate_openrouter_schematic.py, was NOT used here because this
project has no OPENROUTER_API_KEY -- only a Kaon Router key.)

Run:
  python3 generate.py
"""
import base64
import json
import os
import re
import sys

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
KEY_SRC = "/home/ubuntu/smoke_anime/flash_caption_smoke.py"
KAON_BASE = "https://kaon-router.kaonai.com/v1"
MODEL = "huskyi/gpt-image-2"


def _kaon_key():
    text = open(KEY_SRC).read()
    m = re.search(r'KAON_KEY\s*=\s*"([^"]+)"', text)
    if not m:
        raise RuntimeError(f"could not find KAON_KEY in {KEY_SRC}")
    return m.group(1)


def _extract_prompt():
    prompt_md = open(os.path.join(HERE, "PROMPT.md")).read()
    m = re.search(r"## Final prompt.*?```\n(.*?)\n```", prompt_md, re.S)
    if not m:
        raise RuntimeError("could not find fenced prompt block in PROMPT.md")
    return m.group(1).strip()


def main():
    key = _kaon_key()
    prompt = _extract_prompt()
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "n": 1,
        "size": "1536x1024",
    }
    print(f"[generate] calling {KAON_BASE}/images/generations model={MODEL} ...")
    r = requests.post(
        f"{KAON_BASE}/images/generations",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    if r.status_code != 200:
        print(f"[generate] FAILED status={r.status_code}", file=sys.stderr)
        print(r.text[:2000], file=sys.stderr)
        sys.exit(1)
    data = r.json()
    b64 = data["data"][0]["b64_json"]
    out_png = os.path.join(HERE, "mechanism_diagram_kv_injection.png")
    with open(out_png, "wb") as f:
        f.write(base64.b64decode(b64))
    meta = {
        "model": MODEL,
        "base_url": KAON_BASE,
        "size_requested": payload["size"],
        "response_top_level_keys": list(data.keys()),
    }
    with open(os.path.join(HERE, "request_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[generate] saved {out_png}")


if __name__ == "__main__":
    main()
