"""
Synthetic Code-Switched Data Generation for Bengali CS ASR.

WHEN TO RUN:
  - ONLY if CS clips < 3,000 after Step 3 transcription — extreme data shortage
  - Run AFTER 04b_auto_annotate.py and BEFORE 05_build_dataset.py
  - This is a last resort — real speech always preferred over synthetic

WHEN NOT TO RUN:
  - If you have 5,000+ CS clips — not needed, won't help much
  - Do NOT use synthetic data in the TEST SET — only training data
  - Do NOT run before you know your real CS clip count

What this does:
  Two strategies:

  Strategy A — Text-only synthetic transcripts (FAST, no TTS)
    Fill Bengali sentence templates with English words from a curated word bank.
    These transcripts are added to a separate manifest for training.
    Note: You need REAL audio for these — either:
      (a) Pair with TTS audio (Strategy B)
      (b) Use these texts to augment language model training only

  Strategy B — TTS audio + text (SLOW, needs gTTS/Coqui)
    Generate MP3/wav using gTTS for each synthetic transcript.
    Quality is lower than real speech but adds training variety.
    Only use if you have < 3,000 CS clips.

Usage:
    python generate_synthetic_cs.py                     # text-only (Strategy A)
    python generate_synthetic_cs.py --tts               # text + TTS audio (Strategy B)
    python generate_synthetic_cs.py --count 2000        # generate 2000 sentences
    python generate_synthetic_cs.py --tts --count 500   # 500 TTS clips (slow)
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
SYNTH_DIR    = SCRIPT_DIR / "segments" / "synthetic"
OUTPUT_JSON  = SCRIPT_DIR / "transcripts" / "synthetic_cs_transcripts.json"

# ── Sentence templates ────────────────────────────────────────────────────────
# [EN_NOUN], [EN_VERB], [EN_ADJ], [EN_ADV] are placeholders filled from word banks below.
# Bengali text is kept in romanized form here but stored as Bengali Unicode.

TEMPLATES = [
    # Pattern: Bengali statement + English noun
    "আমি [EN_NOUN] টা দেখছি",
    "এই [EN_NOUN] টা অনেক ভালো",
    "আমার [EN_NOUN] টা নষ্ট হয়ে গেছে",
    "তোমার [EN_NOUN] কোথায়",
    "নতুন [EN_NOUN] কিনলাম",
    "এই [EN_NOUN] টার দাম কত",
    "আমি [EN_NOUN] use করি",
    "[EN_NOUN] টা এখানে রাখো",
    "সেরা [EN_NOUN] কোনটা",
    "আমার [EN_NOUN] দরকার",

    # Pattern: Bengali + English verb
    "আমি [EN_VERB] করতে পারব না",
    "তুমি কি [EN_VERB] করেছ",
    "এটা [EN_VERB] করা দরকার",
    "আমি [EN_VERB] করতে চাই",
    "সে [EN_VERB] করছে",
    "কখন [EN_VERB] করবে",
    "এখনই [EN_VERB] করো",
    "তুমি [EN_VERB] করো নাই কেন",
    "আমি already [EN_VERB] করেছি",
    "সে [EN_VERB] করতে পারে না",

    # Pattern: Bengali + English adjective
    "এটা [EN_ADJ] না",
    "অনেক [EN_ADJ] লাগছে",
    "এত [EN_ADJ] কেন",
    "এটা সত্যিই [EN_ADJ]",
    "ব্যাপারটা [EN_ADJ]",
    "এই জিনিসটা [EN_ADJ]",
    "তুমি কি [EN_ADJ] feel করছ",
    "situation টা [EN_ADJ]",
    "idea টা [EN_ADJ]",
    "এটা একদম [EN_ADJ]",

    # Pattern: Bengali + English adverb (high frequency in real speech)
    "[EN_ADV] বলছি এটা ঠিক না",
    "[EN_ADV] এটা কঠিন",
    "[EN_ADV] আমি জানি না",
    "এটা [EN_ADV] ভালো",
    "[EN_ADV] এটা possible না",
    "[EN_ADV] এই কাজটা হবে না",
    "[EN_ADV] এটা অনেক expensive",
    "[EN_ADV] তোমার সাথে কথা বলব",
    "[EN_ADV] এটা important",
    "[EN_ADV] দেখো কী হয়",

    # Pattern: Mixed sentence (multiple English words)
    "আমি [EN_VERB] করে [EN_NOUN] টা [EN_VERB] করব",
    "[EN_ADV] বলছি এই [EN_NOUN] টা [EN_ADJ]",
    "তুমি [EN_NOUN] এ [EN_VERB] করো",
    "এই [EN_NOUN] টা [EN_ADJ] কিন্তু price অনেক বেশি",
    "আমার [EN_NOUN] টা [EN_VERB] করতে হবে",
    "[EN_ADV] তুমি [EN_NOUN] টা [EN_VERB] করো",
    "এটা [EN_ADJ] problem না",
    "তোমার [EN_NOUN] কি [EN_ADJ]",

    # Platform/Tech specific (high frequency in comedy/vlog content)
    "এই video টা [EN_VERB] করো",
    "channel টা subscribe করো",
    "comment এ জানাও",
    "like দাও আর share করো",
    "notification bell টা [EN_VERB] করো",
    "আমার account এ problem হচ্ছে",
    "internet connection টা [EN_ADJ]",
    "battery টা [EN_ADJ] হয়ে গেছে",
    "এই app টা [EN_ADJ]",
    "phone টা update করো",

    # Academic (students)
    "exam এ [EN_ADJ] করেছি",
    "assignment টা [EN_VERB] করতে হবে",
    "result টা [EN_ADJ] হয়েছে",
    "class এ [EN_VERB] করতে পারিনি",
    "professor আমাকে [EN_NOUN] দিয়েছে",

    # Finance/Daily
    "bkash এ payment [EN_VERB] করো",
    "bank এ [EN_VERB] করতে হবে",
    "এই product টার price [EN_ADJ]",
    "delivery কবে আসবে",
    "order টা [EN_VERB] করেছ",
]

# ── Word banks ─────────────────────────────────────────────────────────────────

EN_NOUNS = [
    "phone", "laptop", "camera", "battery", "charger", "screen", "app",
    "video", "channel", "account", "password", "internet", "wifi", "data",
    "sim", "selfie", "screenshot", "notification", "comment", "post",
    "meeting", "office", "class", "assignment", "exam", "result", "project",
    "report", "file", "job", "career", "interview", "internship",
    "payment", "bank", "balance", "loan", "transaction",
    "gym", "diet", "coffee", "party", "trip", "ticket", "hotel",
    "bus", "driver", "delivery", "order", "booking",
    "problem", "solution", "idea", "plan", "decision", "experience",
    "game", "challenge", "content", "reaction",
    "brand", "style", "fashion", "makeup",
    "podcast", "stream", "vlog", "story",
    "doctor", "hospital", "medicine", "test",
    "friend", "group", "team", "partner",
]

EN_VERBS = [
    "try", "check", "fix", "update", "delete", "copy", "paste", "save",
    "send", "receive", "reply", "follow", "like", "share", "subscribe",
    "upload", "download", "install", "login", "register", "verify",
    "submit", "manage", "handle", "confirm", "cancel", "book", "order",
    "call", "record", "edit", "post", "comment",
    "support", "connect", "join", "leave",
]

EN_ADJS = [
    "good", "bad", "amazing", "awesome", "perfect", "fine", "okay",
    "serious", "funny", "crazy", "busy", "ready", "done", "free",
    "fast", "slow", "easy", "difficult", "different", "same", "new",
    "big", "small", "extra", "full", "popular", "official", "real", "fake",
    "positive", "negative", "important", "interesting", "special", "normal",
    "expensive", "cheap", "professional", "original", "digital", "online",
    "offline", "total", "next", "last", "best", "worst",
]

EN_ADVS = [
    "seriously", "actually", "basically", "honestly", "literally",
    "obviously", "definitely", "exactly", "totally", "absolutely",
    "probably", "apparently", "technically", "officially", "personally",
    "normally", "finally", "directly", "quickly", "suddenly",
    "already", "still", "just", "really", "pretty", "quite",
]


def fill_template(template: str) -> str:
    """Replace placeholders in a template with random words from the banks."""
    t = template
    t = t.replace("[EN_NOUN]", random.choice(EN_NOUNS))
    t = t.replace("[EN_VERB]", random.choice(EN_VERBS))
    t = t.replace("[EN_ADJ]",  random.choice(EN_ADJS))
    t = t.replace("[EN_ADV]",  random.choice(EN_ADVS))
    return t


def generate_transcripts(count: int = 2000) -> list[dict]:
    """Generate `count` unique synthetic CS transcripts."""
    seen    = set()
    results = []

    attempts = 0
    while len(results) < count and attempts < count * 10:
        attempts += 1
        template   = random.choice(TEMPLATES)
        transcript = fill_template(template)
        if transcript in seen:
            continue
        seen.add(transcript)

        results.append({
            "clip_id":          f"synth_{len(results):05d}",
            "source_id":        "synthetic",
            "audio_path":       "",                    # filled in if TTS is run
            "transcript":       transcript,
            "human_transcript": transcript,
            "dialect":          "synthetic",
            "domain":           "synthetic",
            "is_code_switched": True,
            "bn_ratio":         0.5,
            "en_ratio":         0.5,
            "duration_sec":     0.0,                   # filled in if TTS is run
            "word_tags":        [],
            "augmented":        False,
            "synthetic":        True,
        })

        if len(results) % 500 == 0:
            print(f"  [Progress] Generated {len(results)} / {count} text sentences...")

    print(f"Generated {len(results)} unique transcripts from {len(TEMPLATES)} templates")
    return results


def generate_tts_audio(clips: list[dict]) -> list[dict]:
    """
    Generate TTS audio for synthetic transcripts using gTTS.
    Mixed-language text is sent as-is — gTTS handles Bengali+English reasonably.
    Requires: pip install gTTS soundfile
    """
    try:
        from gtts import gTTS
    except ImportError:
        print("[ERROR] gTTS not installed. Run: pip install gTTS")
        raise SystemExit(1)

    try:
        import soundfile as sf
        import numpy as np
        from pydub import AudioSegment
        import io
    except ImportError:
        print("[ERROR] pip install soundfile pydub")
        raise SystemExit(1)

    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    saved = 0

    for i, clip in enumerate(clips):
        transcript = clip["transcript"]
        out_path   = str(SYNTH_DIR / f"{clip['clip_id']}.wav")

        print(f"  [{i+1}/{len(clips)}] TTS: {transcript[:55]}...")

        try:
            # gTTS produces MP3 — convert to wav 16kHz mono
            tts = gTTS(text=transcript, lang="bn", slow=False)
            mp3_buf = io.BytesIO()
            tts.write_to_fp(mp3_buf)
            mp3_buf.seek(0)

            audio_seg = AudioSegment.from_mp3(mp3_buf)
            audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)
            audio_seg.export(out_path, format="wav")

            duration = len(audio_seg) / 1000.0  # ms → sec
            clip["audio_path"]   = out_path
            clip["duration_sec"] = round(duration, 3)
            saved += 1

            # Increased delay to prevent Google from IP banning
            time.sleep(2.0)

        except Exception as e:
            print(f"    [WARN] TTS failed for clip {clip['clip_id']}: {e}")
            continue

    print(f"\nTTS audio saved: {saved}/{len(clips)}")
    return [c for c in clips if c["audio_path"]]


def run(count: int = 2000, use_tts: bool = False):
    print(f"\n{'='*60}")
    print(f"Synthetic CS data generation")
    print(f"  Count   : {count}")
    print(f"  TTS     : {'YES (Strategy B)' if use_tts else 'NO — text only (Strategy A)'}")
    print(f"{'='*60}\n")

    clips = generate_transcripts(count)

    if use_tts:
        print(f"\nGenerating TTS audio for {len(clips)} clips...")
        clips = generate_tts_audio(clips)
        if not clips:
            print("[ERROR] No TTS audio generated.")
            return

    # Save to JSON
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(clips, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {OUTPUT_JSON}")

    if use_tts:
        # Merge into transcripts_reviewed.json so 05_build_dataset.py picks them up
        reviewed_path = SCRIPT_DIR / "transcripts" / "transcripts_reviewed.json"
        if reviewed_path.exists():
            with open(reviewed_path, encoding="utf-8") as f:
                existing = json.load(f)
            merged = existing + clips
            with open(reviewed_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            print(f"Merged {len(clips)} synthetic clips into transcripts_reviewed.json")
            print(f"Total clips: {len(merged)}")
        else:
            print(f"[WARN] transcripts_reviewed.json not found — run 04b_auto_annotate.py first")
            print(f"Synthetic clips saved separately to: {OUTPUT_JSON}")
    else:
        print("\n[NOTE] Text-only mode: audio_path is empty.")
        print("  These transcripts are automatically picked up by:")
        print("  → evaluation/09_lm_shallow_fusion.py  (Phase 3, Step 9)")
        print("    The LM builder reads synthetic_cs_transcripts.json to learn")
        print("    code-switching patterns for beam search decoding.")
        print("  To get audio too: re-run with --tts flag")

    print(f"\nNext step in pipeline: dataset_pipeline/05_build_dataset.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000,
                        help="Number of synthetic sentences to generate (default: 2000)")
    parser.add_argument("--tts",   action="store_true",
                        help="Generate TTS audio (Strategy B). Requires pip install gTTS")
    args = parser.parse_args()
    run(count=args.count, use_tts=args.tts)
