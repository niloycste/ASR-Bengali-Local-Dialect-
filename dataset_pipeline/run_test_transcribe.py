# -*- coding: utf-8 -*-
"""
Quick transcription test - 4 clips per dialect.
Run with the ASR conda Python:
    ASR_python run_test_transcribe.py
"""
import sys, os, json, torch, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import imageio_ffmpeg
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
print(f"ffmpeg : {ffmpeg_exe}")

import whisper
from pydub import AudioSegment
from collections import defaultdict

AudioSegment.converter = ffmpeg_exe   # pydub uses imageio ffmpeg


def load_wav(path: str) -> np.ndarray:
    audio = (AudioSegment.from_wav(path)
             .set_frame_rate(16000).set_channels(1).set_sample_width(2))
    return np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32) / 32768.0


# Common English words that appear naturally in Bengali CS speech (loan words)
REAL_CS_WORDS = {
    # tech
    "phone","internet","facebook","video","call","charge","battery","network",
    "sim","app","online","mobile","email","computer","laptop","software",
    "system","api","server","data","code","project","deploy","database",
    # medical
    "doctor","medicine","hospital","operation","patient","blood","pressure",
    "test","report","ecg","icu","oxygen","covid","vaccine","injection",
    # office/education
    "office","meeting","class","exam","result","pass","fail","certificate",
    "job","apply","form","interview","salary","manager","boss","sir","madam",
    # daily
    "okay","ok","yes","no","problem","sorry","please","thank","thanks",
    "good","bad","nice","cool","right","hello","bye","bus","car","road",
    "market","bank","shop","hotel","restaurant",
    # sports/culture
    "football","cricket","team","match","player","goal","win","game",
    "actor","actress","movie","film","song","music",
    # countries/places used in conversation
    "india","pakistan","england","america","london","dubai","canada",
}

def is_real_english(word: str) -> bool:
    """
    Return True only if the word is a plausible real English code-switch,
    not a Whisper hallucination.
    Rules:
      - Must be >= 3 characters (filters 'wo', 'mo', 'we', etc.)
      - Must be in known CS word list OR be a proper noun pattern
      - Short common words (is, in, at, a, i) are allowed only if in known list
    """
    w = word.lower().strip(".,!?-")
    if len(w) < 3:
        return False
    if w in REAL_CS_WORDS:
        return True
    # Accept capitalized words >= 4 chars as potential named entities
    # (Argentina, Actor, Bangladesh etc.)
    if word[0].isupper() and len(w) >= 4 and w.isalpha():
        return True
    return False


def classify(tags: list) -> tuple:
    bn_words = [t for t in tags if t["lang"] == "BN"]
    # Only count REAL English words, not hallucinations
    en_words_real = [t for t in tags if t["lang"] == "EN" and is_real_english(t["word"])]

    bn = len(bn_words)
    en = len(en_words_real)
    total = bn + en

    if not total:
        return "NOISE", 0.0, 0.0, []

    bn_r = bn / total
    en_r = en / total

    # Transcript with no Bengali chars = wrong video or pure English
    has_bengali_script = bn > 0

    if not has_bengali_script:
        label = "EN_ONLY"
    elif en_r >= 0.90:
        label = "EN_ONLY"
    elif bn_r >= 0.95 or en == 0:
        label = "BN_ONLY"
    else:
        label = "CS"   # has both Bengali AND real English words

    en_word_list = [t["word"] for t in en_words_real]
    return label, round(bn_r, 2), round(en_r, 2), en_word_list


# ── Load model ────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device : {device}")
print("Loading Whisper small ...")
model = whisper.load_model("small", device=device)
print("Model ready.\n")

# ── Load clips ────────────────────────────────────────────────────────────────
with open("test_run/manifest.json", encoding="utf-8") as f:
    all_clips = json.load(f)

per_dialect: dict = defaultdict(list)
for c in all_clips:
    per_dialect[c.get("dialect", "unknown")].append(c)

sample = []
for clips in per_dialect.values():
    sample.extend(clips[:4])

print(f"Clips to transcribe : {len(sample)} (4 per dialect)\n")

# ── Transcribe ────────────────────────────────────────────────────────────────
stats: dict = defaultdict(lambda: {"CS": 0, "BN_ONLY": 0, "EN_ONLY": 0, "NOISE": 0, "en_words": []})
results = []

for clip in sample:
    if not os.path.exists(clip["path"]):
        continue

    audio_np = load_wav(clip["path"])
    res  = model.transcribe(audio_np, language="bn",
                            word_timestamps=True, verbose=False)
    text = res["text"].strip()

    tags = []
    for seg in res.get("segments", []):
        for w in seg.get("words", []):
            word = w["word"].strip()
            has_bn = any("\u0980" <= c <= "\u09FF" for c in word)
            if has_bn or not word.isascii():
                tags.append({"word": word, "lang": "BN"})
            elif word.isalpha() and len(word) > 1:
                tags.append({"word": word, "lang": "EN"})

    label, bn_r, en_r, en_words = classify(tags)
    d = clip.get("dialect", "unknown")
    stats[d][label] += 1
    stats[d]["en_words"].extend(en_words[:3])

    icon  = {"CS": "[CS ]", "BN_ONLY": "[BN ]", "NOISE": "[-- ]"}.get(label, "[?? ]")
    en_pct = int(en_r * 100)
    # Print transcript safe for ASCII terminal
    text_p = text[:55].encode("ascii", "replace").decode()
    en_p   = [w.encode("ascii", "replace").decode() for w in en_words[:3]]
    print(f"{icon} [{d:12s}] {en_pct:2d}%EN  en={en_p}  {text_p}")

    results.append({**clip, "transcript": text, "label": label,
                    "is_code_switched": label == "CS", "en_words": en_words})

# ── Quality report ────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("QUALITY REPORT — is the downloaded data code-switched?")
print("=" * 68)

for d, s in sorted(stats.items()):
    total  = s["CS"] + s["BN_ONLY"] + s["NOISE"]
    cs_pct = int(s["CS"] / max(total, 1) * 100)
    en_w   = list(dict.fromkeys(s["en_words"]))[:5]
    en_w_p = [w.encode("ascii", "replace").decode() for w in en_w]

    if cs_pct >= 30:
        verdict = "GOOD — dialect + English mixing confirmed"
    elif cs_pct >= 10:
        verdict = "SOME — limited English mixing"
    else:
        verdict = "LOW  — mostly pure dialect, little English"

    print(f"\n  Dialect  : {d}")
    print(f"  CS clips : {s['CS']}/{total}  ({cs_pct}%)")
    print(f"  EN words : {en_w_p}")
    print(f"  Verdict  : {verdict}")

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = "test_run/sample_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nResults saved: {out_path}")
