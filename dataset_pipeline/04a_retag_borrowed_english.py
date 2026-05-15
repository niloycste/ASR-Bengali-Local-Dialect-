"""
Step 4a: Retag Bengali-script English borrowed words after transcription.

Whisper often writes English loanwords in Bengali script, for example:
  "ফরম" for "form", "ফোন" for "phone", "ক্যামেরা" for "camera".

The first transcription pass tags words by script only, so those words are
initially BN. This post-processing step retags known borrowed English words as
EN with a borrowed_en marker, then recomputes CS labels and ratios.

Reads:
  transcripts/transcripts.json

Writes:
  transcripts/transcripts_before_borrowed_retag.json
  transcripts/transcripts.json
  transcripts/borrowed_retag_report.txt
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

# Smart fallback for parallel worker directory
if (SCRIPT_DIR / "Data ASR" / "transcripts_merged_all" / "transcripts.json").exists():
    TARGET_DIR = SCRIPT_DIR / "Data ASR" / "transcripts_merged_all"
else:
    TARGET_DIR = SCRIPT_DIR / "transcripts"

TRANSCRIPTS_PATH = TARGET_DIR / "transcripts.json"
BACKUP_PATH = TARGET_DIR / "transcripts_before_borrowed_retag.json"
REPORT_PATH = TARGET_DIR / "borrowed_retag_report.txt"

BN_PUNCT_RE = re.compile(r"^[\s,।.!?;:\"'()\[\]{}<>“”‘’\-–—]+|[\s,।.!?;:\"'()\[\]{}<>“”‘’\-–—]+$")

# Conservative lexicon: Bengali-script forms that are normally English-origin
# content words in ASR/code-switching contexts. Add project-specific variants
# here as you inspect transcripts.
BORROWED_EN_WORDS = {
    "ফরম": "form",
    "ফর্ম": "form",
    "ফোন": "phone",
    "মোবাইল": "mobile",
    "ল্যাপটপ": "laptop",
    "কম্পিউটার": "computer",
    "ক্যামেরা": "camera",
    "কেমেরা": "camera",
    "ভিডিও": "video",
    "অডিও": "audio",
    "চ্যানেল": "channel",
    "কমেন্ট": "comment",
    "শেয়ার": "share",
    "শেয়ার": "share",
    "লাইভ": "live",
    "স্ট্রিম": "stream",
    "কনটেন্ট": "content",
    "কন্টেন্ট": "content",
    "ক্রিয়েটর": "creator",
    "ক্রিয়েটর": "creator",
    "ব্লগ": "blog",
    "ভ্লগ": "vlog",
    "অফিস": "office",
    "স্কুল": "school",
    "কলেজ": "college",
    "ইউনিভার্সিটি": "university",
    "ক্লাস": "class",
    "ফাইল": "file",
    "আপলোড": "upload",
    "ডাউনলোড": "download",
    "সাবমিট": "submit",
    "মিটিং": "meeting",
    "ইন্টারভিউ": "interview",
    "প্রজেক্ট": "project",
    "প্রোজেক্ট": "project",
    "অ্যাসাইনমেন্ট": "assignment",
    "এসাইনমেন্ট": "assignment",
    "এক্সাম": "exam",
    "রিপোর্ট": "report",
    "রিপোট": "report",
    "রেজাল্ট": "result",
    "জব": "job",
    "ক্যারিয়ার": "career",
    "ক্যারিয়ার": "career",
    "ডেভেলপার": "developer",
    "সফটওয়্যার": "software",
    "সফটওয়ার": "software",
    "ইঞ্জিনিয়ার": "engineer",
    "ইঞ্জিনিয়ার": "engineer",
    "কোডিং": "coding",
    "রিসার্চ": "research",
    "ল্যাব": "lab",
    "লাইব্রেরি": "library",
    "ওয়াইফাই": "wifi",
    "ওয়াইফাই": "wifi",
    "ইন্টারনেট": "internet",
    "গুগল": "google",
    "ফেসবুক": "facebook",
    "ইউটিউব": "youtube",
    "ইনস্টাগ্রাম": "instagram",
    "টিকটক": "tiktok",
    "অ্যাপ": "app",
    "এ্যাপ": "app",
    "ডাটা": "data",
    "ডেটা": "data",
    "সিম": "sim",
    "পাসওয়ার্ড": "password",
    "পাসওয়ার্ড": "password",
    "একাউন্ট": "account",
    "অ্যাকাউন্ট": "account",
    "ওয়েবসাইট": "website",
    "ওয়েবসাইট": "website",
    "লিংক": "link",
    "নোটিফিকেশন": "notification",
    "হোয়াটসঅ্যাপ": "whatsapp",
    "হোয়াটসঅ্যাপ": "whatsapp",
    "মেসেঞ্জার": "messenger",
    "টেলিগ্রাম": "telegram",
    "জুম": "zoom",
    "ইমেইল": "email",
    "চার্জার": "charger",
    "হেডফোন": "headphone",
    "স্পিকার": "speaker",
    "স্ক্রিন": "screen",
    "সেলফি": "selfie",
    "স্ক্রিনশট": "screenshot",
    "ব্যাংক": "bank",
    "লোন": "loan",
    "পেমেন্ট": "payment",
    "ট্রানজেকশন": "transaction",
    "বিকাশ": "bkash",
    "নগদ": "nagad",
    "রকেট": "rocket",
    "রিচার্জ": "recharge",
    "ব্যালেন্স": "balance",
    "ট্রান্সফার": "transfer",
    "ক্যাশ": "cash",
    "ক্রেডিট": "credit",
    "ডেবিট": "debit",
}


def clean_word(word: str) -> str:
    return BN_PUNCT_RE.sub("", word.strip())


def tag_lang(tag: dict) -> str:
    """Collapse borrowed English into EN for ratio/switch calculation."""
    return "EN" if tag.get("lang") in {"EN", "EN_BORROWED"} else tag.get("lang", "")


def count_language_switches(word_tags: list[dict]) -> int:
    langs = [tag_lang(tag) for tag in word_tags if tag_lang(tag) in {"BN", "EN"}]
    return sum(1 for prev, curr in zip(langs, langs[1:]) if prev != curr)


def classify_code_mixing(word_tags: list[dict]) -> dict:
    if not word_tags:
        return {
            "label": "NOISE",
            "cs_confidence": "NONE",
            "bn_ratio": 0.0,
            "en_ratio": 0.0,
            "bn_word_count": 0,
            "en_word_count": 0,
            "borrowed_en_word_count": 0,
            "switch_count": 0,
            "en_words": [],
            "borrowed_en_words": [],
            "is_code_switched": False,
        }

    bn_words = [tag["word"] for tag in word_tags if tag_lang(tag) == "BN"]
    en_words = [tag["word"] for tag in word_tags if tag_lang(tag) == "EN"]
    borrowed_en_words = [
        tag["word"] for tag in word_tags if tag.get("lang") == "EN_BORROWED"
    ]
    bn_word_count = len(bn_words)
    en_word_count = len(en_words)
    total = bn_word_count + en_word_count

    bn_ratio = bn_word_count / total if total else 0.0
    en_ratio = en_word_count / total if total else 0.0
    switch_count = count_language_switches(word_tags)

    label = "BN_ONLY"
    cs_confidence = "NONE"

    if en_ratio >= 0.90 and bn_word_count <= 1:
        label = "EN_ONLY"
    elif (
        total >= 4
        and bn_word_count >= 2
        and en_word_count >= 2
        and switch_count >= 1
        and 0.10 <= en_ratio <= 0.70
    ):
        label = "CS"
        cs_confidence = "HIGH"
    elif (
        total >= 4
        and bn_word_count >= 2
        and en_word_count >= 1
        and switch_count >= 1
        and 0.05 <= en_ratio <= 0.80
    ):
        label = "CS"
        cs_confidence = "MEDIUM"
    elif bn_word_count >= 2 and en_word_count >= 1:
        label = "BN_ONLY"
        cs_confidence = "LOW"
    elif bn_ratio >= 0.95:
        label = "BN_ONLY"
    elif en_ratio >= 0.60:
        label = "EN_ONLY"

    return {
        "label": label,
        "cs_confidence": cs_confidence,
        "bn_ratio": round(bn_ratio, 2),
        "en_ratio": round(en_ratio, 2),
        "bn_word_count": bn_word_count,
        "en_word_count": en_word_count,
        "borrowed_en_word_count": len(borrowed_en_words),
        "switch_count": switch_count,
        "en_words": en_words,
        "borrowed_en_words": borrowed_en_words,
        "is_code_switched": label == "CS",
    }


def retag_clip(clip: dict) -> tuple[dict, int]:
    word_tags = []
    changed = 0

    for tag in clip.get("word_tags", []):
        new_tag = dict(tag)
        word = clean_word(str(new_tag.get("word", "")))
        if word in BORROWED_EN_WORDS and new_tag.get("lang") == "BN":
            new_tag["lang"] = "EN_BORROWED"
            new_tag["borrowed_en"] = True
            new_tag["canonical_en"] = BORROWED_EN_WORDS[word]
            changed += 1
        word_tags.append(new_tag)

    cls = classify_code_mixing(word_tags)
    clip = dict(clip)
    clip["word_tags"] = word_tags
    clip["clip_label"] = cls["label"]
    clip["is_code_switched"] = cls["is_code_switched"]
    clip["bn_ratio"] = cls["bn_ratio"]
    clip["en_ratio"] = cls["en_ratio"]
    clip["bn_word_count"] = cls["bn_word_count"]
    clip["en_word_count"] = cls["en_word_count"]
    clip["borrowed_en_word_count"] = cls["borrowed_en_word_count"]
    clip["borrowed_en_words"] = cls["borrowed_en_words"]
    clip["switch_count"] = cls["switch_count"]
    clip["cs_confidence"] = cls["cs_confidence"]
    return clip, changed


def main() -> None:
    if not TRANSCRIPTS_PATH.exists():
        print(f"[ERROR] {TRANSCRIPTS_PATH} not found. Run 03_auto_transcribe.py first.")
        raise SystemExit(1)

    with open(TRANSCRIPTS_PATH, encoding="utf-8") as f:
        clips = json.load(f)

    if not BACKUP_PATH.exists():
        shutil.copy2(TRANSCRIPTS_PATH, BACKUP_PATH)

    updated_clips = []
    changed_clips = 0
    changed_words = 0
    label_before = Counter(clip.get("clip_label", "UNKNOWN") for clip in clips)

    borrowed_counter: Counter[str] = Counter()
    for clip in clips:
        updated, changed = retag_clip(clip)
        if changed:
            changed_clips += 1
            changed_words += changed
            borrowed_counter.update(updated.get("borrowed_en_words", []))
        updated_clips.append(updated)

    label_after = Counter(clip.get("clip_label", "UNKNOWN") for clip in updated_clips)

    with open(TRANSCRIPTS_PATH, "w", encoding="utf-8") as f:
        json.dump(updated_clips, f, ensure_ascii=False, indent=2)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("Borrowed English retag report\n")
        f.write("=" * 32 + "\n")
        f.write(f"Input clips: {len(clips)}\n")
        f.write(f"Changed clips: {changed_clips}\n")
        f.write(f"Retagged words: {changed_words}\n\n")
        f.write(f"Labels before: {dict(label_before)}\n")
        f.write(f"Labels after : {dict(label_after)}\n\n")
        f.write("Top borrowed English surface forms:\n")
        for word, count in borrowed_counter.most_common(50):
            f.write(f"  {word}: {count}\n")

    print(f"Updated: {TRANSCRIPTS_PATH}")
    print(f"Backup : {BACKUP_PATH}")
    print(f"Report : {REPORT_PATH}")
    print(f"Changed clips: {changed_clips}")
    print(f"Retagged words: {changed_words}")
    print(f"Labels before: {dict(label_before)}")
    print(f"Labels after : {dict(label_after)}")


if __name__ == "__main__":
    main()
