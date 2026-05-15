"""
Text normalization for Bengali dialect + English code-switching ASR.

Two levels of normalization are provided:

  1. normalize_transcript(text)
       Full cleaning pipeline applied BEFORE training and evaluation.
       Strips noise (URLs, emojis, hashtags, punctuation) and normalizes
       digits, Unicode form, and whitespace. The model trains on this.

  2. normalize_pronunciation_variants(text)
       Applied ON TOP of level-1 normalization during evaluation only.
       Maps phonetic/script variants of borrowed English words to a single
       canonical form so that "miting" and "meeting" count as correct.

Both functions are idempotent — safe to call multiple times.
"""

from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

SCRIPT_DIR    = Path(__file__).resolve().parent
VARIANTS_PATH = SCRIPT_DIR / "pronunciation_variants.json"

# ── Punctuation to strip ───────────────────────────────────────────────────────
# Bengali danda (।), double danda (॥), common ASCII punctuation
PUNCT_RE = re.compile(
    r"[.,!?;:\"'()\[\]{}<>|/\\`~@#$%^&*_+=\-\u0964\u0965]+"
)

# ── Digit maps ─────────────────────────────────────────────────────────────────
# Bengali digits → ASCII digits
_BN_DIGIT = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# ── Noise patterns ─────────────────────────────────────────────────────────────
_URL_RE      = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HASHTAG_RE  = re.compile(r"#\w+")
_MENTION_RE  = re.compile(r"@\w+")
_EMOJI_RE    = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F1E0-\U0001F1FF"   # flags
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001f926-\U0001f937"
    "\U00010000-\U0010ffff"
    "\u2640-\u2642"
    "\u2600-\u2B55"
    "\u200d"                  # zero-width joiner (keep separate from ZWNJ)
    "\u23cf\u23e9\u231a\ufe0f\u3030"
    "]+",
    flags=re.UNICODE,
)

# Invisible / control characters that corrupt Bengali rendering
_INVISIBLE_RE = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\u202a-\u202e\ufeff\u00ad]+"
)

# Repeated character noise: হাহাহাহা → হাহা, lollll → lol (max 2 repeats)
_REPEAT_CHAR_RE = re.compile(r"(.)\1{2,}")

# Multiple whitespace
_WHITESPACE_RE = re.compile(r"\s+")

# Common Bengali suffix list used when matching borrowed words + suffix
COMMON_SUFFIXES = (
    "টা", "টি", "খানা", "খান",
    "গুলো", "গুলা", "গুলোর", "গুলার",
    "দের", "টার", "টির",
    "এ", "তে", "য়ে", "য়", "র", "এর", "কে",
)


# ── Level 1: Full transcript cleaning ─────────────────────────────────────────

def normalize_transcript(text: str) -> str:
    """
    Full cleaning pipeline for transcripts.

    Steps (in order):
      1. Unicode NFC normalization — ensures consistent codepoint representation
      2. Remove invisible/zero-width characters (ZWNJ, BOM, soft-hyphen …)
      3. Remove URLs, hashtags, @mentions
      4. Remove emojis
      5. Bengali digit → ASCII digit
      6. Collapse repeated characters (noise like "হাহাহাহা")
      7. Strip punctuation
      8. Collapse whitespace
      9. Strip leading/trailing whitespace
    """
    if not text or not text.strip():
        return ""

    # 1. NFC — critical for Bengali: some chars have multiple Unicode forms
    text = unicodedata.normalize("NFC", text)

    # 2. Remove invisible characters
    text = _INVISIBLE_RE.sub(" ", text)

    # 3. Remove URLs, hashtags, mentions
    text = _URL_RE.sub(" ", text)
    text = _HASHTAG_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)

    # 4. Remove emojis
    text = _EMOJI_RE.sub(" ", text)

    # 5. Bengali digits → ASCII
    text = text.translate(_BN_DIGIT)

    # 6. Collapse excessive character repetition
    #    হাহাহাহা → হাহা, yessss → yes (keep max 2 repeats)
    text = _REPEAT_CHAR_RE.sub(r"\1\1", text)

    # 7. Strip punctuation (preserves Bengali script and Latin letters)
    text = PUNCT_RE.sub(" ", text)

    # 8. Collapse whitespace
    text = _WHITESPACE_RE.sub(" ", text)

    # 9. Strip
    return text.strip()


# ── Level 2: Pronunciation variant normalization ───────────────────────────────

@lru_cache(maxsize=1)
def _load_variant_map() -> tuple[dict[str, str], tuple[str, ...]]:
    """
    Load pronunciation_variants.json and build an inverted lookup.

    Returns:
      variant_map   — {surface_form_lower: canonical_lower}
      sorted_keys   — variant strings sorted longest-first for greedy matching
    """
    with open(VARIANTS_PATH, encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError as e:
            print(f"\n[ERROR] Invalid JSON syntax in {VARIANTS_PATH.name}")
            print(f"        Please fix the missing comma or quote at line {e.lineno}, column {e.colno}.")
            raise SystemExit(1)

    variant_map: dict[str, str] = {}
    for canonical, variants in raw.items():
        if canonical.startswith("_"):          # skip comment keys
            continue
        canon_lower = canonical.casefold()
        variant_map[canon_lower] = canon_lower
        for v in variants:
            variant_map[v.casefold()] = canon_lower

    sorted_keys = tuple(sorted(variant_map, key=len, reverse=True))
    return variant_map, sorted_keys


def _normalize_token(token: str) -> str:
    """
    Normalize one whitespace-separated token.

    Tries exact match first. Then tries stripping a known Bengali suffix from
    borrowed-word + suffix compounds (e.g. "meetingটা" → "meeting").
    Returns the canonical form if found, otherwise the original token.
    """
    if not token:
        return token

    variant_map, _ = _load_variant_map()
    folded = token.casefold()

    # Exact match
    if folded in variant_map:
        return variant_map[folded]

    # Suffix stripping: check if token = known_variant + Bengali_suffix
    for suffix in COMMON_SUFFIXES:
        if token.endswith(suffix):
            stem = token[: -len(suffix)]
            stem_folded = stem.casefold()
            if stem_folded in variant_map:
                return variant_map[stem_folded]

    return token


def normalize_pronunciation_variants(text: str) -> str:
    """
    Map phonetic/script variants of borrowed words to canonical English form.

    Examples:
      "miting korchi"         → "meeting korchi"
      "ofisটা boro"          → "office boro"
      "মিটিং ache"            → "meeting ache"
      "amar iskul balo"       → "amar school balo"
      "seriously bolchi"      → "seriously bolchi"   (already canonical)

    Call this AFTER normalize_transcript().
    Used during evaluation: both reference and hypothesis are normalized
    before WER computation so that pronunciation variants don't inflate WER.
    """
    if not text or not text.strip():
        return text.strip()

    return " ".join(_normalize_token(tok) for tok in text.split())


# ── Combined convenience function ─────────────────────────────────────────────

def full_normalize(text: str) -> str:
    """
    Apply both levels in sequence.
    Use this when preparing references for evaluation.
    """
    return normalize_pronunciation_variants(normalize_transcript(text))


# ── Quick self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        # (input, expected after full_normalize)
        ("আমি miting এ যাচ্ছি।", "আমি meeting এ যাচ্ছি"),
        ("ofisটা অনেক বড়", "office অনেক বড়"),
        ("হাহাহাহাহা 😂😂 #funny @user", "হাহা"),
        ("আমার iskul টা ভালো", "আমার school টা ভালো"),
        ("https://youtube.com এটা দেখো", "এটা দেখো"),
        ("seriously বলছি এটা amazing", "seriously বলছি এটা amazing"),
        ("আমি phone টা কিনছি", "আমি phone টা কিনছি"),
        ("০১৭১২৩৪৫৬৭৮", "01712345678"),
        ("bkash এ পাঠাও", "bkash এ পাঠাও"),
        ("ইউটিউব চ্যানেলটা subscribe করো", "youtube channel টা subscribe করো"),
        ("  extra   spaces  here  ", "extra spaces here"),
        ("মিটিং করলাম", "meeting করলাম"),
    ]

    print("=" * 60)
    print("Text normalization self-test")
    print("=" * 60)
    passed = 0
    for inp, expected in test_cases:
        result = full_normalize(inp)
        status = "PASS" if result == expected else "DIFF"
        if status == "PASS":
            passed += 1
        print(f"  [{status}]")
        print(f"    IN : {inp}")
        print(f"    OUT: {result}")
        if status == "DIFF":
            print(f"    EXP: {expected}")
    print(f"\n{passed}/{len(test_cases)} passed")
