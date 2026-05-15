"""
Helpers for Bengali-English code-mixing detection.

The goal is to keep clips where Bengali or a Bengali dialect is the base
language and English appears inside the same utterance with enough evidence
to be meaningful, not just a stray token.
"""

from __future__ import annotations


def detect_word_tags(result: dict) -> list:
    """Extract BN/EN word tags from a Whisper result."""
    tags = []
    for segment in result.get("segments", []):
        for word_info in segment.get("words", []):
            word = word_info["word"].strip()
            has_bengali = any("\u0980" <= char <= "\u09FF" for char in word)
            if has_bengali or not word.isascii():
                tags.append(
                    {
                        "word": word,
                        "lang": "BN",
                        "start": word_info.get("start"),
                        "end": word_info.get("end"),
                    }
                )
            elif word.isalpha() and len(word) > 1:
                tags.append(
                    {
                        "word": word,
                        "lang": "EN",
                        "start": word_info.get("start"),
                        "end": word_info.get("end"),
                    }
                )
    return tags


def count_language_switches(word_tags: list) -> int:
    """Count BN<->EN switches across the tagged word sequence."""
    langs = [tag["lang"] for tag in word_tags]
    return sum(1 for prev, curr in zip(langs, langs[1:]) if prev != curr)


def classify_code_mixing(word_tags: list) -> dict:
    """
    Classify code-mixing with stricter confidence thresholds.

    HIGH:
      Enough Bengali and English words, at least one language switch,
      and English ratio stays in a realistic code-mixing range.

    MEDIUM:
      Mixed clip, but lighter evidence. Keep for review / optional use.

    LOW:
      Bengali clip with only a weak English trace. Do not treat as strong CS.
    """
    if not word_tags:
        return {
            "label": "NOISE",
            "cs_confidence": "NONE",
            "bn_ratio": 0.0,
            "en_ratio": 0.0,
            "bn_word_count": 0,
            "en_word_count": 0,
            "switch_count": 0,
            "en_words": [],
            "is_code_switched": False,
        }

    bn_words = [tag["word"] for tag in word_tags if tag["lang"] == "BN"]
    en_words = [tag["word"] for tag in word_tags if tag["lang"] == "EN"]
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
        "switch_count": switch_count,
        "en_words": en_words,
        "is_code_switched": label == "CS",
    }
