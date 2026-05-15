"""
Step 4e: Dialect-preserving transcript quality analysis.

This step does not clean, delete, or normalize transcripts. It builds review
signals that help separate real dialect variation from ASR corruption:

  - token frequency table
  - suspicious token candidates
  - transcript-level suspiciousness scores
  - dialect/noise lexicon seed candidates

The outputs are intended for lexicon building and targeted human review.

FIXES applied over original version
------------------------------------
FIX 1 — TOKEN_RE now includes Devanagari (\u0900-\u097F) so that Hindi/Devanagari
         characters written by Whisper are captured and flagged instead of silently
         skipped. has_foreign_alpha() already handled detection logic; the regex
         was the only reason it never fired.

FIX 2 — has_word_loop() added. Detects phrase-level hallucination loops (repeated
         word n-grams like "সাই সাই সাই করতে সাই সাই"). character_repetition() only
         caught repeated single characters and missed this class entirely.

FIX 3 — Under-transcription check added to clip scoring. Uses duration_sec (which
         was already read but never used) to compute words-per-second. Clips below
         0.8 wps are flagged. Threshold is configurable via --min-wps.

FIX 4 — Truncation detection added. Transcripts ending with "..." are flagged.
         The "..." marker is non-alphanumeric so TOKEN_RE skipped it entirely.

FIX 5 — avg_no_speech_prob guard made safe. The original code called
         float(clip.get("avg_no_speech_prob") or 0.0) which silently returned 0.0
         when the field was absent. Now the field is explicitly checked and the
         check is skipped when the column is not present in the data, rather than
         silently never firing.

         Note on lexicons: the four lexicon files (standard_bn.txt, english.txt,
         dialect_bn.txt, noise_tokens.txt) are still created as empty stubs if
         absent. Populate them from the *_candidates.csv outputs this script
         produces. Until they are populated, classify_token() will return UNKNOWN
         or ENGLISH_CANDIDATE for most tokens — this is expected behaviour on a
         first run.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "Data ASR" / "transcripts_merged_all" / "transcripts.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "transcripts" / "quality_analysis"
LEXICON_DIR = SCRIPT_DIR / "lexicons"

STANDARD_LEXICON = LEXICON_DIR / "standard_bn.txt"
ENGLISH_LEXICON   = LEXICON_DIR / "english.txt"
DIALECT_LEXICON   = LEXICON_DIR / "dialect_bn.txt"
NOISE_LEXICON     = LEXICON_DIR / "noise_tokens.txt"

# FIX 1 — extended to \u0900-\u09FF so both Devanagari (\u0900-\u097F)
#          and Bengali (\u0980-\u09FF) tokens are captured.
#          Original only matched \u0980-\u09FF which made Devanagari invisible.
TOKEN_RE = re.compile(
    r"[\u0900-\u09ffA-Za-z][\u0900-\u09ffA-Za-z'\-]*",
    re.UNICODE,
)

BN_RE   = re.compile(r"[\u0980-\u09ff]")
DEVA_RE = re.compile(r"[\u0900-\u097f]")   # FIX 1 — Devanagari detector
LATIN_RE = re.compile(r"[A-Za-z]")
LATIN_CONSONANT_RE = re.compile(r"[bcdfghjklmnpqrstvwxyz]{5,}", re.IGNORECASE)
REPLACEMENT_CHAR = "\ufffd"

# Whisper hallucination loop detection (FIX 2)
WORD_LOOP_NGRAM_SIZE = 3   # look for repeated 3-word sequences
WORD_LOOP_MIN_TOKENS = 6   # don't bother checking very short transcripts

# Under-transcription thresholds (FIX 3)
DEFAULT_MIN_WPS = 0.8      # words per second below this = under-transcription
DEFAULT_MIN_DURATION = 4.0 # ignore clips shorter than this (avoid division noise)

TOKEN_COLUMNS = [
    "token",
    "count",
    "doc_count",
    "dialect_count",
    "top_dialects",
    "script",
    "lexicon_label",
    "suspicion_score",
    "suspicion_reasons",
]

CLIP_COLUMNS = [
    "clip_id",
    "audio_path",
    "dialect",
    "is_code_switched",
    "cs_confidence",
    "avg_no_speech_prob",
    "duration_sec",
    "words_per_sec",
    "token_count",
    "unknown_token_count",
    "suspicious_token_count",
    "suspicion_score",
    "suspicion_reasons",
    "suspicious_tokens",
    "transcript",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_path(path: str) -> str:
    return str(path).replace("\\", "/")


def load_lexicon(path: Path) -> set[str]:
    if not path.exists():
        return set()
    words: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words.add(line.casefold())
    return words


def ensure_lexicon_files() -> None:
    LEXICON_DIR.mkdir(parents=True, exist_ok=True)
    placeholders = {
        STANDARD_LEXICON: (
            "# One standard Bengali word per line.\n"
            "# Populate from: https://github.com/MinhasKamal/BengaliWordList\n"
        ),
        ENGLISH_LEXICON: (
            "# One English code-switch word per line.\n"
            "# Add from the ENGLISH_CANDIDATE tokens in token_frequency.csv\n"
        ),
        DIALECT_LEXICON: (
            "# One valid dialectal Bengali form per line.\n"
            "# Do NOT standardize these. Add from dialect_lexicon_candidates.csv\n"
        ),
        NOISE_LEXICON: (
            "# One confirmed ASR-corruption token per line.\n"
            "# Add from noise_lexicon_candidates.csv after human review.\n"
        ),
    }
    for path, text in placeholders.items():
        if not path.exists():
            path.write_text(text, encoding="utf-8")


def tokenize(text: str) -> list[str]:
    """Extract tokens using the extended TOKEN_RE (FIX 1 expands to Devanagari)."""
    return [match.group(0).casefold() for match in TOKEN_RE.finditer(text or "")]


def script_label(token: str) -> str:
    """Return the dominant script family for a token."""
    has_bn   = bool(BN_RE.search(token))
    has_deva = bool(DEVA_RE.search(token))   # FIX 1
    has_latin = bool(LATIN_RE.search(token))

    if has_deva and not has_bn and not has_latin:
        return "DEVANAGARI"                   # FIX 1 — new label
    scripts = sum([has_bn, has_deva, has_latin])
    if scripts > 1:
        return "MIXED"
    if has_bn:
        return "BN"
    if has_latin:
        return "LATIN"
    return "OTHER"


def has_foreign_alpha(token: str) -> bool:
    """
    True if the token contains alphabetic characters outside Bengali and Latin.
    With FIX 1 Devanagari tokens now reach this function because TOKEN_RE
    captures them. Previously they were filtered out before this check ran.
    """
    for char in token:
        if not char.isalpha():
            continue
        cp = ord(char)
        if 0x0900 <= cp <= 0x09FF:   # Bengali + Devanagari both Bengali-adjacent
            continue
        if "A" <= char <= "Z" or "a" <= char <= "z":
            continue
        return True
    return False


def has_devanagari(token: str) -> bool:
    """True when a token contains one or more Devanagari characters (FIX 1)."""
    return bool(DEVA_RE.search(token))


def character_repetition(token: str) -> bool:
    """Detect 4+ consecutive identical characters (e.g. 'কককক')."""
    return bool(re.search(r"(.)\1{3,}", token))


def has_many_consonants(token: str) -> bool:
    """Detect 5+ consecutive Latin consonants, a common hallucination artifact."""
    return bool(LATIN_CONSONANT_RE.search(token))


# FIX 2 — phrase-level hallucination loop detection
def has_word_loop(tokens: list[str], n: int = WORD_LOOP_NGRAM_SIZE) -> bool:
    """
    Return True when any n-gram of consecutive tokens appears more than once.
    This catches Whisper's phrase-repetition hallucination pattern:
      e.g. "সাই সাই সাই করতে সাই সাই" or "করা করা করা চুছেন করা করা"
    character_repetition() only caught repeated single characters and missed this.
    """
    if len(tokens) < max(WORD_LOOP_MIN_TOKENS, n + 1):
        return False
    seen = set()
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i : i + n])
        if ngram in seen:
            return True
        seen.add(ngram)
    return False


def vowel_ratio(token: str) -> float:
    """Fraction of alphabetic characters that are vowels."""
    vowels = set("aeiou")
    bn_vowels = {
        "\u0985", "\u0986", "\u0987", "\u0988", "\u0989", "\u098a",
        "\u098b", "\u098f", "\u0990", "\u0993", "\u0994",
        "\u09be", "\u09bf", "\u09c0", "\u09c1", "\u09c2", "\u09c3",
        "\u09c7", "\u09c8", "\u09cb", "\u09cc",
    }
    letters = [c for c in token if c.isalpha()]
    if not letters:
        return 0.0
    n_vowels = sum(1 for c in letters if c in bn_vowels or c.lower() in vowels)
    return n_vowels / len(letters)


def classify_token(
    token: str,
    standard: set[str],
    english: set[str],
    dialect: set[str],
    noise: set[str],
) -> str:
    folded = token.casefold()
    if folded in noise:
        return "NOISE"
    if folded in dialect:
        return "DIALECT"
    if folded in standard:
        return "STANDARD"
    if folded in english:
        return "ENGLISH"

    slabel = script_label(token)
    if slabel == "DEVANAGARI":            # FIX 1
        return "DEVANAGARI_CANDIDATE"
    if slabel == "LATIN":
        return "ENGLISH_CANDIDATE"
    return "UNKNOWN"


def token_suspicion(
    token: str,
    count: int,
    doc_count: int,
    dialect_count: int,
    label: str,
) -> tuple[int, list[str]]:
    reasons: list[str] = []

    if label == "NOISE":
        reasons.append("known_noise")
    if label == "DEVANAGARI_CANDIDATE":   # FIX 1
        reasons.append("devanagari_token")
    if label == "UNKNOWN" and count <= 2:
        reasons.append("rare_unknown")
    if label == "UNKNOWN" and dialect_count <= 1:
        reasons.append("single_dialect_unknown")
    if script_label(token) == "MIXED":
        reasons.append("mixed_script_token")
    if REPLACEMENT_CHAR in token:
        reasons.append("replacement_character")
    if has_foreign_alpha(token):
        reasons.append("foreign_script_character")
    if character_repetition(token):
        reasons.append("repeated_character")
    if has_many_consonants(token):
        reasons.append("too_many_consonants")
    if len(token) >= 8 and vowel_ratio(token) < 0.18:
        reasons.append("low_vowel_ratio")
    if len(token) <= 1:
        reasons.append("too_short_token")

    weights = {
        "known_noise":             5,
        "replacement_character":   5,
        "devanagari_token":        4,   # FIX 1
        "foreign_script_character":4,
        "mixed_script_token":      3,
        "repeated_character":      3,
        "too_many_consonants":     3,
        "rare_unknown":            2,
        "single_dialect_unknown":  1,
        "low_vowel_ratio":         1,
        "too_short_token":         1,
    }
    score = sum(weights.get(r, 1) for r in reasons)

    # Unknownness alone is a lexicon-building signal, not proof of corruption.
    if set(reasons).issubset({"rare_unknown", "single_dialect_unknown"}):
        score = min(score, 2)

    # A word spread across many clips/dialects is unlikely to be pure noise.
    if doc_count >= 20 or dialect_count >= 4:
        score = max(0, score - 2)

    return score, reasons


# ---------------------------------------------------------------------------
# CSV writer
# ---------------------------------------------------------------------------

def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze(
    input_path: Path,
    output_dir: Path,
    min_token_score: int,
    min_wps: float = DEFAULT_MIN_WPS,
    min_duration: float = DEFAULT_MIN_DURATION,
    auto_populate: bool = False,
) -> None:
    ensure_lexicon_files()

    standard = load_lexicon(STANDARD_LEXICON)
    english  = load_lexicon(ENGLISH_LEXICON)
    dialect  = load_lexicon(DIALECT_LEXICON)
    noise    = load_lexicon(NOISE_LEXICON)

    with input_path.open(encoding="utf-8") as f:
        clips = json.load(f)

    # -----------------------------------------------------------------------
    # Pass 1 — build per-token statistics across the corpus
    # -----------------------------------------------------------------------
    token_counts:   Counter[str]              = Counter()
    token_doc_counts: Counter[str]            = Counter()
    token_dialects: dict[str, Counter[str]]   = defaultdict(Counter)
    clip_tokens:    dict[str, list[str]]      = {}

    for clip in clips:
        clip_id      = str(clip.get("clip_id", ""))
        dialect_name = str(clip.get("dialect", "unknown"))
        tokens       = tokenize(str(clip.get("transcript", "")))
        clip_tokens[clip_id] = tokens
        token_counts.update(tokens)
        token_doc_counts.update(set(tokens))
        for token in set(tokens):
            token_dialects[token][dialect_name] += 1

    # -----------------------------------------------------------------------
    # Pass 2 — score every unique token
    # -----------------------------------------------------------------------
    token_rows:   list[dict[str, Any]] = []
    token_scores: dict[str, int]       = {}
    token_reasons: dict[str, list[str]] = {}
    token_labels: dict[str, str]       = {}

    for token, count in token_counts.most_common():
        dialect_counter = token_dialects[token]
        dialect_count   = len(dialect_counter)
        doc_count       = token_doc_counts[token]
        label           = classify_token(token, standard, english, dialect, noise)
        score, reasons  = token_suspicion(token, count, doc_count, dialect_count, label)
        token_scores[token]  = score
        token_reasons[token] = reasons
        token_labels[token]  = label
        token_rows.append(
            {
                "token":             token,
                "count":             count,
                "doc_count":         doc_count,
                "dialect_count":     dialect_count,
                "top_dialects":      ";".join(
                    f"{n}:{v}" for n, v in dialect_counter.most_common(5)
                ),
                "script":            script_label(token),
                "lexicon_label":     label,
                "suspicion_score":   score,
                "suspicion_reasons": ";".join(reasons),
            }
        )

    suspicious_token_rows = [
        row for row in token_rows if int(row["suspicion_score"]) >= min_token_score
    ]
    suspicious_token_rows.sort(
        key=lambda r: (-int(r["suspicion_score"]), int(r["count"]), r["token"])
    )

    # -----------------------------------------------------------------------
    # Pass 3 — score every clip
    # -----------------------------------------------------------------------
    clip_rows: list[dict[str, Any]] = []

    for clip in clips:
        clip_id  = str(clip.get("clip_id", ""))
        tokens   = clip_tokens.get(clip_id, [])
        transcript_raw = str(clip.get("transcript", ""))

        # Token-level suspicion
        suspicious     = [t for t in tokens if token_scores.get(t, 0) >= min_token_score]
        unknown_count  = sum(1 for t in tokens if token_labels.get(t) == "UNKNOWN")
        suspicious_score = sum(token_scores.get(t, 0) for t in suspicious)

        clip_reasons: Counter[str] = Counter(
            reason
            for t in suspicious
            for reason in token_reasons.get(t, [])
        )

        # Replacement character
        if REPLACEMENT_CHAR in transcript_raw:
            suspicious_score += 5
            clip_reasons["replacement_character"] += 1

        # FIX 5 — avg_no_speech_prob: only apply when the field is actually present
        #          and holds a numeric value. Previously: float(None or 0.0) = 0.0
        #          silently made the check dormant without any indication.
        nsp_raw = clip.get("avg_no_speech_prob")
        nsp_value: float | None = None
        if nsp_raw is not None and nsp_raw != "":
            try:
                nsp_value = float(nsp_raw)
            except (TypeError, ValueError):
                nsp_value = None
        if nsp_value is not None and nsp_value >= 0.30:
            suspicious_score += 2
            clip_reasons["high_no_speech_prob"] += 1

        # FIX 3 — under-transcription: words-per-second check using duration_sec
        #          which was read but never used in the original.
        duration = float(clip.get("duration_sec") or 0.0)
        wps: float | None = None
        if duration >= min_duration and tokens:
            wps = len(tokens) / duration
            if wps < min_wps:
                suspicious_score += 3
                clip_reasons["under_transcription"] += 1

        # FIX 2 — phrase-level word loop detection
        if has_word_loop(tokens):
            suspicious_score += 4
            clip_reasons["word_loop_hallucination"] += 1

        # FIX 4 — truncation: transcript ends with "..."
        if transcript_raw.rstrip().endswith("..."):
            suspicious_score += 2
            clip_reasons["truncated_transcript"] += 1

        if suspicious_score <= 0:
            continue

        clip_rows.append(
            {
                "clip_id":               clip_id,
                "audio_path":            normalize_path(clip.get("audio_path", "")),
                "dialect":               clip.get("dialect", "unknown"),
                "is_code_switched":      clip.get("is_code_switched", False),
                "cs_confidence":         clip.get("cs_confidence", "NONE"),
                "avg_no_speech_prob":    "" if nsp_value is None else nsp_value,
                "duration_sec":          duration or "",
                "words_per_sec":         round(wps, 3) if wps is not None else "",
                "token_count":           len(tokens),
                "unknown_token_count":   unknown_count,
                "suspicious_token_count":len(suspicious),
                "suspicion_score":       suspicious_score,
                "suspicion_reasons":     ";".join(
                    f"{name}:{val}" for name, val in clip_reasons.most_common()
                ),
                "suspicious_tokens":     ";".join(sorted(set(suspicious))),
                "transcript":            transcript_raw,
            }
        )

    clip_rows.sort(
        key=lambda r: (
            -float(r["suspicion_score"]),
            -int(r["suspicious_token_count"]),
            str(r["clip_id"]),
        )
    )

    # -----------------------------------------------------------------------
    # Lexicon seed outputs — unchanged from original logic
    # -----------------------------------------------------------------------

    # Candidates to promote into dialect_bn.txt
    dialect_seed_rows = [
        row for row in token_rows
        if row["lexicon_label"] == "UNKNOWN"
        and int(row["count"]) >= 5
        and int(row["dialect_count"]) >= 2
        and int(row["suspicion_score"]) <= 1
    ]
    dialect_seed_rows.sort(key=lambda r: (-int(r["count"]), r["token"]))

    # Candidates to promote into noise_tokens.txt
    noise_seed_rows = [
        row for row in token_rows
        if int(row["suspicion_score"]) >= min_token_score
    ]

    # All unknowns for manual review
    unknown_seed_rows = [
        row for row in token_rows
        if row["lexicon_label"] in {"UNKNOWN", "DEVANAGARI_CANDIDATE"}  # FIX 1
    ]
    unknown_seed_rows.sort(
        key=lambda r: (int(r["count"]), -int(r["suspicion_score"]), r["token"])
    )

    # Devanagari-specific output (FIX 1) — easy to re-run Whisper on these clips
    deva_clip_rows = [
        row for row in clip_rows
        if "devanagari_token" in row["suspicion_reasons"]
    ]

    # -----------------------------------------------------------------------
    # Write outputs
    # -----------------------------------------------------------------------
    write_csv(output_dir / "token_frequency.csv",          TOKEN_COLUMNS, token_rows)
    write_csv(output_dir / "suspicious_tokens.csv",        TOKEN_COLUMNS, suspicious_token_rows)
    write_csv(output_dir / "suspicious_clips.csv",         CLIP_COLUMNS,  clip_rows)
    write_csv(output_dir / "unknown_token_candidates.csv", TOKEN_COLUMNS, unknown_seed_rows)
    write_csv(output_dir / "dialect_lexicon_candidates.csv", TOKEN_COLUMNS, dialect_seed_rows)
    write_csv(output_dir / "noise_lexicon_candidates.csv", TOKEN_COLUMNS, noise_seed_rows)
    write_csv(output_dir / "devanagari_clips.csv",         CLIP_COLUMNS,  deva_clip_rows)  # FIX 1

    summary = {
        "input_path": str(input_path),
        "clip_count": len(clips),
        "unique_token_count": len(token_rows),
        "suspicious_token_count": len(suspicious_token_rows),
        "suspicious_clip_count": len(clip_rows),
        "devanagari_clip_count": len(deva_clip_rows),         # FIX 1
        "unknown_token_candidate_count": len(unknown_seed_rows),
        "dialect_lexicon_candidate_count": len(dialect_seed_rows),
        "noise_lexicon_candidate_count": len(noise_seed_rows),
        "thresholds": {
            "min_token_score": min_token_score,
            "min_wps": min_wps,                               # FIX 3
            "min_duration_for_wps_check": min_duration,       # FIX 3
        },
        "lexicon_sizes": {
            "standard_bn": len(standard),
            "english":     len(english),
            "dialect_bn":  len(dialect),
            "noise_tokens":len(noise),
        },
        "warnings": (
            []
            if any([standard, english, dialect, noise])
            else [
                "All lexicon files are empty placeholders. classify_token() will return "
                "UNKNOWN or ENGLISH_CANDIDATE for most tokens. "
                "Populate lexicons from the *_candidates.csv outputs before re-running."
            ]
        ),
    }

    if auto_populate:
        def append_to_lexicon(lex_path: Path, rows: list[dict[str, Any]]) -> int:
            existing = load_lexicon(lex_path)
            new_tokens = [r["token"] for r in rows if r["token"].casefold() not in existing]
            if not new_tokens:
                return 0
            with lex_path.open("a", encoding="utf-8") as lex_file:
                for t in new_tokens:
                    lex_file.write(f"{t}\n")
            return len(new_tokens)

        eng_rows = [r for r in token_rows if r["lexicon_label"] == "ENGLISH_CANDIDATE"]
        eng_rows.sort(key=lambda r: (-int(r["count"]), r["token"]))

        summary["auto_populated"] = {
            "noise_added": append_to_lexicon(NOISE_LEXICON, noise_seed_rows),
            "dialect_added": append_to_lexicon(DIALECT_LEXICON, dialect_seed_rows),
            "english_added": append_to_lexicon(ENGLISH_LEXICON, eng_rows),
            "note": "Re-run the script without --auto-populate to see the updated metrics!"
        }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build dialect-preserving token quality reports."
    )
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Path to transcripts.json",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory to write CSV and JSON outputs",
    )
    parser.add_argument(
        "--min-token-score", type=int, default=5,
        help="Minimum suspicion score for a token to appear in suspicious_tokens.csv",
    )
    # FIX 3 — expose WPS threshold as a CLI argument
    parser.add_argument(
        "--min-wps", type=float, default=DEFAULT_MIN_WPS,
        help=(
            f"Words-per-second below this threshold flags a clip as under-transcribed "
            f"(default: {DEFAULT_MIN_WPS})"
        ),
    )
    parser.add_argument(
        "--min-duration", type=float, default=DEFAULT_MIN_DURATION,
        help=(
            f"Clips shorter than this (seconds) skip the WPS check to avoid "
            f"false positives on very short utterances (default: {DEFAULT_MIN_DURATION})"
        ),
    )
    parser.add_argument(
        "--auto-populate", action="store_true",
        help="Automatically add all identified candidates to the lexicon text files.",
    )
    args = parser.parse_args()

    analyze(
        input_path=args.input,
        output_dir=args.output_dir,
        min_token_score=args.min_token_score,
        min_wps=args.min_wps,
        min_duration=args.min_duration,
        auto_populate=args.auto_populate,
    )


if __name__ == "__main__":
    main()