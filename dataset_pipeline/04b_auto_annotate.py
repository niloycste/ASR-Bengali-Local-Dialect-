"""
Step 4b: Automated annotation using Whisper transcripts as silver labels.

This is a fast alternative to full manual review:
  - HIGH confidence clips are accepted directly
  - MEDIUM confidence clips are accepted but flagged for spot checks
  - obvious low-confidence cases are rejected

Outputs:
  transcripts/transcripts_reviewed.json
  transcripts/auto_annotation_report.txt
"""

from __future__ import annotations

import json
import os
import argparse
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

MERGED_TRANSCRIPTS_PATH = SCRIPT_DIR / "Data ASR" / "transcripts_merged_all" / "transcripts.json"
if MERGED_TRANSCRIPTS_PATH.exists():
    TRANSCRIPTS_PATH = str(MERGED_TRANSCRIPTS_PATH)
else:
    TRANSCRIPTS_PATH = str(SCRIPT_DIR / "transcripts" / "transcripts.json")

REVIEWED_PATH = str(SCRIPT_DIR / "transcripts" / "transcripts_reviewed.json")
REPORT_PATH = str(SCRIPT_DIR / "transcripts" / "auto_annotation_report.txt")

HIGH_CONF_THRESH = 0.30
MED_CONF_THRESH = 0.59
MIN_BN_CHARS = 3
CS_EN_RATIO_MIN = 0.05
CS_EN_RATIO_MAX = 0.65


def avg_no_speech_prob(clip: dict) -> float | None:
    """Return stored average no_speech_prob when available."""
    value = clip.get("avg_no_speech_prob")
    return float(value) if value is not None else None


def transcript_quality(transcript: str) -> str:
    """Estimate quality from Bengali-script coverage."""
    bn_chars = sum(1 for c in transcript if "\u0980" <= c <= "\u09FF")
    return "HIGH" if bn_chars >= MIN_BN_CHARS else "LOW"


def cs_quality_ok(clip: dict) -> bool:
    """Reject CS clips whose English ratio looks unrealistic."""
    if not clip.get("is_code_switched"):
        return True
    en_ratio = clip.get("en_ratio", 0.0)
    return CS_EN_RATIO_MIN <= en_ratio <= CS_EN_RATIO_MAX


def determine_annotation_tier(clip: dict) -> tuple[str, str | None]:
    """
    Decide whether an auto-transcript is HIGH or MEDIUM confidence.

    Missing confidence metadata is handled conservatively as MEDIUM.
    """
    nsp = avg_no_speech_prob(clip)
    if nsp is None:
        return "MEDIUM", None
    if nsp < HIGH_CONF_THRESH:
        return "HIGH", None
    if nsp < MED_CONF_THRESH:
        return "MEDIUM", None
    return "", "high_no_speech_prob"


def build_reviewed_clip(clip: dict, transcript: str, tier: str) -> dict:
    """Project the transcript entry into the reviewed schema."""
    return {
        "clip_id": clip["clip_id"],
        "audio_path": clip["audio_path"],
        "duration_sec": clip["duration_sec"],
        "transcript": transcript,
        "human_transcript": transcript,
        "word_tags": clip.get("word_tags", []),
        "clip_label": clip.get("clip_label", "BN_ONLY"),
        "is_code_switched": clip.get("is_code_switched", False),
        "bn_ratio": clip.get("bn_ratio", 0.0),
        "en_ratio": clip.get("en_ratio", 0.0),
        "en_word_count": clip.get("en_word_count", 0),
        "bn_word_count": clip.get("bn_word_count", 0),
        "switch_count": clip.get("switch_count", 0),
        "cs_confidence": clip.get("cs_confidence", "NONE"),
        "avg_no_speech_prob": clip.get("avg_no_speech_prob"),
        "dialect": clip.get("dialect", "unknown"),
        "domain": clip.get("domain", "General"),
        "source_id": clip.get("source_id"),
        "source_path": clip.get("source_path"),
        "source_bucket": clip.get("source_bucket"),
        "source_root": clip.get("source_root"),
        "needs_human_review": tier == "MEDIUM",
        "annotation_tier": tier,
    }


def auto_annotate(
    transcripts_path: str = TRANSCRIPTS_PATH,
    reviewed_path: str = REVIEWED_PATH,
    report_path: str = REPORT_PATH,
    high_thresh: float = HIGH_CONF_THRESH,
    med_thresh: float = MED_CONF_THRESH,
    min_bn_chars: int = MIN_BN_CHARS,
):
    if not os.path.exists(transcripts_path):
        print(f"[ERROR] {transcripts_path} not found.")
        print("        Run 03_auto_transcribe.py first.")
        raise SystemExit(1)

    with open(transcripts_path, encoding="utf-8") as f:
        clips = json.load(f)

    print(f"Loaded {len(clips)} clips from {transcripts_path}")

    accepted_high: list[dict] = []
    accepted_medium: list[dict] = []
    rejected: list[tuple[dict, str]] = []

    for clip in clips:
        transcript = clip.get("transcript", "").strip()

        if not transcript:
            rejected.append((clip, "empty_transcript"))
            continue

        bn_chars = sum(1 for c in transcript if "\u0980" <= c <= "\u09FF")
        
        if bn_chars < min_bn_chars:
            rejected.append((clip, "too_short"))
            continue

        if not cs_quality_ok(clip):
            rejected.append((clip, "cs_en_ratio_out_of_range"))
            continue

        # Inline determine_annotation_tier logic using the new variables
        nsp = avg_no_speech_prob(clip)
        if nsp is None:
            tier, rejection_reason = "MEDIUM", None
        elif nsp < high_thresh:
            tier, rejection_reason = "HIGH", None
        elif nsp < med_thresh:
            tier, rejection_reason = "MEDIUM", None
        else:
            tier, rejection_reason = "", "high_no_speech_prob"
            
        if rejection_reason:
            rejected.append((clip, rejection_reason))
            continue

        reviewed_clip = build_reviewed_clip(clip, transcript, tier)
        if tier == "HIGH":
            accepted_high.append(reviewed_clip)
        else:
            accepted_medium.append(reviewed_clip)

    all_accepted = accepted_high + accepted_medium

    os.makedirs(Path(reviewed_path).parent, exist_ok=True)
    with open(reviewed_path, "w", encoding="utf-8") as f:
        json.dump(all_accepted, f, ensure_ascii=False, indent=2)

    cs_count = sum(1 for c in all_accepted if c["is_code_switched"])
    bn_count = sum(1 for c in all_accepted if not c["is_code_switched"])
    total_hrs = sum(c["duration_sec"] for c in all_accepted) / 3600

    dialect_counts = Counter(c["dialect"] for c in all_accepted)
    cs_by_dialect = Counter(c["dialect"] for c in all_accepted if c["is_code_switched"])
    cs_conf_counts = Counter(
        c["cs_confidence"] for c in all_accepted if c["is_code_switched"]
    )
    rej_reasons = Counter(reason for _, reason in rejected)

    lines = [
        "=" * 60,
        "AUTO-ANNOTATION REPORT",
        "=" * 60,
        f"Input clips:              {len(clips)}",
        f"Accepted (HIGH conf):     {len(accepted_high)}",
        f"Accepted (MEDIUM conf):   {len(accepted_medium)}",
        f"Rejected:                 {len(rejected)}",
        f"Total accepted:           {len(all_accepted)}",
        f"Total duration:           {total_hrs:.2f} hours",
        "",
        "LABEL BREAKDOWN",
        f"  Code-switched (CS):     {cs_count}  ({cs_count/max(len(all_accepted),1)*100:.1f}%)",
        f"  Bengali-only (BN_ONLY): {bn_count}  ({bn_count/max(len(all_accepted),1)*100:.1f}%)",
        "",
        "CS CONFIDENCE (among CS clips)",
    ]
    for conf, cnt in sorted(cs_conf_counts.items()):
        lines.append(f"  {conf:<8}: {cnt}")

    lines += [
        "",
        "PER-DIALECT BREAKDOWN",
    ]
    for dialect, total in sorted(dialect_counts.items()):
        cs = cs_by_dialect.get(dialect, 0)
        hrs = sum(c["duration_sec"] for c in all_accepted if c["dialect"] == dialect) / 3600
        lines.append(
            f"  {dialect:<18}: {total:>5} clips  {hrs:>6.2f} hrs  "
            f"CS={cs} ({cs/max(total,1)*100:.0f}%)"
        )

    lines += [
        "",
        "REJECTION REASONS",
    ]
    for reason, cnt in sorted(rej_reasons.items()):
        lines.append(f"  {reason:<30}: {cnt}")

    lines += [
        "",
        "FOR YOUR PAPER",
        "  Transcripts are silver-standard labels produced by Whisper large-v3.",
        "  Validate with a manually reviewed sample before claiming final quality.",
        "=" * 60,
    ]

    report = "\n".join(lines)
    print("\n" + report)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")

    print(f"\nReviewed JSON : {reviewed_path}")
    print(f"Report saved  : {report_path}")
    print("Next step     : python 05_build_dataset.py")
    return all_accepted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-accept usable Whisper transcripts as silver labels."
    )
    parser.add_argument(
        "--input",
        default=TRANSCRIPTS_PATH,
        help=(
            "Input transcripts JSON. Defaults to Data ASR/transcripts_merged_all/"
            "transcripts.json when it exists, otherwise transcripts/transcripts.json."
        ),
    )
    parser.add_argument(
        "--output",
        default=REVIEWED_PATH,
        help="Output reviewed JSON for 05_build_dataset.py.",
    )
    parser.add_argument(
        "--report",
        default=REPORT_PATH,
        help="Output text report path.",
    )
    parser.add_argument(
        "--high-thresh",
        type=float,
        default=HIGH_CONF_THRESH,
        help="Threshold for HIGH confidence auto-acceptance (default: 0.30).",
    )
    parser.add_argument(
        "--med-thresh",
        type=float,
        default=MED_CONF_THRESH,
        help="Threshold for MEDIUM confidence (flagged for review) (default: 0.59).",
    )
    parser.add_argument(
        "--min-bn-chars",
        type=int,
        default=MIN_BN_CHARS,
        help="Minimum number of Bengali characters required (default: 3).",
    )
    args = parser.parse_args()

    auto_annotate(
        transcripts_path=args.input,
        reviewed_path=args.output,
        report_path=args.report,
        high_thresh=args.high_thresh,
        med_thresh=args.med_thresh,
        min_bn_chars=args.min_bn_chars,
    )
