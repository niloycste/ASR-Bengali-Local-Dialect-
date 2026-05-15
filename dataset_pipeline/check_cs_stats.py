"""
Quick stats checker — run this after any of these steps:
  - After 03_auto_transcribe.py   → checks transcripts.json
  - After 04b_auto_annotate.py    → checks transcripts_reviewed.json
  - After 05_build_dataset.py     → checks final_dataset manifests

Usage:
    python check_cs_stats.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def check_json(path: Path, label: str):
    if not path.exists():
        print(f"  [NOT FOUND] {path.name}")
        return

    with open(path, encoding="utf-8") as f:
        clips = json.load(f)

    total   = len(clips)
    cs      = sum(1 for c in clips if c.get("is_code_switched"))
    bn_only = total - cs
    hours   = sum(c.get("duration_sec", 0) for c in clips) / 3600

    cs_pct  = cs / total * 100 if total else 0

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"  File   : {path.name}")
    print(f"{'='*55}")
    print(f"  Total clips      : {total:>7,}")
    print(f"  CS clips         : {cs:>7,}  ({cs_pct:.1f}%)")
    print(f"  BN_ONLY clips    : {bn_only:>7,}  ({100-cs_pct:.1f}%)")
    print(f"  Total hours      : {hours:>7.2f}h")

    # ── Augmentation decision helper ──────────────────────────────────────
    print(f"\n  ── Augmentation Decision ──")
    if cs >= 10_000:
        print(f"  ✓ CS clips = {cs:,} — SUFFICIENT. Skip augmentation.")
    elif cs >= 5_000:
        print(f"  ~ CS clips = {cs:,} — BORDERLINE. Augmentation optional.")
        print(f"    Run: python augment_data.py")
    elif cs >= 3_000:
        print(f"  ! CS clips = {cs:,} — LOW. Run audio augmentation.")
        print(f"    Run: python augment_data.py --factor 2")
    else:
        print(f"  !! CS clips = {cs:,} — VERY LOW. Run augmentation + synthetic.")
        print(f"    Run: python augment_data.py --factor 2")
        print(f"    Run: python generate_synthetic_cs.py --tts --count 2000")

    # ── Per-dialect breakdown ─────────────────────────────────────────────
    print(f"\n  ── Per-dialect CS breakdown ──")
    dialect_total = Counter()
    dialect_cs    = Counter()
    for c in clips:
        d = c.get("dialect", "unknown")
        dialect_total[d] += 1
        if c.get("is_code_switched"):
            dialect_cs[d] += 1

    print(f"  {'Dialect':<18} {'Total':>7} {'CS':>7} {'CS%':>6}")
    print(f"  {'-'*42}")
    for dialect in sorted(dialect_total, key=lambda d: -dialect_total[d]):
        t   = dialect_total[dialect]
        c   = dialect_cs[dialect]
        pct = c / t * 100 if t else 0
        flag = " ← low CS" if pct < 15 else ""
        print(f"  {dialect:<18} {t:>7,} {c:>7,} {pct:>5.1f}%{flag}")


def check_manifests():
    """Check final_dataset splits (after 05_build_dataset.py)."""
    import pandas as pd

    splits_dir = SCRIPT_DIR / "final_dataset"
    if not splits_dir.exists():
        print(f"\n  [NOT FOUND] final_dataset/ — run 05_build_dataset.py first")
        return

    print(f"\n{'='*55}")
    print(f"  FINAL DATASET SPLITS (after 05_build_dataset.py)")
    print(f"{'='*55}")

    grand_total = grand_cs = grand_hrs = 0

    for split in ["train", "dev", "test"]:
        csv_path = splits_dir / split / "manifest.csv"
        if not csv_path.exists():
            print(f"  [{split}] NOT FOUND")
            continue

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  [{split}] Error reading: {e}")
            continue

        total = len(df)
        cs    = df["is_code_switched"].sum() if "is_code_switched" in df.columns else 0
        hrs   = df["duration_sec"].sum() / 3600 if "duration_sec" in df.columns else 0
        pct   = cs / total * 100 if total else 0

        print(f"\n  {split.upper()}:")
        print(f"    Clips   : {total:,}")
        print(f"    CS      : {cs:,}  ({pct:.1f}%)")
        print(f"    Hours   : {hrs:.2f}h")

        grand_total += total
        grand_cs    += cs
        grand_hrs   += hrs

    if grand_total:
        print(f"\n  TOTAL  : {grand_total:,} clips | {grand_cs:,} CS ({grand_cs/grand_total*100:.1f}%) | {grand_hrs:.2f}h")


def main():
    print("\nBanglaMix CS Statistics Checker")

    # Step 3 output
    check_json(
        SCRIPT_DIR / "transcripts" / "transcripts.json",
        "AFTER STEP 3 — Auto-transcribe (Whisper raw output)"
    )

    # Step 4b output
    check_json(
        SCRIPT_DIR / "transcripts" / "transcripts_reviewed.json",
        "AFTER STEP 4b — Auto-annotate (accepted transcripts)"
    )

    # Step 5 output
    try:
        import pandas
        check_manifests()
    except ImportError:
        print("\n  [SKIP] pandas not installed — cannot check final_dataset manifests")

    print()


if __name__ == "__main__":
    main()
