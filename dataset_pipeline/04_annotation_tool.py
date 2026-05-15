"""
Step 4: Review transcripts either in Excel/Sheets or interactively in terminal.

Default mode:
  Import corrections from transcripts/for_human_review.tsv and write
  transcripts/transcripts_reviewed.json for dataset building.

Interactive mode:
  python 04_annotation_tool.py --interactive
"""

from __future__ import annotations

import csv
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR / "Data ASR" / "transcripts_merged_all" / "transcripts.json").exists():
    TARGET_DIR = SCRIPT_DIR / "Data ASR" / "transcripts_merged_all"
else:
    TARGET_DIR = SCRIPT_DIR / "transcripts"

TRANSCRIPTS_PATH = str(TARGET_DIR / "transcripts.json")
REVIEW_TSV_PATH  = str(TARGET_DIR / "for_human_review.tsv")
REVIEWED_PATH    = str(SCRIPT_DIR / "transcripts" / "transcripts_reviewed.json")


def resolve_pipeline_path(path: str) -> str:
    """Resolve dataset_pipeline-relative paths."""
    if os.path.isabs(path):
        return path
    return os.path.join(os.fspath(SCRIPT_DIR), path)


def play_audio(path: str):
    """Play audio file in terminal (cross-platform)."""
    path = resolve_pipeline_path(path)
    system = platform.system()
    if system == "Windows":
        os.startfile(path)
    elif system == "Darwin":
        subprocess.Popen(["afplay", path])
    else:
        subprocess.Popen(["aplay", path])


def import_review_tsv(
    transcripts_path: str = TRANSCRIPTS_PATH,
    review_tsv_path: str = REVIEW_TSV_PATH,
    reviewed_path: str = REVIEWED_PATH,
):
    """
    Import spreadsheet review and build reviewed JSON.

    If a row leaves human_transcript blank, the auto transcript is kept.
    This makes the Excel workflow usable even when only a subset is edited.
    """
    if not os.path.exists(transcripts_path):
        print(f"[ERROR] Missing transcripts JSON: {transcripts_path}")
        raise SystemExit(1)
    if not os.path.exists(review_tsv_path):
        print(f"[ERROR] Missing review TSV: {review_tsv_path}")
        print("        Run 03_auto_transcribe.py first, then edit the TSV.")
        raise SystemExit(1)

    with open(transcripts_path, encoding="utf-8") as f:
        clips = json.load(f)

    rows_by_id = {}
    with open(review_tsv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            clip_id = (row.get("clip_id") or "").strip()
            if clip_id:
                rows_by_id[clip_id] = row

    reviewed = []
    for clip in clips:
        reviewed_clip = dict(clip)
        row = rows_by_id.get(clip["clip_id"], {})

        auto_transcript = (row.get("auto_transcript") or clip.get("transcript", "")).strip()
        human_transcript = (row.get("human_transcript") or "").strip()

        reviewed_clip["human_transcript"] = human_transcript or auto_transcript
        reviewed_clip["needs_human_review"] = False

        dialect = (row.get("dialect") or "").strip()
        if dialect:
            reviewed_clip["dialect"] = dialect

        domain = (row.get("domain") or "").strip()
        if domain:
            reviewed_clip["domain"] = domain

        notes = (row.get("notes") or "").strip()
        if notes:
            reviewed_clip["notes"] = notes

        reviewed.append(reviewed_clip)

    with open(reviewed_path, "w", encoding="utf-8") as f:
        json.dump(reviewed, f, ensure_ascii=False, indent=2)

    print(f"Imported {len(reviewed)} reviewed clips from: {review_tsv_path}")
    print(f"Saved reviewed JSON: {reviewed_path}")


def annotate_interactively(
    transcripts_path: str = TRANSCRIPTS_PATH,
    reviewed_path: str = REVIEWED_PATH,
):
    """Interactive annotation loop with audio playback."""
    with open(transcripts_path, encoding="utf-8") as f:
        clips = json.load(f)

    if os.path.exists(reviewed_path):
        with open(reviewed_path, encoding="utf-8") as f:
            reviewed = json.load(f)
        reviewed_ids = {r["clip_id"] for r in reviewed}
    else:
        reviewed = []
        reviewed_ids = set()

    pending = [c for c in clips if c["clip_id"] not in reviewed_ids]
    print(f"\nTotal clips: {len(clips)}")
    print(f"Already reviewed: {len(reviewed_ids)}")
    print(f"Remaining: {len(pending)}")
    print("\nInstructions:")
    print("  Press ENTER to keep auto transcript")
    print("  Type corrected transcript to replace")
    print("  Type 'skip' to skip this clip")
    print("  Type 'quit' to save and exit\n")

    for i, clip in enumerate(pending):
        print(f"\n{'='*60}")
        print(f"Clip {i+1}/{len(pending)}: {clip['clip_id']}")
        print(f"Dialect: {clip.get('dialect', 'unknown')}")
        print(f"Domain: {clip.get('domain', 'General')}")
        print(f"Duration: {clip['duration_sec']}s")
        print(f"Code-switched: {clip['is_code_switched']}")
        print(f"\nAuto transcript:\n  {clip['transcript']}")

        audio_path = resolve_pipeline_path(clip["audio_path"])
        if os.path.exists(audio_path):
            print("\n[Playing audio...]")
            play_audio(audio_path)
        else:
            print(f"\n[Audio not found: {clip['audio_path']}]")

        user_input = input("\nYour correction (Enter=keep, skip, quit): ").strip()

        if user_input.lower() == "quit":
            break
        if user_input.lower() == "skip":
            continue

        clip["human_transcript"] = user_input or clip["transcript"]
        clip["needs_human_review"] = False

        dialect = input(f"Dialect [{clip.get('dialect', 'unknown')}]: ").strip()
        if dialect:
            clip["dialect"] = dialect

        domain = input(f"Domain [{clip.get('domain', 'General')}]: ").strip()
        if domain:
            clip["domain"] = domain

        reviewed.append(clip)

        with open(reviewed_path, "w", encoding="utf-8") as f:
            json.dump(reviewed, f, ensure_ascii=False, indent=2)

    print(f"\nAnnotation session complete.")
    print(f"Reviewed {len(reviewed)} clips total.")
    print(f"Saved to: {reviewed_path}")


if __name__ == "__main__":
    if "--interactive" in sys.argv:
        annotate_interactively()
    else:
        import_review_tsv()
