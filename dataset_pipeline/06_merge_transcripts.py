"""
Merge transcript outputs from parallel workers.

Expected worker outputs:
  transcripts_worker1/transcripts.json
  transcripts_worker2/transcripts.json
  transcripts_worker3/transcripts.json
  transcripts_worker4/transcripts.json

Writes:
  transcripts/transcripts.json
  transcripts/for_human_review.tsv
  transcripts/code_switched_clips.tsv

This keeps the original transcript schema and reuses 03_auto_transcribe.save_results
to regenerate TSV files from the merged JSON records.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from importlib.machinery import SourceFileLoader

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "transcripts"

auto_transcribe = SourceFileLoader(
    "auto_transcribe",
    str(SCRIPT_DIR / "03_auto_transcribe.py"),
).load_module()


def load_worker_transcripts(worker_dir: Path) -> list[dict]:
    path = worker_dir / "transcripts.json"
    if not path.exists():
        print(f"[SKIP] {path} not found")
        return []
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    print(f"[LOAD] {path}: {len(records)} records")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers",
        nargs="+",
        default=[
            str(SCRIPT_DIR / "transcripts_worker1"),
            str(SCRIPT_DIR / "transcripts_worker2"),
            str(SCRIPT_DIR / "transcripts_worker3"),
            str(SCRIPT_DIR / "transcripts_worker4"),
        ],
        help="Worker transcript directories to merge.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    merged: list[dict] = []
    seen_clip_ids: set[str] = set()
    duplicates = 0

    for worker in args.workers:
        for record in load_worker_transcripts(Path(worker)):
            clip_id = record.get("clip_id")
            if clip_id in seen_clip_ids:
                duplicates += 1
                continue
            seen_clip_ids.add(clip_id)
            merged.append(record)

    merged.sort(key=lambda item: item.get("clip_id", ""))
    output_dir = Path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    auto_transcribe.save_results(merged, str(output_dir))
    print(f"\nMerged records: {len(merged)}")
    print(f"Skipped duplicate clip_ids: {duplicates}")
    print(f"Output: {output_dir / 'transcripts.json'}")


if __name__ == "__main__":
    main()
