"""
Merge all transcript shard folders into one combined transcript set.

By default this reads every direct subfolder under:
  dataset_pipeline/transcripts/

Each input folder is expected to contain transcripts.json. The script removes
duplicates by clip_id, writes the merged JSON, and regenerates the review TSVs:
  transcripts.json
  for_human_review.tsv
  code_switched.tsv
  code_switched_clips.tsv

Example:
  python dataset_pipeline/06_merge_all_transcripts.py
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "transcripts"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR

REVIEW_COLUMNS = [
    "clip_id",
    "audio_path",
    "duration_sec",
    "auto_transcript",
    "is_code_switched",
    "dialect",
    "domain",
    "bn_ratio",
    "en_ratio",
    "cs_confidence",
    "switch_count",
    "human_transcript",
    "notes",
]

CODE_SWITCHED_COLUMNS = [
    "clip_id",
    "audio_path",
    "duration_sec",
    "auto_transcript",
    "dialect",
    "domain",
    "bn_ratio",
    "en_ratio",
    "cs_confidence",
    "switch_count",
    "human_transcript",
    "notes",
]


def shard_sort_key(path: Path) -> tuple[int, str]:
    """Sort range folders in their natural shard order."""
    match = re.search(r"_(\d+)(?:_\d+)?$", path.name)
    start = int(match.group(1)) if match else 0
    return start, path.name.lower()


def discover_shards(input_dir: Path) -> list[Path]:
    shards = [
        child
        for child in input_dir.iterdir()
        if child.is_dir() and (child / "transcripts.json").exists()
    ]
    return sorted(shards, key=shard_sort_key)


def load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list")

    records: list[dict[str, Any]] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{path} row {index} is not a JSON object")
        records.append(item)
    return records


def merge_records(shards: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    merged: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []

    for shard in shards:
        transcript_path = shard / "transcripts.json"
        records = load_json_records(transcript_path)
        print(f"[LOAD] {transcript_path}: {len(records)} records")

        for row_number, record in enumerate(records, start=1):
            clip_id = str(record.get("clip_id", "")).strip()
            if not clip_id:
                raise ValueError(f"{transcript_path} row {row_number} has no clip_id")

            if clip_id in seen:
                duplicates.append(
                    {
                        "clip_id": clip_id,
                        "kept_from": seen[clip_id],
                        "skipped_from": str(transcript_path),
                    }
                )
                continue

            seen[clip_id] = str(transcript_path)
            merged.append(record)

    return merged, duplicates


def review_row(record: dict[str, Any]) -> list[Any]:
    return [
        record.get("clip_id", ""),
        record.get("audio_path", ""),
        record.get("duration_sec", ""),
        record.get("transcript", record.get("auto_transcript", "")),
        record.get("is_code_switched", False),
        record.get("dialect", ""),
        record.get("domain", "General"),
        record.get("bn_ratio", 0.0),
        record.get("en_ratio", 0.0),
        record.get("cs_confidence", "NONE"),
        record.get("switch_count", 0),
        record.get("human_transcript", ""),
        record.get("notes", ""),
    ]


def code_switched_row(record: dict[str, Any]) -> list[Any]:
    return [
        record.get("clip_id", ""),
        record.get("audio_path", ""),
        record.get("duration_sec", ""),
        record.get("transcript", record.get("auto_transcript", "")),
        record.get("dialect", ""),
        record.get("domain", "General"),
        record.get("bn_ratio", 0.0),
        record.get("en_ratio", 0.0),
        record.get("cs_confidence", "NONE"),
        record.get("switch_count", 0),
        record.get("human_transcript", ""),
        record.get("notes", ""),
    ]


def write_tsv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def save_outputs(
    records: list[dict[str, Any]],
    duplicates: list[dict[str, str]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "transcripts.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    write_tsv(
        output_dir / "for_human_review.tsv",
        REVIEW_COLUMNS,
        [review_row(record) for record in records],
    )

    code_switched_records = [
        record for record in records if bool(record.get("is_code_switched", False))
    ]
    code_switched_rows = [code_switched_row(record) for record in code_switched_records]

    write_tsv(output_dir / "code_switched.tsv", CODE_SWITCHED_COLUMNS, code_switched_rows)
    write_tsv(
        output_dir / "code_switched_clips.tsv",
        CODE_SWITCHED_COLUMNS,
        code_switched_rows,
    )

    if duplicates:
        with (output_dir / "duplicates.json").open("w", encoding="utf-8") as f:
            json.dump(duplicates, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge transcript shard folders and remove duplicate clip_ids."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing transcript shard subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where merged outputs will be written.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=13273,
        help="Warn if the deduped transcript count does not match this value.",
    )
    args = parser.parse_args()

    shards = discover_shards(args.input_dir)
    if not shards:
        raise SystemExit(f"No transcript shards found in {args.input_dir}")

    print("[SHARDS]")
    for shard in shards:
        print(f"  {shard}")

    merged, duplicates = merge_records(shards)
    save_outputs(merged, duplicates, args.output_dir)

    code_switched_count = sum(
        1 for record in merged if bool(record.get("is_code_switched", False))
    )

    print("")
    print(f"Merged unique records: {len(merged)}")
    print(f"Skipped duplicate records: {len(duplicates)}")
    print(f"Code-switched records: {code_switched_count}")
    print(f"Output directory: {args.output_dir}")

    if args.expected_count and len(merged) != args.expected_count:
        print(
            f"[WARN] Expected {args.expected_count} records, "
            f"but merged {len(merged)} unique clip_ids."
        )


if __name__ == "__main__":
    main()
