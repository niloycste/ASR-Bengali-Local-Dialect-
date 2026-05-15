"""
Merge Data ASR transcript worker folders into one combined output.

Reads by default:
  dataset_pipeline/Data ASR/transcripts_worker1/transcripts.json
  dataset_pipeline/Data ASR/transcripts_worker2/transcripts.json
  dataset_pipeline/Data ASR/transcripts_worker3/transcripts.json
  dataset_pipeline/Data ASR/transcripts_worker4/transcripts.json

Writes by default:
  dataset_pipeline/Data ASR/transcripts_merged_all/transcripts.json
  dataset_pipeline/Data ASR/transcripts_merged_all/for_human_review.tsv
  dataset_pipeline/Data ASR/transcripts_merged_all/for_human_review.csv
  dataset_pipeline/Data ASR/transcripts_merged_all/code_switched.tsv
  dataset_pipeline/Data ASR/transcripts_merged_all/code_switched.csv
  dataset_pipeline/Data ASR/transcripts_merged_all/code_switched_clips.tsv
  dataset_pipeline/Data ASR/transcripts_merged_all/code_switched_clips.csv
  dataset_pipeline/Data ASR/transcripts_merged_all/needs_encoding_review.csv
  dataset_pipeline/Data ASR/transcripts_merged_all/needs_code_switched_encoding_review.csv
  dataset_pipeline/Data ASR/transcripts_merged_all/duplicates.json  (only if needed)

Example:
  python dataset_pipeline/07_merge_data_asr_workers.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_ASR_DIR = SCRIPT_DIR / "Data ASR"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_ASR_DIR / "transcripts_merged_all"

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

REPLACEMENT_CHAR = "\ufffd"


def load_records(path: Path) -> list[dict[str, Any]]:
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


def merge_worker_folders(worker_dirs: list[Path]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    merged: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    duplicates: list[dict[str, str]] = []

    for worker_dir in worker_dirs:
        path = worker_dir / "transcripts.json"
        if not path.exists():
            print(f"[SKIP] Missing {path}")
            continue

        records = load_records(path)
        print(f"[LOAD] {path}: {len(records)} records")

        for row_number, record in enumerate(records, start=1):
            clip_id = str(record.get("clip_id", "")).strip()
            if not clip_id:
                raise ValueError(f"{path} row {row_number} has no clip_id")

            if clip_id in seen:
                duplicates.append(
                    {
                        "clip_id": clip_id,
                        "kept_from": seen[clip_id],
                        "skipped_from": str(path),
                    }
                )
                continue

            seen[clip_id] = str(path)
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


def write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def try_write_csv(path: Path, header: list[str], rows: list[list[Any]]) -> bool:
    try:
        write_csv(path, header, rows)
        return True
    except PermissionError:
        print(f"[WARN] Could not write {path}. Close it in Excel or any editor and rerun.")
        return False


def save_outputs(
    records: list[dict[str, Any]],
    duplicates: list[dict[str, str]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "transcripts.json").open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    needs_encoding_review = [
        record for record in records if REPLACEMENT_CHAR in str(record.get("transcript", ""))
    ]
    review_records = [
        record
        for record in records
        if REPLACEMENT_CHAR not in str(record.get("transcript", ""))
    ]
    review_rows = [review_row(record) for record in review_records]
    write_tsv(output_dir / "for_human_review.tsv", REVIEW_COLUMNS, review_rows)
    try_write_csv(output_dir / "for_human_review.csv", REVIEW_COLUMNS, review_rows)

    code_switched = [record for record in records if record.get("is_code_switched")]
    needs_code_switched_encoding_review = [
        record
        for record in code_switched
        if REPLACEMENT_CHAR in str(record.get("transcript", ""))
    ]
    review_code_switched = [
        record
        for record in code_switched
        if REPLACEMENT_CHAR not in str(record.get("transcript", ""))
    ]
    code_switched_rows = [code_switched_row(record) for record in review_code_switched]
    write_tsv(output_dir / "code_switched.tsv", CODE_SWITCHED_COLUMNS, code_switched_rows)
    try_write_csv(output_dir / "code_switched.csv", CODE_SWITCHED_COLUMNS, code_switched_rows)
    write_tsv(
        output_dir / "code_switched_clips.tsv",
        CODE_SWITCHED_COLUMNS,
        code_switched_rows,
    )
    try_write_csv(
        output_dir / "code_switched_clips.csv",
        CODE_SWITCHED_COLUMNS,
        code_switched_rows,
    )

    try_write_csv(
        output_dir / "needs_encoding_review.csv",
        REVIEW_COLUMNS,
        [review_row(record) for record in needs_encoding_review],
    )
    try_write_csv(
        output_dir / "needs_code_switched_encoding_review.csv",
        CODE_SWITCHED_COLUMNS,
        [code_switched_row(record) for record in needs_code_switched_encoding_review],
    )

    if duplicates:
        with (output_dir / "duplicates.json").open("w", encoding="utf-8") as f:
            json.dump(duplicates, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge Data ASR transcripts_worker1-4 into one transcript set."
    )
    parser.add_argument(
        "--data-asr-dir",
        type=Path,
        default=DEFAULT_DATA_ASR_DIR,
        help="Directory containing transcripts_worker1, transcripts_worker2, etc.",
    )
    parser.add_argument(
        "--workers",
        nargs="+",
        default=[
            "transcripts_worker1",
            "transcripts_worker2",
            "transcripts_worker3",
            "transcripts_worker4",
        ],
        help="Worker folder names under --data-asr-dir, in merge priority order.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where merged outputs will be written.",
    )
    args = parser.parse_args()

    worker_dirs = [args.data_asr_dir / worker for worker in args.workers]

    print("[WORKERS]")
    for worker_dir in worker_dirs:
        print(f"  {worker_dir}")

    merged, duplicates = merge_worker_folders(worker_dirs)
    save_outputs(merged, duplicates, args.output_dir)

    code_switched_count = sum(1 for record in merged if record.get("is_code_switched"))
    needs_encoding_review_count = sum(
        1 for record in merged if REPLACEMENT_CHAR in str(record.get("transcript", ""))
    )
    needs_code_switched_encoding_review_count = sum(
        1
        for record in merged
        if record.get("is_code_switched")
        and REPLACEMENT_CHAR in str(record.get("transcript", ""))
    )

    print("")
    print(f"Merged unique records: {len(merged)}")
    print(f"Skipped duplicate records: {len(duplicates)}")
    print(f"Code-switched records: {code_switched_count}")
    print(f"Records needing encoding review: {needs_encoding_review_count}")
    print(
        "Code-switched records needing encoding review: "
        f"{needs_code_switched_encoding_review_count}"
    )
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
