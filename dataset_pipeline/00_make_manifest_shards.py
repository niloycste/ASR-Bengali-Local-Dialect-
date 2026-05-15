"""
Create balanced manifest shards for parallel transcription workers.

Reads:
  segments/segments_manifest.json

Writes:
  segments/manifest_shards/segments_manifest_worker1.json
  segments/manifest_shards/segments_manifest_worker2.json
  segments/manifest_shards/segments_manifest_worker3.json
  segments/manifest_shards/segments_manifest_worker4.json

The default 4-way split is dialect-balanced by clip count. It keeps each dialect
entirely in one shard, which makes worker progress easy to reason about.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "segments" / "segments_manifest.json"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "segments" / "manifest_shards"


def load_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def shard_by_dialect(clips: list[dict], num_shards: int) -> list[list[dict]]:
    by_dialect: dict[str, list[dict]] = defaultdict(list)
    for clip in clips:
        by_dialect[clip.get("dialect", "unknown")].append(clip)

    dialect_groups = sorted(
        by_dialect.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    shards: list[list[dict]] = [[] for _ in range(num_shards)]
    shard_counts = [0 for _ in range(num_shards)]

    for _dialect, dialect_clips in dialect_groups:
        target = min(range(num_shards), key=lambda idx: shard_counts[idx])
        shards[target].extend(dialect_clips)
        shard_counts[target] += len(dialect_clips)

    return shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num-shards", type=int, default=4)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clips = load_manifest(manifest_path)
    shards = shard_by_dialect(clips, args.num_shards)

    print(f"Loaded {len(clips)} clips from {manifest_path}")
    for idx, shard in enumerate(shards, start=1):
        out_path = output_dir / f"segments_manifest_worker{idx}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(shard, f, ensure_ascii=False, indent=2)

        dialect_counts: dict[str, int] = {}
        for clip in shard:
            dialect = clip.get("dialect", "unknown")
            dialect_counts[dialect] = dialect_counts.get(dialect, 0) + 1

        print(f"\nworker{idx}: {len(shard)} clips -> {out_path}")
        for dialect, count in sorted(dialect_counts.items()):
            print(f"  {dialect}: {count}")


if __name__ == "__main__":
    main()
