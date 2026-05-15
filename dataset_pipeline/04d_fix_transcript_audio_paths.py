"""
Fix transcript audio_path values before building the final dataset.

Some Colab/Hugging Face extraction layouts can produce paths like:
  Barishal/clip.wav

The canonical project layout is:
  segments/Barishal/clip.wav

This script checks transcript JSON files and rewrites audio_path to the
canonical "segments/..." form whenever that file exists.
"""

from __future__ import annotations

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
TRANSCRIPTS_DIR = SCRIPT_DIR / "transcripts"
MERGED_DIR = SCRIPT_DIR / "Data ASR" / "transcripts_merged_all"

TARGET_FILES = [
    TRANSCRIPTS_DIR / "transcripts.json",
    TRANSCRIPTS_DIR / "transcripts_reviewed.json",
    MERGED_DIR / "transcripts.json",
    MERGED_DIR / "transcripts_reviewed.json",
]


def canonical_audio_path(path: str) -> str:
    normalized = path.replace("\\", "/").lstrip("/")
    if not normalized or normalized.startswith("segments/"):
        return normalized

    direct = SCRIPT_DIR / normalized
    under_segments = SCRIPT_DIR / "segments" / normalized

    if under_segments.exists() and not direct.exists():
        return f"segments/{normalized}"
    if under_segments.exists():
        return f"segments/{normalized}"
    return normalized


def fix_file(path: Path) -> None:
    if not path.exists():
        print(f"[SKIP] {path} not found")
        return

    with open(path, encoding="utf-8") as f:
        clips = json.load(f)

    fixed = 0
    missing_after_fix = 0
    for clip in clips:
        old_path = str(clip.get("audio_path", ""))
        new_path = canonical_audio_path(old_path)
        if new_path != old_path:
            clip["audio_path"] = new_path
            fixed += 1

        resolved = SCRIPT_DIR / clip.get("audio_path", "")
        if not resolved.exists():
            missing_after_fix += 1

    backup = path.with_suffix(path.suffix + ".path_backup")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(clips, f, ensure_ascii=False, indent=2)

    print(f"[OK] {path.name}: fixed={fixed}, missing_after_fix={missing_after_fix}")
    print(f"     backup: {backup.name}")


def main() -> None:
    for path in TARGET_FILES:
        fix_file(path)


if __name__ == "__main__":
    main()
