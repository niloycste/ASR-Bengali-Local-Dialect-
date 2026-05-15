"""
Step 2: Segment long audio files into short clips (5–15 seconds)
Uses Voice Activity Detection (VAD) to cut at silence points
"""

import os
import json
import numpy as np
from pathlib import Path

from pipeline_utils import exit_with_import_error, require_command_runs, require_commands

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR  = str(SCRIPT_DIR / "raw_audio")
OUTPUT_DIR = str(SCRIPT_DIR / "segments")
MIN_CLIP_SEC = 5      # discard clips shorter than this (consistent with README)
MAX_CLIP_SEC = 15     # split clips longer than this
VAD_AGGRESSIVENESS = 2  # 0=least aggressive, 3=most aggressive (cuts more)
SAMPLE_RATE = 16000
FRAME_MS = 30         # VAD frame size in milliseconds
SUPPORTED_AUDIO_EXTS = {".wav", ".webm", ".m4a", ".opus", ".mp3", ".flac", ".ogg", ".aac"}


def clip_id_prefix(clip_id: str) -> str:
    """Return the source prefix portion from a clip id like stem_0000."""
    prefix, sep, tail = clip_id.rpartition("_")
    if sep and tail.isdigit():
        return prefix
    return clip_id


def manifest_relpath(path: str | Path) -> str:
    """Store clip paths relative to this script directory for portability."""
    return os.path.relpath(os.fspath(path), os.fspath(SCRIPT_DIR))


def resolve_manifest_path(path: str) -> str:
    """Resolve a manifest path entry to an absolute filesystem path."""
    if os.path.isabs(path):
        return path
    return os.path.join(os.fspath(SCRIPT_DIR), path)


def vad_segment(audio_path: str, output_dir: str, file_prefix: str) -> list:
    """
    Use WebRTC VAD to find speech regions and segment audio.
    Returns list of (start_ms, end_ms, output_filepath).
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        import webrtcvad
    except ImportError:
        exit_with_import_error("webrtcvad", "webrtcvad-wheels")

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    try:
        from pydub import AudioSegment
    except ImportError:
        exit_with_import_error("pydub")

    audio = AudioSegment.from_file(audio_path)
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)

    frame_len = int(SAMPLE_RATE * FRAME_MS / 1000)  # samples per frame
    raw = audio.raw_data

    # Split into frames and run VAD
    frames = []
    for i in range(0, len(raw), frame_len * 2):
        frame = raw[i: i + frame_len * 2]
        if len(frame) < frame_len * 2:
            break  # last fragment too short for VAD — skip it
        is_speech = vad.is_speech(frame, SAMPLE_RATE)
        frames.append(is_speech)

    # Merge consecutive speech frames into segments
    segments = []
    in_speech = False
    start_frame = 0
    padding = 10  # frames of silence padding around speech

    for i, is_speech in enumerate(frames):
        if not in_speech and is_speech:
            start_frame = max(0, i - padding)
            in_speech = True
        elif in_speech and not is_speech:
            # Check if enough silence has passed
            end_frame = min(len(frames) - 1, i + padding)
            if all(not frames[j] for j in range(i, min(i + padding, len(frames)))):
                in_speech = False
                segments.append((start_frame, end_frame))

    if in_speech:
        segments.append((start_frame, len(frames) - 1))

    # Convert frame indices to milliseconds and save clips
    saved_clips = []
    clip_idx = 0

    for (s_frame, e_frame) in segments:
        start_ms = s_frame * FRAME_MS
        end_ms   = e_frame * FRAME_MS
        duration_sec = (end_ms - start_ms) / 1000

        if duration_sec < MIN_CLIP_SEC:
            continue

        # Split very long segments
        if duration_sec > MAX_CLIP_SEC:
            # Split into equal sub-clips
            n_splits = int(np.ceil(duration_sec / MAX_CLIP_SEC))
            split_len = (end_ms - start_ms) / n_splits
            for k in range(n_splits):
                sub_start = start_ms + k * split_len
                sub_end   = start_ms + (k + 1) * split_len
                clip = audio[sub_start:sub_end]
                out_path = os.path.join(output_dir, f"{file_prefix}_{clip_idx:04d}.wav")
                clip.export(out_path, format="wav")
                saved_clips.append({
                    "clip_id": f"{file_prefix}_{clip_idx:04d}",
                    "path": manifest_relpath(out_path),
                    "start_ms": int(sub_start),
                    "end_ms": int(sub_end),
                    "duration_sec": round((sub_end - sub_start) / 1000, 2)
                })
                clip_idx += 1
        else:
            clip = audio[start_ms:end_ms]
            out_path = os.path.join(output_dir, f"{file_prefix}_{clip_idx:04d}.wav")
            clip.export(out_path, format="wav")
            saved_clips.append({
                "clip_id": f"{file_prefix}_{clip_idx:04d}",
                "path": manifest_relpath(out_path),
                "start_ms": int(start_ms),
                "end_ms": int(end_ms),
                "duration_sec": round(duration_sec, 2)
            })
            clip_idx += 1

    print(f"  {file_prefix}: {len(saved_clips)} clips saved "
          f"(total {sum(c['duration_sec'] for c in saved_clips)/60:.1f} min)")
    return saved_clips


def load_folder_metadata(input_dir: str) -> dict:
    """
    Load folder→dialect mapping saved by 01_download_audio.py.
    Falls back to using folder name as dialect if file not found.

    This fixes the labeling bug where topic buckets like "Tech_IT_CS"
    were incorrectly used as dialect labels.
    """
    meta_path = os.path.join(input_dir, "folder_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            data = json.load(f)
        try:
            from download_query_profiles import YOUTUBE_FOLDER_METADATA
            for key, value in YOUTUBE_FOLDER_METADATA.items():
                data.setdefault(key, value)
        except ImportError:
            pass
        print(f"Loaded folder metadata from: {meta_path}")
        return data
    print(f"  [WARNING] folder_metadata.json not found at {meta_path}")
    print(f"  Dialect labels will use folder names — run 01_download_audio.py first.")
    return {}


def load_existing_segments(output_dir: str) -> tuple[list[dict], set[str]]:
    """Load existing manifest entries so new runs can append safely."""
    manifest_path = os.path.join(output_dir, "segments_manifest.json")
    if not os.path.exists(manifest_path):
        return [], set()

    with open(manifest_path, encoding="utf-8") as f:
        existing_entries = json.load(f)

    kept_entries = []
    for entry in existing_entries:
        raw_path = entry.get("path", "")
        resolved_path = resolve_manifest_path(raw_path)
        if os.path.exists(resolved_path):
            entry["path"] = manifest_relpath(resolved_path)
            entry["source_id"] = entry.get("source_id") or clip_id_prefix(entry["clip_id"])
            if entry.get("source_path"):
                entry["source_path"] = manifest_relpath(resolve_manifest_path(entry["source_path"]))
            kept_entries.append(entry)

    existing_entries = kept_entries
    existing_prefixes = {entry["source_id"] for entry in existing_entries}
    print(
        f"Loaded existing manifest: {len(existing_entries)} clips "
        f"across {len(existing_prefixes)} segmented sources"
    )
    return existing_entries, existing_prefixes


def resolve_bucket_metadata(audio_path: Path, input_path: Path, folder_meta: dict) -> tuple[str, str, str, str]:
    """
    Resolve dialect/domain from the raw_audio relative path.

    Supports:
      raw_audio/youtube/<bucket>/<file>
      raw_audio/extra/<dialect>/<file>
      raw_audio/<legacy_folder>/<file>
    """
    rel_parts = audio_path.relative_to(input_path).parts
    source_root = rel_parts[0] if rel_parts else ""
    bucket_name = audio_path.parent.name
    candidate_keys = []

    if len(rel_parts) >= 3:
        bucket_name = rel_parts[1]
        if source_root == "extra":
            candidate_keys.append(f"extra_{bucket_name}")
        candidate_keys.append(bucket_name)
    else:
        candidate_keys.append(bucket_name)

    meta = {}
    for key in candidate_keys:
        meta = folder_meta.get(key, {})
        if meta:
            break

    dialect = meta.get("dialect")
    domain = meta.get("domain")

    # Fallback for bucket names like "Barishal__Tech_Review" when metadata is stale.
    if not dialect and "__" in bucket_name:
        dialect, _, inferred_domain = bucket_name.partition("__")
        domain = domain or inferred_domain or "General"

    dialect = dialect or bucket_name
    domain = domain or "General"
    return dialect, domain, bucket_name, source_root


def build_output_prefix(audio_path: Path, bucket_name: str, dialect: str, source_root: str) -> str:
    """Create a stable, collision-safe prefix for output clips."""
    stem = audio_path.stem
    if source_root == "extra":
        return f"extra__{bucket_name}__{stem}"
    if bucket_name != dialect:
        return f"{bucket_name}__{stem}"
    return stem


def preflight_segmentation_dependencies():
    """Ensure ffmpeg tools needed by pydub are available."""
    resolved = require_commands(
        [
            ("ffmpeg", ("ffmpeg",), "install ffmpeg and add it to PATH"),
            ("ffprobe", ("ffprobe",), "install ffmpeg and add it to PATH"),
        ],
        context="audio segmentation",
    )
    require_command_runs("ffmpeg", [resolved["ffmpeg"], "-version"], context="audio segmentation")
    require_command_runs("ffprobe", [resolved["ffprobe"], "-version"], context="audio segmentation")


def segment_all(input_dir: str = INPUT_DIR, output_dir: str = OUTPUT_DIR):
    """
    Segment all supported audio files in input_dir.
    Reads folder_metadata.json (written by 01_download_audio.py) to assign
    correct dialect and domain labels — folder names like 'Tech_IT_CS' are
    topic buckets, not dialect names.
    """
    input_path   = Path(input_dir)
    audio_files  = [
        path for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTS
    ]
    folder_meta  = load_folder_metadata(input_dir)
    existing_entries, existing_prefixes = load_existing_segments(output_dir)

    print(f"Found {len(audio_files)} audio files to segment.\n")

    new_clips = []
    for audio_path in audio_files:
        dialect, domain, bucket_name, source_root = resolve_bucket_metadata(audio_path, input_path, folder_meta)
        prefix = build_output_prefix(audio_path, bucket_name, dialect, source_root)

        if prefix in existing_prefixes:
            print(f"Skipping already-segmented source [{bucket_name}]: {audio_path.name}")
            continue

        print(f"Segmenting [{dialect} / {domain}] from [{bucket_name}]: {audio_path.name}")

        # Save segments into dialect subfolder
        clip_output_dir = os.path.join(output_dir, dialect)
        clips = vad_segment(str(audio_path), clip_output_dir, prefix)

        # Tag each clip with dialect AND domain (both stored separately)
        for clip in clips:
            clip["dialect"] = dialect   # e.g. "Sylhet", "Dhaka", "unknown"
            clip["domain"]  = domain    # e.g. "Tech_IT", "Medical", "General"
            clip["source_id"] = prefix
            clip["source_path"] = manifest_relpath(audio_path)
            clip["source_bucket"] = bucket_name
            clip["source_root"] = source_root
        new_clips.extend(clips)
        existing_prefixes.add(prefix)

    all_clips = existing_entries + new_clips

    # Save segment manifest
    manifest_path = os.path.join(output_dir, "segments_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(all_clips, f, ensure_ascii=False, indent=2)

    total_hours = sum(c["duration_sec"] for c in all_clips) / 3600
    print(f"\nExisting clips kept: {len(existing_entries)}")
    print(f"New clips added: {len(new_clips)}")
    print(f"Total clips: {len(all_clips)}")
    print(f"Total duration: {total_hours:.2f} hours")
    print(f"Manifest saved: {manifest_path}")

    # Per-dialect summary
    from collections import Counter
    dialect_counts = Counter(c["dialect"] for c in all_clips)
    print("\nPer-dialect breakdown:")
    for d, count in sorted(dialect_counts.items()):
        hrs = sum(c["duration_sec"] for c in all_clips if c["dialect"] == d) / 3600
        print(f"  {d}: {count} clips ({hrs:.2f} hrs)")

    return all_clips


if __name__ == "__main__":
    preflight_segmentation_dependencies()
    segment_all()
