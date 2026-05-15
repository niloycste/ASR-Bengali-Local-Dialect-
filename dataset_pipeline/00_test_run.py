"""
TEST RUN - Full pipeline on small data (2 videos per source bucket).

Run this first before the full download. It will:
  1. Download a small sample from YouTube
  2. Segment into clips
  3. Transcribe with Whisper
  4. Print a quality report
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from pathlib import Path

from code_mixing_utils import classify_code_mixing, detect_word_tags
from pipeline_utils import (
    exit_with_import_error,
    require_commands,
    require_command_runs,
    warn_if_missing_js_runtime,
)

# Configuration
TEST_DIR     = "test_run"
SAMPLE_RATE  = 16000
AUDIO_FORMAT = "wav"
VIDEOS_PER_BUCKET = 2
MIN_CLIP_SEC = 5
MAX_CLIP_SEC = 15
WHISPER_MODEL = "small"

# One representative query per source bucket.
# These are biased toward local dialect speech with English inserted inside it,
# not just pure regional dialect clips.
TEST_BUCKETS = {
    "Chittagong_CS": {
        "query": "চাটগাঁইয়া ভাষা english mixed vlog interview",
        "dialect": "Chittagong",
        "domain": "Regional_CS",
    },
    "Sylhet_Diaspora_CS": {
        "query": "british bangladeshi sylheti vlog",
        "dialect": "Sylhet",
        "domain": "Diaspora_Vlog",
    },
    "Barishal_CS": {
        "query": "বরিশাল ভাষা english mixed vlog interview",
        "dialect": "Barishal",
        "domain": "Regional_CS",
    },
    "Noakhali_CS": {
        "query": "নোয়াখালী ভাষা english mixed vlog interview",
        "dialect": "Noakhali",
        "domain": "Regional_CS",
    },
    "Rajshahi_CS": {
        "query": "রাজশাহী ভাষা english mixed vlog interview",
        "dialect": "Rajshahi",
        "domain": "Regional_CS",
    },
    "Dhaka_Urban_CS": {
        "query": "বাংলিশ কথোপকথন funny",
        "dialect": "Dhaka",
        "domain": "Urban_CS",
    },
    "Tech_IT_CS": {
        "query": "bangladesh software engineer interview bangla english",
        "dialect": "unknown",
        "domain": "Tech_IT",
    },
    "Medical_CS": {
        "query": "bangladeshi doctor bangla english lecture",
        "dialect": "unknown",
        "domain": "Medical",
    },
}


def preflight_test_dependencies():
    """Ensure download/transcription prerequisites exist before starting."""
    resolved = require_commands(
        [
            ("yt-dlp", ("yt-dlp",), "pip install yt-dlp"),
            ("ffmpeg", ("ffmpeg",), "install ffmpeg and add it to PATH"),
            ("ffprobe", ("ffprobe",), "install ffmpeg and add it to PATH"),
        ],
        context="test run",
    )
    require_command_runs("ffmpeg", [resolved["ffmpeg"], "-version"], context="test run")
    require_command_runs("ffprobe", [resolved["ffprobe"], "-version"], context="test run")
    warn_if_missing_js_runtime()


def reset_test_dir(path: str = TEST_DIR):
    """Remove stale test output so each run starts clean."""
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def download_test(buckets: dict, out_dir: str, n: int = VIDEOS_PER_BUCKET):
    """Download a small sample of audio files per source bucket."""
    print("\n" + "=" * 55)
    print(f"STEP 1 - Downloading {n} videos per source bucket")
    print("=" * 55)

    for bucket_name, spec in buckets.items():
        query = spec["query"]
        bucket_dir = os.path.join(out_dir, "audio", bucket_name)
        os.makedirs(bucket_dir, exist_ok=True)

        print(f"\n  [{bucket_name}] Searching: {query[:55]}")

        from pipeline_utils import find_command
        ffmpeg_path = find_command("ffmpeg")
        ffmpeg_dir  = os.path.dirname(ffmpeg_path) if ffmpeg_path else None
        ytdlp_exe   = find_command("yt-dlp") or "yt-dlp"

        cmd = [
            ytdlp_exe,
            f"ytsearch{n}:{query}",
            "--extract-audio",
            "--audio-format", AUDIO_FORMAT,
            "--audio-quality", "0",
            "--output", os.path.join(bucket_dir, "%(id)s.%(ext)s"),
            "--no-playlist",
            "--quiet",
            "--ignore-errors",
        ]
        if ffmpeg_dir:
            cmd += ["--ffmpeg-location", ffmpeg_dir]
        subprocess.run(cmd)

        downloaded = list(Path(bucket_dir).glob("*.wav"))
        print(f"  -> {len(downloaded)} files downloaded")

    print("\nDownload complete.")


def segment_audio(audio_path: str, out_dir: str, prefix: str) -> list:
    """Split one audio file into short clips using VAD."""
    try:
        import webrtcvad
    except ImportError:
        exit_with_import_error("webrtcvad", "webrtcvad-wheels")

    try:
        from pydub import AudioSegment
    except ImportError:
        exit_with_import_error("pydub")

    vad = webrtcvad.Vad(2)
    audio = AudioSegment.from_wav(audio_path)
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)

    frame_ms = 30
    frame_len = int(SAMPLE_RATE * frame_ms / 1000)
    raw = audio.raw_data
    frames = []

    for i in range(0, len(raw) - frame_len * 2, frame_len * 2):
        frame = raw[i: i + frame_len * 2]
        frames.append(vad.is_speech(frame, SAMPLE_RATE))

    segments, in_speech, start = [], False, 0
    padding = 10
    for i, is_speech in enumerate(frames):
        if not in_speech and is_speech:
            start = max(0, i - padding)
            in_speech = True
        elif in_speech and not is_speech:
            if all(not frames[j] for j in range(i, min(i + padding, len(frames)))):
                in_speech = False
                segments.append((start, min(len(frames) - 1, i + padding)))
    if in_speech:
        segments.append((start, len(frames) - 1))

    os.makedirs(out_dir, exist_ok=True)
    clips, idx = [], 0

    for start_frame, end_frame in segments:
        start_ms = start_frame * frame_ms
        end_ms = end_frame * frame_ms
        duration_sec = (end_ms - start_ms) / 1000

        if duration_sec < MIN_CLIP_SEC:
            continue

        if duration_sec > MAX_CLIP_SEC:
            import numpy as np

            n_splits = int(np.ceil(duration_sec / MAX_CLIP_SEC))
            split_len = (end_ms - start_ms) / n_splits
            for split_idx in range(n_splits):
                sub_start = start_ms + split_idx * split_len
                sub_end = start_ms + (split_idx + 1) * split_len
                path = os.path.join(out_dir, f"{prefix}_{idx:04d}.wav")
                audio[sub_start:sub_end].export(path, format="wav")
                clips.append(
                    {
                        "clip_id": f"{prefix}_{idx:04d}",
                        "path": path,
                        "duration_sec": round((sub_end - sub_start) / 1000, 2),
                    }
                )
                idx += 1
        else:
            path = os.path.join(out_dir, f"{prefix}_{idx:04d}.wav")
            audio[start_ms:end_ms].export(path, format="wav")
            clips.append(
                {
                    "clip_id": f"{prefix}_{idx:04d}",
                    "path": path,
                    "duration_sec": round(duration_sec, 2),
                }
            )
            idx += 1

    return clips


def segment_all_test(audio_base: str, out_dir: str, buckets: dict) -> list:
    """Segment all downloaded test audio and attach dialect/domain metadata."""
    print("\n" + "=" * 55)
    print("STEP 2 - Segmenting audio into clips")
    print("=" * 55)

    all_clips = []
    existing_dirs = {path.name for path in Path(audio_base).iterdir() if path.is_dir()} if os.path.exists(audio_base) else set()
    stale_dirs = sorted(existing_dirs - set(buckets))
    if stale_dirs:
        print(f"[INFO] Ignoring stale test bucket folders from previous runs: {', '.join(stale_dirs)}")

    for bucket_name, meta in buckets.items():
        bucket_dir = Path(audio_base) / bucket_name
        if not bucket_dir.exists():
            print(f"\n  [{meta['dialect']} / {meta['domain']}] 0 audio files")
            continue

        dialect = meta["dialect"]
        domain = meta["domain"]
        wav_files = list(bucket_dir.glob("*.wav"))
        print(f"\n  [{dialect} / {domain}] {len(wav_files)} audio files")

        for wav in wav_files:
            clips_dir = os.path.join(out_dir, "clips", dialect)
            clips = segment_audio(str(wav), clips_dir, wav.stem)
            for clip in clips:
                clip["dialect"] = dialect
                clip["domain"] = domain
            all_clips.extend(clips)
            print(f"    {wav.name} -> {len(clips)} clips")

    manifest = os.path.join(out_dir, "manifest.json")
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(all_clips, f, ensure_ascii=False, indent=2)

    total_min = sum(c["duration_sec"] for c in all_clips) / 60
    print(f"\n  Total clips: {len(all_clips)}  ({total_min:.1f} min)")
    return all_clips


def transcribe_test(clips: list, out_dir: str) -> list:
    """Run Whisper transcription on the test clips."""
    try:
        import torch
    except ImportError:
        exit_with_import_error("torch")

    try:
        import whisper
    except ImportError:
        exit_with_import_error("whisper", "openai-whisper")

    print("\n" + "=" * 55)
    print(f"STEP 3 - Transcribing with Whisper ({WHISPER_MODEL})")
    print("=" * 55)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    model = whisper.load_model(WHISPER_MODEL, device=device)

    results = []
    for clip in clips:
        if not os.path.exists(clip["path"]):
            continue

        result = model.transcribe(clip["path"], language="bn", word_timestamps=True, verbose=False)
        text = result["text"].strip()

        tags = detect_word_tags(result)
        cls = classify_code_mixing(tags)
        entry = {
            "clip_id": clip["clip_id"],
            "audio_path": clip["path"],
            "dialect": clip.get("dialect", "unknown"),
            "domain": clip.get("domain", "General"),
            "duration_sec": clip["duration_sec"],
            "transcript": text,
            "label": cls["label"],
            "cs_confidence": cls["cs_confidence"],
            "bn_ratio": cls["bn_ratio"],
            "en_ratio": cls["en_ratio"],
            "bn_word_count": cls["bn_word_count"],
            "en_word_count": cls["en_word_count"],
            "switch_count": cls["switch_count"],
            "en_words": cls["en_words"],
        }
        results.append(entry)

        tag = cls["label"]
        en_pct = int(cls["en_ratio"] * 100)
        icon = {
            "CS": f"[CS-{cls['cs_confidence']:<6} {en_pct:2d}%EN]",
            "BN_ONLY": "[BN_ONLY ]",
            "EN_ONLY": "[EN_ONLY ]",
            "NOISE": "[NOISE   ]",
        }[tag]
        print(f"  {icon} [{clip['dialect'][:10]:10s}] {text[:50]}")

    out_path = os.path.join(out_dir, "test_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


def quality_report(results: list, out_dir: str):
    """Print a compact quality report and save the rows as TSV."""
    print("\n" + "=" * 55)
    print("QUALITY REPORT")
    print("=" * 55)

    dialects = sorted(set(result["dialect"] for result in results))

    total_cs = sum(1 for result in results if result["label"] == "CS")
    total_cs_high = sum(
        1 for result in results if result["label"] == "CS" and result.get("cs_confidence") == "HIGH"
    )
    total_cs_medium = sum(
        1 for result in results if result["label"] == "CS" and result.get("cs_confidence") == "MEDIUM"
    )
    total_cs_low = sum(
        1 for result in results if result["label"] != "CS" and result.get("cs_confidence") == "LOW"
    )
    total_bn = sum(1 for result in results if result["label"] == "BN_ONLY")
    total_skip = sum(1 for result in results if result["label"] in ("EN_ONLY", "NOISE"))
    total = len(results)
    total_min = sum(result["duration_sec"] for result in results) / 60

    print(f"\n  Total clips processed : {total}")
    print(f"  Total duration        : {total_min:.1f} min")
    print(f"  Code-switched (CS)    : {total_cs}  ({total_cs/max(total, 1)*100:.0f}%)")
    print(f"    High confidence     : {total_cs_high}")
    print(f"    Medium confidence   : {total_cs_medium}")
    print(f"  Low-confidence mix    : {total_cs_low}")
    print(f"  Pure dialect (BN_ONLY): {total_bn}  ({total_bn/max(total, 1)*100:.0f}%)")
    print(f"  Skipped (EN/noise)    : {total_skip}")

    print("\n  Per-dialect breakdown:")
    print(
        f"  {'Dialect':<20} {'Total':>6} {'CS':>6} {'High':>6} {'BN_ONLY':>8} {'CS%':>6}  Sample English words"
    )
    print("  " + "-" * 75)

    for dialect in dialects:
        dialect_clips = [result for result in results if result["dialect"] == dialect]
        dialect_cs = [result for result in dialect_clips if result["label"] == "CS"]
        dialect_cs_high = [
            result for result in dialect_cs if result.get("cs_confidence") == "HIGH"
        ]
        dialect_bn = [result for result in dialect_clips if result["label"] == "BN_ONLY"]
        cs_pct = int(len(dialect_cs) / max(len(dialect_clips), 1) * 100)

        en_sample = []
        for result in dialect_cs[:3]:
            en_sample.extend(result["en_words"][:3])
        en_sample = list(dict.fromkeys(en_sample))[:6]
        en_str = ", ".join(en_sample) if en_sample else "-"

        print(
            f"  {dialect:<20} {len(dialect_clips):>6} {len(dialect_cs):>6} {len(dialect_cs_high):>6} "
            f"{len(dialect_bn):>8} {cs_pct:>5}%  {en_str}"
        )

    tsv_path = os.path.join(out_dir, "test_review.tsv")
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "clip_id",
                "dialect",
                "domain",
                "label",
                "cs_confidence",
                "en_ratio",
                "en_word_count",
                "switch_count",
                "transcript",
                "en_words",
                "audio_path",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result["clip_id"],
                    result["dialect"],
                    result.get("domain", "General"),
                    result["label"],
                    result.get("cs_confidence", "NONE"),
                    result["en_ratio"],
                    result.get("en_word_count", 0),
                    result.get("switch_count", 0),
                    result["transcript"],
                    " ".join(result.get("en_words", [])),
                    result["audio_path"],
                ]
            )

    print(f"\n  Full results saved -> {tsv_path}")
    print("  Open this file in Excel or Google Sheets to review clips.\n")


if __name__ == "__main__":
    print("\n" + "*" * 55)
    print("  BENGALI DIALECT CS DATASET - TEST RUN")
    print(f"  {VIDEOS_PER_BUCKET} videos per bucket  |  model: Whisper-{WHISPER_MODEL}")
    print("*" * 55)

    preflight_test_dependencies()
    reset_test_dir()

    audio_dir = os.path.join(TEST_DIR, "audio")
    download_test(TEST_BUCKETS, TEST_DIR, n=VIDEOS_PER_BUCKET)

    clips = segment_all_test(audio_dir, TEST_DIR, TEST_BUCKETS)
    if not clips:
        print("\n[ERROR] No clips found. Check yt-dlp, ffmpeg, and the search queries.")
        raise SystemExit(1)

    results = transcribe_test(clips, TEST_DIR)
    quality_report(results, TEST_DIR)

    print("=" * 55)
    print("TEST RUN COMPLETE")
    print("=" * 55)
    print("\nNext steps:")
    print("  1. Open test_run/test_review.tsv in Excel")
    print("  2. Listen to a few CS clips and check quality")
    print("  3. If quality is good, run the full pipeline:")
    print("       python 01_download_audio.py")
    print("       python 02_segment_audio.py")
    print("       python 03_auto_transcribe.py")
