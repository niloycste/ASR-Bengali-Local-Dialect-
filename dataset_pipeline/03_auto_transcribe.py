"""
Step 3: Auto-transcribe segments using Whisper
Produces rough draft transcripts — humans will correct these later.
Also detects language switches (code-switching detection).
"""

import os
import csv
import json
import shutil
import argparse
from pathlib import Path

from code_mixing_utils import classify_code_mixing, detect_word_tags
from pipeline_utils import exit_with_import_error, require_command_runs, require_commands

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
SEGMENTS_DIR  = str(SCRIPT_DIR / "segments")
TRANSCRIPTS_DIR = str(SCRIPT_DIR / "transcripts")
MANIFEST_PATH = str(SCRIPT_DIR / "segments" / "segments_manifest.json")

# Model size: "tiny", "base", "small", "medium", "large-v3"
# Use "large-v3" for best quality. Use "small" if GPU memory is limited.
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")

# "faster" uses faster-whisper/CTranslate2 and is the recommended backend for
# Colab H100/A100 inference. Set WHISPER_BACKEND=openai to use the original
# openai-whisper implementation.
WHISPER_BACKEND = os.environ.get("WHISPER_BACKEND", "faster").lower()
WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "1"))
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "float16")
SAVE_EVERY_CLIPS = int(os.environ.get("SAVE_EVERY_CLIPS", "100"))
DRIVE_BACKUP_DIR = os.environ.get(
    "DRIVE_BACKUP_DIR",
    "/content/drive/MyDrive/ASR_Transcripts_Backup",
)

# Force both Bengali and English recognition
# Whisper will still detect English words naturally
LANGUAGE = "bn"  # primary language is Bengali


def resolve_pipeline_path(path: str) -> str:
    """Resolve a manifest-relative path against dataset_pipeline/."""
    if os.path.isabs(path):
        return path
    return os.path.join(os.fspath(SCRIPT_DIR), path)


def pipeline_relpath(path: str) -> str:
    """Store paths relative to dataset_pipeline/ for portability."""
    return os.path.relpath(path, os.fspath(SCRIPT_DIR))


def backup_results_to_drive(output_dir: str, clip_count: int | str) -> None:
    """Copy current transcript outputs to Google Drive when Drive is mounted."""
    drive_root = "/content/drive/MyDrive"
    if not os.path.exists(drive_root):
        return

    os.makedirs(DRIVE_BACKUP_DIR, exist_ok=True)
    for filename in ("transcripts.json", "for_human_review.tsv", "code_switched_clips.tsv"):
        src = os.path.join(output_dir, filename)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(DRIVE_BACKUP_DIR, filename))
    print(f"  [BACKUP] Saved transcript checkpoint to Google Drive at clip {clip_count}")


def load_model():
    try:
        import torch
    except ImportError:
        exit_with_import_error("torch")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if WHISPER_BACKEND == "faster":
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("[WARNING] faster-whisper not installed. Falling back to openai-whisper.")
            print("          For Colab: pip install faster-whisper")
        else:
            compute_type = WHISPER_COMPUTE_TYPE if device == "cuda" else "int8"
            print(
                f"Loading faster-whisper {WHISPER_MODEL} on {device} "
                f"({compute_type}, beam={WHISPER_BEAM_SIZE})..."
            )
            model = WhisperModel(WHISPER_MODEL, device=device, compute_type=compute_type)
            return {"backend": "faster", "model": model}, device

    try:
        import whisper
    except ImportError:
        exit_with_import_error("whisper", "openai-whisper")

    print(f"Loading openai-whisper {WHISPER_MODEL} on {device}...")
    model = whisper.load_model(WHISPER_MODEL, device=device)
    return {"backend": "openai", "model": model}, device


def transcribe_clip(model, audio_np) -> dict:
    """
    Transcribe a single audio clip (numpy array or file path).

    Language is forced to Bengali (bn) so Whisper never switches to
    Telugu, Hindi, or other Indian scripts — output is always Bengali
    script + any English words the speaker actually said.

    suppress_tokens disables automatic language switching mid-clip.
    """
    if isinstance(model, dict) and model.get("backend") == "faster":
        segments_iter, _info = model["model"].transcribe(
            audio_np,
            language=LANGUAGE,
            task="transcribe",
            word_timestamps=True,
            beam_size=WHISPER_BEAM_SIZE,
            condition_on_previous_text=False,
            vad_filter=False,
        )
        segments_out = []
        texts = []
        for segment in segments_iter:
            words = []
            for word in segment.words or []:
                words.append(
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    }
                )
            texts.append(segment.text or "")
            segments_out.append(
                {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "words": words,
                    "no_speech_prob": segment.no_speech_prob,
                }
            )
        result = {"text": "".join(texts).strip(), "segments": segments_out}
    else:
        if isinstance(model, dict):
            model = model["model"]

        import imageio_ffmpeg
        from pydub import AudioSegment
        import numpy as np

        # If given a file path, load as numpy so Whisper doesn't call ffmpeg itself
        if isinstance(audio_np, str):
            AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
            audio = (AudioSegment.from_wav(audio_np)
                     .set_frame_rate(16000).set_channels(1).set_sample_width(2))
            audio_np = (np.frombuffer(audio.raw_data, dtype=np.int16)
                        .astype(np.float32) / 32768.0)

        result = model.transcribe(
            audio_np,
        language=LANGUAGE,          # force Bengali — never switch language
            word_timestamps=True,
            task="transcribe",
            verbose=False,
        condition_on_previous_text=False,  # prevents hallucination loops
            beam_size=WHISPER_BEAM_SIZE,
        )

    # ── Music / noise detection ────────────────────────────────────────────
    # Whisper sets no_speech_prob per segment (0.0 = definitely speech,
    # 1.0 = definitely not speech). Music clips get high values because
    # the model hears no human voice, yet still hallucinates words.
    # If the AVERAGE no_speech_prob across all segments is above the
    # threshold, treat the whole clip as noise and blank it.
    NO_SPEECH_THRESHOLD = 0.60
    segments = result.get("segments", [])
    avg_no_speech = None
    if segments:
        avg_no_speech = sum(s.get("no_speech_prob", 0.0) for s in segments) / len(segments)
        if avg_no_speech >= NO_SPEECH_THRESHOLD:
            result["text"] = ""
            result["segments"] = []
            result["avg_no_speech_prob"] = round(avg_no_speech, 4)
            return result  # will become NOISE in classify_code_mixing

    # ── Language filter ────────────────────────────────────────────────────
    # Keep only clips whose transcript is Bengali script (+ optional English).
    # Discard anything that is primarily another language.
    text = result.get("text", "")

    def _script_char_count(s, lo, hi):
        return sum(1 for c in s if lo <= c <= hi)

    bengali_chars    = _script_char_count(text, "\u0980", "\u09FF")
    devanagari_chars = _script_char_count(text, "\u0900", "\u097F")  # Hindi/Marathi/Nepali
    telugu_chars     = _script_char_count(text, "\u0C00", "\u0C7F")
    tamil_chars      = _script_char_count(text, "\u0B80", "\u0BFF")
    arabic_chars     = _script_char_count(text, "\u0600", "\u06FF")  # Urdu/Arabic
    latin_chars      = sum(1 for c in text if "a" <= c.lower() <= "z")
    other_indian     = devanagari_chars + telugu_chars + tamil_chars + arabic_chars

    # Blank the transcript if:
    # 1. No Bengali characters at all (hallucination or wrong language)
    # 2. Another Indian script dominates over Bengali (e.g. Hindi video)
    # 3. Pure Latin with no Bengali (English-only video)
    if bengali_chars == 0:
        result["text"] = ""
        result["segments"] = []
    elif other_indian > bengali_chars:
        # More Hindi/Telugu/etc. chars than Bengali → wrong language
        result["text"] = ""
        result["segments"] = []

    result["avg_no_speech_prob"] = None if avg_no_speech is None else round(avg_no_speech, 4)
    return result


def preflight_transcription_dependencies():
    """Ensure ffmpeg tools required by Whisper are available."""
    resolved = require_commands(
        [
            ("ffmpeg", ("ffmpeg",), "install ffmpeg and add it to PATH"),
            ("ffprobe", ("ffprobe",), "install ffmpeg and add it to PATH"),
        ],
        context="audio transcription",
    )
    require_command_runs("ffmpeg", [resolved["ffmpeg"], "-version"], context="audio transcription")
    require_command_runs("ffprobe", [resolved["ffprobe"], "-version"], context="audio transcription")


def load_existing_results(output_dir: str) -> list:
    """Load an existing transcript checkpoint so interrupted runs can resume."""
    json_path = os.path.join(output_dir, "transcripts.json")
    if not os.path.exists(json_path):
        return []

    try:
        with open(json_path, encoding="utf-8") as f:
            results = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARNING] Could not load existing checkpoint {json_path}: {exc}")
        print("          Starting a fresh transcript run.")
        return []

    if not isinstance(results, list):
        print(f"[WARNING] Existing checkpoint {json_path} is not a list. Starting fresh.")
        return []

    return results


def load_skipped_clip_ids(output_dir: str) -> set:
    """Load clip IDs already rejected as EN_ONLY or NOISE."""
    skipped_path = os.path.join(output_dir, "skipped_clips.json")
    if not os.path.exists(skipped_path):
        return set()

    try:
        with open(skipped_path, encoding="utf-8") as f:
            skipped = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[WARNING] Could not load skipped checkpoint {skipped_path}: {exc}")
        return set()

    if isinstance(skipped, list):
        return {str(clip_id) for clip_id in skipped if clip_id}
    if isinstance(skipped, dict):
        return {str(clip_id) for clip_id in skipped.keys() if clip_id}

    print(f"[WARNING] Skipped checkpoint {skipped_path} is not a list/dict. Ignoring it.")
    return set()


def save_skipped_clip_ids(skipped_clip_ids: set, output_dir: str) -> None:
    """Persist rejected clip IDs so reruns do not re-transcribe known rejects."""
    skipped_path = os.path.join(output_dir, "skipped_clips.json")
    with open(skipped_path, "w", encoding="utf-8") as f:
        json.dump(sorted(skipped_clip_ids), f, ensure_ascii=False, indent=2)


def dedupe_results_by_clip_id(results: list) -> list:
    """Keep one transcript per clip_id while preserving original order."""
    deduped = []
    seen_clip_ids = set()
    for row in results:
        clip_id = row.get("clip_id") if isinstance(row, dict) else None
        if not clip_id or clip_id in seen_clip_ids:
            continue
        deduped.append(row)
        seen_clip_ids.add(clip_id)
    return deduped


def find_resume_start_index(clips: list, known_clip_ids: set) -> int:
    """Return the manifest index after the last known saved/skipped clip."""
    last_known_index = -1
    for i, clip in enumerate(clips):
        if clip.get("clip_id") in known_clip_ids:
            last_known_index = i
    return last_known_index + 1


def transcribe_all(manifest_path: str = MANIFEST_PATH,
                   output_dir: str = TRANSCRIPTS_DIR,
                   resume_after_last_saved: bool = False):
    """Transcribe all segments and save results."""
    os.makedirs(output_dir, exist_ok=True)

    with open(manifest_path, encoding="utf-8") as f:
        clips = json.load(f)

    results = dedupe_results_by_clip_id(load_existing_results(output_dir))
    completed_clip_ids = {r["clip_id"] for r in results if isinstance(r, dict) and r.get("clip_id")}
    skipped_clip_ids = load_skipped_clip_ids(output_dir)
    known_clip_ids = completed_clip_ids | skipped_clip_ids
    completed_before_run = len(completed_clip_ids)
    skipped_before_run = len(skipped_clip_ids)
    if completed_before_run:
        print(f"Resuming from checkpoint: {completed_before_run} clips already saved.")
        save_results(results, output_dir)
    if skipped_before_run:
        print(f"Resuming with skipped checkpoint: {skipped_before_run} rejected clips already known.")

    start_index = 0
    if resume_after_last_saved:
        start_index = find_resume_start_index(clips, known_clip_ids)
        remaining = max(len(clips) - start_index, 0)
        print(
            f"Fast resume enabled: starting at manifest row {start_index + 1} "
            f"of {len(clips)} ({remaining} clips left to scan)."
        )

    model, device = load_model()
    cs_clips = [r for r in results if isinstance(r, dict) and r.get("is_code_switched")]
    new_results_since_save = 0
    new_skips_since_save = 0

    print(f"\nTranscribing {len(clips)} clips...\n")

    for i, clip in enumerate(clips):
        if i < start_index:
            continue

        if clip["clip_id"] in completed_clip_ids:
            print(f"[{i+1}/{len(clips)}] {clip['clip_id']} ... [RESUME-SKIP]")
            continue
        if clip["clip_id"] in skipped_clip_ids:
            print(f"[{i+1}/{len(clips)}] {clip['clip_id']} ... [SKIP-KNOWN]")
            continue

        audio_path = resolve_pipeline_path(clip["path"])
        if not os.path.exists(audio_path):
            continue

        print(f"[{i+1}/{len(clips)}] {clip['clip_id']}", end=" ... ")

        result    = transcribe_clip(model, audio_path)
        word_tags = detect_word_tags(result)
        cls       = classify_code_mixing(word_tags)

        # Auto-reject pure English and noise — no manual review needed for these
        if cls["label"] in ("EN_ONLY", "NOISE"):
            print(f"[SKIP-{cls['label']}] {result['text'].strip()[:50]}")
            skipped_clip_ids.add(clip["clip_id"])
            new_skips_since_save += 1
            if new_skips_since_save >= SAVE_EVERY_CLIPS:
                save_skipped_clip_ids(skipped_clip_ids, output_dir)
                new_skips_since_save = 0
            continue

        transcript_entry = {
            "clip_id":          clip["clip_id"],
            "audio_path":       pipeline_relpath(audio_path),
            "duration_sec":     clip["duration_sec"],
            "transcript":       result["text"].strip(),
            "word_tags":        word_tags,
            "clip_label":       cls["label"],        # "CS" or "BN_ONLY"
            "is_code_switched": cls["is_code_switched"],
            "bn_ratio":         cls["bn_ratio"],
            "en_ratio":         cls["en_ratio"],
            "en_word_count":    cls["en_word_count"],
            "bn_word_count":    cls["bn_word_count"],
            "switch_count":     cls["switch_count"],
            "cs_confidence":    cls["cs_confidence"],
            "avg_no_speech_prob": result.get("avg_no_speech_prob"),
            "needs_human_review": True,
            "human_transcript": "",
            "dialect":          clip.get("dialect", "unknown"),
            "domain":           clip.get("domain", "General"),
            "source_id":        clip.get("source_id"),
            "source_path":      clip.get("source_path"),
            "source_bucket":    clip.get("source_bucket"),
            "source_root":      clip.get("source_root"),
        }

        results.append(transcript_entry)
        completed_clip_ids.add(clip["clip_id"])
        new_results_since_save += 1
        if cls["label"] == "CS":
            cs_clips.append(transcript_entry)
            en_pct = int(cls["en_ratio"] * 100)
            print(f"[CS-{cls['cs_confidence']:<6} {en_pct:2d}%EN] {result['text'].strip()[:55]}")
        else:
            print(f"[BN_ONLY]    {result['text'].strip()[:55]}")

        # Save incrementally by newly accepted transcript count.
        if new_results_since_save >= SAVE_EVERY_CLIPS:
            save_results(results, output_dir)
            save_skipped_clip_ids(skipped_clip_ids, output_dir)
            backup_results_to_drive(output_dir, len(results))
            new_results_since_save = 0

    save_results(results, output_dir)
    save_skipped_clip_ids(skipped_clip_ids, output_dir)
    backup_results_to_drive(output_dir, "final")

    print(f"\nTotal clips transcribed: {len(results)}")
    print(f"New clips transcribed this run: {len(results) - completed_before_run}")
    print(f"Rejected clips remembered: {len(skipped_clip_ids)}")
    print(f"Code-switched clips: {len(cs_clips)}")
    print(f"Results saved to: {output_dir}")
    return results, cs_clips


def save_results(results: list, output_dir: str):
    """Save transcription results to JSON and a simple TSV for annotation."""
    # Full JSON
    json_path = os.path.join(output_dir, "transcripts.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # TSV for easy review in Excel/Google Sheets
    tsv_path = os.path.join(output_dir, "for_human_review.tsv")
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
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
        )
        for r in results:
            writer.writerow(
                [
                    r["clip_id"],
                    r["audio_path"],
                    r["duration_sec"],
                    r["transcript"],
                    r["is_code_switched"],
                    r["dialect"],
                    r.get("domain", "General"),
                    r.get("bn_ratio", 0.0),
                    r.get("en_ratio", 0.0),
                    r.get("cs_confidence", "NONE"),
                    r.get("switch_count", 0),
                    r.get("human_transcript", ""),
                    "",
                ]
            )

    # Separate TSV with only code-switched clips (priority for annotation)
    cs_tsv_path = os.path.join(output_dir, "code_switched_clips.tsv")
    with open(cs_tsv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
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
        )
        for r in results:
            if r["is_code_switched"]:
                writer.writerow(
                    [
                        r["clip_id"],
                        r["audio_path"],
                        r["duration_sec"],
                        r["transcript"],
                        r["dialect"],
                        r.get("domain", "General"),
                        r.get("bn_ratio", 0.0),
                        r.get("en_ratio", 0.0),
                        r.get("cs_confidence", "NONE"),
                        r.get("switch_count", 0),
                        r.get("human_transcript", ""),
                        "",
                    ]
                )
    print(f"  TSV files saved for human review.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=MANIFEST_PATH,
        help="Path to the segment manifest JSON to transcribe.",
    )
    parser.add_argument(
        "--output-dir",
        default=TRANSCRIPTS_DIR,
        help="Directory where transcript JSON/TSV outputs will be written.",
    )
    parser.add_argument(
        "--resume-after-last-saved",
        action="store_true",
        help="Start after the latest already saved/skipped manifest row.",
    )
    args = parser.parse_args()

    preflight_transcription_dependencies()
    transcribe_all(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        resume_after_last_saved=args.resume_after_last_saved,
    )
