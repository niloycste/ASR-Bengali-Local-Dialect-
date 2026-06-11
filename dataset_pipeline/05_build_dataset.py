"""
Step 5: Build the final dataset from reviewed transcripts.

The key rule here is source isolation:
clips from the same original audio source must stay in the same split.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline_utils import shuffled
from text_normalization import normalize_pronunciation_variants

SCRIPT_DIR = Path(__file__).resolve().parent
REVIEWED_PATH = str(SCRIPT_DIR / "transcripts" / "transcripts_reviewed.json")
NOISE_LEXICON_PATH = SCRIPT_DIR / "lexicons" / "noise_tokens.txt"
DATASET_DIR = str(SCRIPT_DIR / "final_dataset")

MANIFEST_COLUMNS = [
    "clip_id",
    "source_id",
    "audio_path",
    "transcript",
    "normalized_transcript",
    "dialect",
    "domain",
    "is_code_switched",
    "bn_ratio",
    "en_ratio",
    "duration_sec",
    "word_tags",
]


TOKEN_RE = re.compile(r"[\u0900-\u09ffA-Za-z][\u0900-\u09ffA-Za-z'\-]*", re.UNICODE)


def clip_id_prefix(clip_id: str) -> str:
    """Return the source prefix portion from a clip id like stem_0000."""
    prefix, sep, tail = clip_id.rpartition("_")
    if sep and tail.isdigit():
        return prefix
    return clip_id


def resolve_pipeline_path(path: str) -> str:
    """Resolve an audio path so it works on ANY machine.

    Transcripts may carry paths baked on a different machine — e.g. Windows
    absolute paths (C:\\...\\segments\\augmented\\x.wav) or Colab paths
    (/content/.../segments/Sylhet/x.wav). Rather than trust those, we re-root
    every path onto the LOCAL segments/ tree, so the same transcripts_reviewed.json
    works on Windows, Colab, or anywhere else.
    """
    raw = str(path)
    norm = raw.replace("\\", "/")

    # 1. Absolute path that actually exists on THIS machine — use as-is.
    if os.path.isabs(raw) and os.path.exists(raw):
        return raw

    # 2. Re-root anything containing ".../segments/..." onto the local tree.
    #    Handles foreign absolute paths (Windows C:\ or Colab /content) by
    #    keeping only the part from "segments/" onward.
    marker = "segments/"
    idx = norm.rfind(marker)
    if idx != -1:
        tail = norm[idx + len(marker):]            # e.g. "augmented/x.wav", "Sylhet/x.wav"
        candidate = os.path.join(os.fspath(SCRIPT_DIR), "segments", tail)
        if os.path.exists(candidate):
            return candidate

    # 3. Plain relative path joined to the pipeline dir.
    direct_path = os.path.join(os.fspath(SCRIPT_DIR), raw)
    if os.path.exists(direct_path):
        return direct_path

    # 4. Relative path that omits the leading "segments/" (e.g. "Barishal/clip.wav").
    if not norm.startswith("segments/"):
        segments_path = os.path.join(os.fspath(SCRIPT_DIR), "segments", raw)
        if os.path.exists(segments_path):
            return segments_path

    # 5. Last resort: locate by basename under segments/{,augmented,synthetic}.
    base = os.path.basename(norm)
    for sub in ("", "augmented", "synthetic"):
        cand = os.path.join(os.fspath(SCRIPT_DIR), "segments", sub, base)
        if os.path.exists(cand):
            return cand

    return direct_path


def infer_source_id(clip: dict) -> str:
    """Use explicit source_id when present, otherwise derive it from clip_id."""
    return clip.get("source_id") or clip_id_prefix(clip["clip_id"])


def group_clips_by_source(clips: list[dict]) -> list[dict]:
    """Aggregate clips by original raw-audio source."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for clip in clips:
        grouped[infer_source_id(clip)].append(clip)

    groups = []
    for source_id, source_clips in grouped.items():
        dialect_counts = defaultdict(int)
        for clip in source_clips:
            dialect_counts[clip.get("dialect", "unknown")] += 1
        dialect = max(dialect_counts.items(), key=lambda item: item[1])[0]
        groups.append(
            {
                "source_id": source_id,
                "dialect": dialect,
                "clips": source_clips,
            }
        )
    return groups


def random_group_split(groups: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Fallback source-level split when stratification is not possible."""
    groups = shuffled(groups, seed=42)
    total = len(groups)

    if total < 3:
        return groups, [], []

    train_groups, temp_groups = train_test_split(groups, test_size=0.2, random_state=42)
    if len(temp_groups) < 2:
        return train_groups, temp_groups, []

    dev_groups, test_groups = train_test_split(temp_groups, test_size=0.5, random_state=42)
    return train_groups, dev_groups, test_groups


def expand_groups(groups: list[dict]) -> list[dict]:
    """Flatten grouped clips after splitting."""
    clips: list[dict] = []
    for group in groups:
        clips.extend(group["clips"])
    return clips


def keep_pinned_for_train(
    pinned: list[dict],
    dev_groups: list[dict],
    test_groups: list[dict],
) -> list[dict]:
    """
    Filter pinned (augmented/synthetic) clips to avoid train/eval leakage.

    Augmented clips inherit the source_id of the original clip they were derived
    from. If that source was assigned to dev/test, force-adding the augmented
    copy to train would leak the same raw source across splits. We therefore keep
    a pinned clip only when its source did NOT land in dev/test. Synthetic clips
    (source_id == "synthetic") never match an eval source, so they stay in train.
    """
    if not pinned:
        return []

    eval_sources = {group["source_id"] for group in dev_groups}
    eval_sources |= {group["source_id"] for group in test_groups}

    kept = [clip for clip in pinned if infer_source_id(clip) not in eval_sources]
    dropped = len(pinned) - len(kept)
    if dropped:
        print(
            f"[LEAKAGE GUARD] Dropped {dropped} pinned (augmented/synthetic) clips "
            f"whose source was assigned to dev/test — prevents train/eval leakage."
        )
    print(f"[INFO] {len(kept)} pinned clips kept in train.")
    return kept


def split_clips(clips: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split train/dev/test by source, not by clip.

    This prevents the same raw source from leaking across train/dev/test.

    Pinning rules (applied before stratified split):
      - clip["split_pin"] == "train"  → always goes to train
        (augmented clips and synthetic clips are pinned this way)
      - clip source_id == "synthetic" → always goes to train
    """
    # Separate out pinned-to-train clips before splitting
    pinned_train = [c for c in clips
                    if c.get("split_pin") == "train"
                    or c.get("source_id") == "synthetic"
                    or c.get("synthetic", False)]
    free_clips   = [c for c in clips
                    if c not in pinned_train]

    if pinned_train:
        print(f"[INFO] {len(pinned_train)} clips pinned to train "
              f"(augmented/synthetic) — excluded from split sampling.")

    groups = group_clips_by_source(free_clips)
    dialects = [group["dialect"] for group in groups]
    unique_dialects = set(dialects)

    print(f"[INFO] Found {len(groups)} distinct source groups.")

    if len(groups) < 10 or len(unique_dialects) < 2:
        print("[INFO] Too few source groups for reliable stratification. Using random source split.")
        train_groups, dev_groups, test_groups = random_group_split(groups)
    else:
        try:
            train_groups, temp_groups = train_test_split(
                groups, test_size=0.2, random_state=42, stratify=dialects
            )
            temp_dialects = [group["dialect"] for group in temp_groups]
            dev_groups, test_groups = train_test_split(
                temp_groups, test_size=0.5, random_state=42, stratify=temp_dialects
            )
            print("[INFO] Using dialect-stratified source split.")
        except ValueError as exc:
            print(f"[WARNING] Stratified source split not possible: {exc}")
            print("          Falling back to random source split.")
            train_groups, dev_groups, test_groups = random_group_split(groups)

    # Add pinned clips to train, but only those whose source did not land in
    # dev/test (prevents augmented clips leaking a source across splits).
    pinned_kept = keep_pinned_for_train(pinned_train, dev_groups, test_groups)
    return (expand_groups(train_groups) + pinned_kept,
            expand_groups(dev_groups),
            expand_groups(test_groups))


def materialize_split(split_clips: list[dict], audio_dir: str, copy_audio: bool = True):
    """Build manifest rows for one split, optionally copying audio.

    copy_audio=True  → copy each wav into final_dataset/<split>/audio/ (self-contained,
                       but duplicates ~33 GB of audio).
    copy_audio=False → leave audio in place and point the manifest at the source file
                       (saves ~33 GB; requires segments/ to stay where it is).
    """
    rows = []
    kept_clips = []
    missing_audio = 0

    for clip in split_clips:
        src = resolve_pipeline_path(clip["audio_path"])
        if not os.path.exists(src):
            missing_audio += 1
            continue

        if copy_audio:
            audio_out = os.path.join(audio_dir, Path(src).name)
            shutil.copy2(src, audio_out)
        else:
            audio_out = src

        row = {
            "clip_id": clip["clip_id"],
            "source_id": infer_source_id(clip),
            "audio_path": audio_out,
            "transcript": clip.get("human_transcript", ""),
            "normalized_transcript": normalize_pronunciation_variants(
                clip.get("human_transcript", "")
            ),
            "dialect": clip.get("dialect", "unknown"),
            "domain": clip.get("domain", "General"),
            "is_code_switched": clip.get("is_code_switched", False),
            "bn_ratio": clip.get("bn_ratio", 0.0),
            "en_ratio": clip.get("en_ratio", 0.0),
            "duration_sec": clip.get("duration_sec", 0),
            "word_tags": json.dumps(clip.get("word_tags", []), ensure_ascii=False),
        }
        rows.append(row)

        kept_clip = dict(clip)
        kept_clip["audio_path"] = audio_out
        kept_clip["source_id"] = infer_source_id(clip)
        kept_clips.append(kept_clip)

    return rows, kept_clips, missing_audio


REPLACEMENT_CHAR = "�"


def is_low_quality(text: str) -> bool:
    """
    Flag transcripts unsuitable for an ASR benchmark.

    Drops:
      - empty transcripts
      - text containing the Unicode replacement char (U+FFFD '�'), which means
        bytes were lost during decoding — the text is unrecoverable
      - text containing letters from a foreign script (anything that is an
        alphabetic character but is neither Bengali (U+0980–U+09FF) nor ASCII
        Latin) — e.g. Devanagari, Arabic, CJK contamination
    """
    if not text or not text.strip():
        return True
    if REPLACEMENT_CHAR in text:
        return True
    for ch in text:
        code = ord(ch)
        if code < 128:                       # ASCII (English CS words, digits)
            continue
        if "ঀ" <= ch <= "৿":       # Bengali script
            continue
        if ch.isspace():
            continue
        if ch.isalpha():                     # alphabetic but foreign script
            return True
    return False


def load_noise_tokens() -> set[str]:
    """Load known hallucination/noise tokens."""
    if not NOISE_LEXICON_PATH.exists():
        return set()
    with open(NOISE_LEXICON_PATH, encoding="utf-8") as f:
        return {
            line.strip().casefold() for line in f
            if line.strip() and not line.startswith("#")
        }


def build_dataset(
    reviewed_path: str = REVIEWED_PATH,
    output_dir: str = DATASET_DIR,
    cs_only: bool = False,
    filter_noise: bool = False,
    filter_quality: bool = True,
    copy_audio: bool = True,
):
    """Build train/dev/test manifests and audio folders."""
    with open(reviewed_path, encoding="utf-8") as f:
        clips = json.load(f)

    for clip in clips:
        if not clip.get("human_transcript", "").strip():
            clip["human_transcript"] = clip.get("transcript", "").strip()
        clip["source_id"] = infer_source_id(clip)

    clips = [clip for clip in clips if clip.get("human_transcript", "").strip()]

    if filter_quality:
        before = len(clips)
        clips = [c for c in clips if not is_low_quality(c.get("human_transcript", ""))]
        print(
            f"[QUALITY] Dropped {before - len(clips)} clips with corrupted (�) or "
            f"foreign-script transcripts. ({len(clips)} remain)"
        )

    if filter_noise:
        noise_set = load_noise_tokens()
        if noise_set:
            clean_clips = []
            dropped = 0
            for clip in clips:
                text = clip.get("human_transcript", "")
                tokens = [m.group(0).casefold() for m in TOKEN_RE.finditer(text)]
                if any(t in noise_set for t in tokens):
                    dropped += 1
                else:
                    clean_clips.append(clip)
            clips = clean_clips
            print(f"[CLEANUP] Dropped {dropped} clips containing known noise tokens.")

    if cs_only:
        clips = [clip for clip in clips if clip.get("is_code_switched")]
        print(f"Code-switched clips only: {len(clips)}")
    else:
        print(f"Total annotated clips: {len(clips)}")

    if not clips:
        print("No clips to process.")
        return

    for split in ["train", "dev", "test"]:
        os.makedirs(os.path.join(output_dir, split, "audio"), exist_ok=True)

    train_clips, dev_clips, test_clips = split_clips(clips)
    planned_splits = {"train": train_clips, "dev": dev_clips, "test": test_clips}
    materialized_splits = {}

    for split_name, split_rows in planned_splits.items():
        audio_dir = os.path.join(output_dir, split_name, "audio")
        rows, kept_clips, missing_audio = materialize_split(split_rows, audio_dir, copy_audio=copy_audio)

        df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
        csv_path = os.path.join(output_dir, split_name, "manifest.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")

        total_hrs = df["duration_sec"].sum() / 3600 if not df.empty else 0.0
        print(f"  {split_name}: {len(df)} clips, {total_hrs:.2f} hrs -> {csv_path}")
        if missing_audio:
            print(f"    [WARNING] Skipped {missing_audio} clips with missing audio files.")

        materialized_splits[split_name] = kept_clips

    print_stats(materialized_splits)


def print_stats(splits: dict):
    """Print dataset statistics for the clips actually written to disk."""
    print("\n" + "=" * 50)
    print("DATASET STATISTICS")
    print("=" * 50)

    all_clips = []
    for name, clips in splits.items():
        all_clips.extend(clips)
        total_sec = sum(c.get("duration_sec", 0) for c in clips)
        cs_count = sum(1 for c in clips if c.get("is_code_switched"))
        source_count = len({infer_source_id(c) for c in clips})
        print(f"\n{name.upper()}:")
        print(f"  Clips:           {len(clips)}")
        print(f"  Sources:         {source_count}")
        print(f"  Duration:        {total_sec/3600:.2f} hrs")
        print(f"  Code-switched:   {cs_count} ({cs_count/max(len(clips), 1)*100:.1f}%)")

        dialect_counts: dict[str, int] = {}
        for clip in clips:
            dialect = clip.get("dialect", "unknown")
            dialect_counts[dialect] = dialect_counts.get(dialect, 0) + 1
        for dialect, count in sorted(dialect_counts.items()):
            print(f"    {dialect}: {count} clips")

    total_sec = sum(c.get("duration_sec", 0) for c in all_clips)
    total_sources = len({infer_source_id(c) for c in all_clips})
    print(f"\nTOTAL: {len(all_clips)} clips, {total_sources} sources, {total_sec/3600:.2f} hours")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cs-only", action="store_true", help="Only include code-switched clips")
    parser.add_argument("--filter-noise", action="store_true", help="Drop clips containing words from lexicons/noise_tokens.txt")
    parser.add_argument("--keep-low-quality", action="store_true",
                        help="Disable the quality filter (keep transcripts with � or foreign-script contamination)")
    parser.add_argument("--no-copy", action="store_true",
                        help="Don't copy audio into final_dataset/; point manifests at the source files (saves ~33 GB)")
    args = parser.parse_args()

    build_dataset(
        cs_only=args.cs_only,
        filter_noise=args.filter_noise,
        filter_quality=not args.keep_low_quality,
        copy_audio=not args.no_copy,
    )
