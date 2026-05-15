"""
Data Augmentation for Bengali CS ASR.

WHEN TO RUN:
  - ONLY if your CS clips are below ~5,000 after Step 3 transcription
  - Run AFTER 04b_auto_annotate.py and BEFORE 05_build_dataset.py
  - Do NOT run if you already have 10,000+ CS clips — you don't need it

WHEN NOT TO RUN:
  - If your CS data is sufficient (>10k clips) — augmentation won't help much
  - Do NOT run on test set — augmentation is TRAINING DATA ONLY
  - Do NOT run before transcription is done

What this does:
  Three augmentation techniques applied to TRAINING clips only:
  1. Speed perturbation (0.9x and 1.1x) — most important for ASR
  2. Additive noise (Gaussian, low SNR) — robustness
  3. Pitch shift (±2 semitones) — dialect variation simulation

  Each augmentation produces a new clip with a new clip_id.
  The transcript is copied unchanged (augmentation is audio-only).
  Augmented clips are added to transcripts_reviewed.json.

Usage:
    python augment_data.py                   # augment CS clips only (recommended)
    python augment_data.py --all             # augment all clips
    python augment_data.py --factor 2        # produce 2x augmented copies per clip
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

SCRIPT_DIR      = Path(__file__).resolve().parent
REVIEWED_PATH   = SCRIPT_DIR / "transcripts" / "transcripts_reviewed.json"
AUG_AUDIO_DIR   = SCRIPT_DIR / "segments" / "augmented"
OUTPUT_PATH     = SCRIPT_DIR / "transcripts" / "transcripts_reviewed.json"  # overwrite in-place

SAMPLE_RATE = 16000
SPEEDS      = [0.9, 1.1]         # speed perturbation factors
NOISE_SNR   = 20                  # dB — higher = less noise
PITCH_STEPS = [-2, 2]             # semitones


def load_audio(path: str):
    """Load a wav file as numpy float32 array."""
    import numpy as np
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr
    except Exception:
        try:
            import librosa
            audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"Cannot load {path}: {e}")


def save_audio(audio, sr: int, path: str):
    """Save float32 numpy array as wav."""
    import soundfile as sf
    import numpy as np
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Clip to [-1, 1] to avoid clipping artifacts
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, audio, sr, subtype="PCM_16")


def speed_perturb(audio, sr: int, factor: float):
    """Change speed without changing pitch (time-stretch)."""
    try:
        import librosa
        return librosa.effects.time_stretch(audio, rate=factor), sr
    except ImportError:
        # Fallback: simple resampling (changes pitch too, still useful)
        import numpy as np
        new_len = int(len(audio) / factor)
        indices = np.linspace(0, len(audio) - 1, new_len).astype(int)
        return audio[indices], sr


def add_noise(audio, snr_db: float = NOISE_SNR):
    """Add Gaussian white noise at specified SNR."""
    import numpy as np
    signal_power = np.mean(audio ** 2)
    if signal_power < 1e-10:
        return audio
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)
    return audio + noise


def pitch_shift(audio, sr: int, n_steps: float):
    """Shift pitch by n_steps semitones without changing speed."""
    try:
        import librosa
        return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    except ImportError:
        # Fallback: return original if librosa unavailable
        return audio


def augment_clip(clip: dict, aug_type: str, param) -> dict | None:
    """
    Apply one augmentation to a clip. Returns a new clip dict or None on error.
    aug_type: 'speed' | 'noise' | 'pitch'
    """
    # Resolve relative paths the same way 05_build_dataset.py does
    raw_path = clip["audio_path"]
    src_path = raw_path if os.path.isabs(raw_path) else str(SCRIPT_DIR / raw_path)
    if not os.path.exists(src_path):
        return None

    try:
        audio, sr = load_audio(src_path)
    except Exception as e:
        print(f"    [WARN] Cannot load {src_path}: {e}")
        return None

    clip_id = clip["clip_id"]

    if aug_type == "speed":
        aug_audio, sr = speed_perturb(audio, sr, param)
        suffix = f"sp{str(param).replace('.', '')}"
    elif aug_type == "noise":
        aug_audio = add_noise(audio, snr_db=param)
        suffix = f"noise{int(param)}"
    elif aug_type == "pitch":
        aug_audio = pitch_shift(audio, sr, param)
        suffix = f"pitch{'p' if param > 0 else 'n'}{abs(int(param))}"
    else:
        return None

    # Duration check — skip if augmented clip is too short or too long
    duration = len(aug_audio) / sr
    if duration < 1.0 or duration > 20.0:
        return None

    new_clip_id  = f"{clip_id}_{suffix}"
    out_filename = f"{new_clip_id}.wav"
    out_path     = str(AUG_AUDIO_DIR / out_filename)

    try:
        save_audio(aug_audio, sr, out_path)
    except Exception as e:
        print(f"    [WARN] Cannot save {out_path}: {e}")
        return None

    new_clip = dict(clip)
    new_clip["clip_id"]      = new_clip_id
    new_clip["audio_path"]   = out_path
    new_clip["duration_sec"] = round(duration, 3)
    new_clip["augmented"]    = True
    new_clip["aug_type"]     = aug_type
    new_clip["aug_param"]    = param
    # CRITICAL: keep the same source_id as the original so 05_build_dataset.py
    # puts this augmented clip in the same split as its parent. Without this,
    # an augmented variant can leak into dev/test.
    new_clip["source_id"]    = clip.get("source_id") or clip["clip_id"]
    new_clip["split_pin"]    = "train"   # extra safety flag
    return new_clip


def run_augmentation(augment_all: bool = False, factor: int = 1, min_dialect: int = 0):
    if not REVIEWED_PATH.exists():
        print(f"[ERROR] {REVIEWED_PATH} not found.")
        print("        Run 04b_auto_annotate.py first.")
        raise SystemExit(1)

    with open(REVIEWED_PATH, encoding="utf-8") as f:
        clips = json.load(f)

    # Only augment ORIGINAL clips (not previously augmented ones)
    original_clips = [c for c in clips if not c.get("augmented", False)]

    from collections import Counter
    dialect_counts = Counter(c.get("dialect", "unknown") for c in original_clips)

    if augment_all:
        to_augment = original_clips
        print(f"Augmenting ALL {len(to_augment)} original clips...")
    else:
        to_augment = []
        for c in original_clips:
            is_cs = c.get("is_code_switched", False)
            dialect = c.get("dialect", "unknown")
            if is_cs or dialect_counts[dialect] < min_dialect:
                to_augment.append(c)
        print(f"Augmenting CS clips + dialects with < {min_dialect} clips: {len(to_augment)} clips")
        print("  (Use --all to augment all clips)")

    if not to_augment:
        print("[WARN] No clips to augment.")
        return

    AUG_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    augmentations = []
    # Speed perturbation — highest priority, apply to all selected clips
    for speed in SPEEDS:
        augmentations.append(("speed", speed))
    # Noise — apply to all
    augmentations.append(("noise", NOISE_SNR))
    # Pitch — apply to all
    for steps in PITCH_STEPS:
        augmentations.append(("pitch", steps))

    new_clips   = []
    total_tried = 0
    total_saved = 0

    for i, clip in enumerate(to_augment):
        print(f"  [{i+1}/{len(to_augment)}] {clip['clip_id']}")

        # Apply `factor` random augmentations per clip
        selected_augs = random.sample(augmentations, min(factor, len(augmentations)))

        for aug_type, param in selected_augs:
            total_tried += 1
            new_clip = augment_clip(clip, aug_type, param)
            if new_clip:
                new_clips.append(new_clip)
                total_saved += 1

    # Merge into existing clips list (original + new)
    all_clips = clips + new_clips

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_clips, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Augmentation complete")
    print(f"  Original clips   : {len(original_clips)}")
    print(f"  New aug clips    : {total_saved} / {total_tried} attempted")
    print(f"  Total clips now  : {len(all_clips)}")
    print(f"  Output           : {OUTPUT_PATH}")
    print(f"\nNext: python 05_build_dataset.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",    action="store_true",
                        help="Augment all clips, not just CS clips")
    parser.add_argument("--factor", type=int, default=1,
                        help="Number of augmentation variants per clip (default: 1)")
    parser.add_argument("--min-dialect", type=int, default=0,
                        help="Augment all clips for dialects with fewer than this many clips (e.g. 3000)")
    args = parser.parse_args()
    run_augmentation(augment_all=args.all, factor=args.factor, min_dialect=args.min_dialect)
