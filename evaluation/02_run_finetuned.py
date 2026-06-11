"""
Evaluation Step 2: Run fine-tuned checkpoint-backed models on the test set.

Supported families:
  - Whisper fine-tunes
  - wav2vec2 / WavLM CTC fine-tunes

LM shallow-fusion variants are produced by 09_lm_shallow_fusion.py and are not
re-run here as checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fine_tuning.config import DATASET_DIR, FINETUNED, MODELS_DIR, RESULTS_DIR

SAMPLE_RATE = 16_000
LEGACY_MODEL_DIR_ALIASES = {
    "ft_whisper_all": "ft_all",
    "ft_whisper_bn_only": "ft_bn_only",
    "ft_whisper_cs_only": "ft_cs_only",
}


def checkpoint_model_keys() -> list[str]:
    """Return finetuned keys backed by actual model checkpoints."""
    return [key for key in FINETUNED if not key.endswith("_lm")]


def model_family(model_key: str) -> str:
    """Classify a finetuned model key by architecture family."""
    if model_key.endswith("_lm"):
        return "lm"
    if model_key.startswith("ft_whisper_"):
        return "whisper"
    if model_key.startswith("ft_wav2vec2_") or model_key.startswith("ft_wavlm_"):
        return "ctc"
    return "unknown"


def resolve_model_dir(model_key: str) -> Path:
    """Find the checkpoint directory, including backward-compatible aliases."""
    candidates = [MODELS_DIR / model_key]
    legacy = LEGACY_MODEL_DIR_ALIASES.get(model_key)
    if legacy:
        candidates.append(MODELS_DIR / legacy)

    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def load_test_manifest() -> list[dict]:
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pip install pandas")
        raise SystemExit(1)

    csv_path = DATASET_DIR / "test" / "manifest.csv"
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found. Run 05_build_dataset.py first.")
        raise SystemExit(1)

    df = pd.read_csv(csv_path, encoding="utf-8")
    rows = df.to_dict("records")
    rows = [row for row in rows if Path(str(row["audio_path"])).exists()]
    print(f"Test set: {len(rows)} clips")
    return rows


def load_audio_np(audio_path: str):
    import numpy as np
    from pydub import AudioSegment

    audio = (
        AudioSegment.from_file(audio_path)
        .set_frame_rate(SAMPLE_RATE)
        .set_channels(1)
        .set_sample_width(2)
    )
    return np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32) / 32768.0


def prediction_record(clip: dict, pred_text: str) -> dict:
    """Build a prediction row with shared metadata."""
    return {
        "clip_id": clip["clip_id"],
        "audio_path": clip["audio_path"],
        "reference": str(clip.get("transcript", "")).strip(),
        "hypothesis": pred_text.strip(),
        "dialect": clip.get("dialect", "unknown"),
        "domain": clip.get("domain", "General"),
        "is_code_switched": bool(clip.get("is_code_switched", False)),
        "bn_ratio": float(clip.get("bn_ratio", 0.0)),
        "en_ratio": float(clip.get("en_ratio", 0.0)),
        "duration_sec": float(clip.get("duration_sec", 0.0)),
    }


def run_whisper(model_dir: Path, clips: list[dict], device: str) -> list[dict]:
    """Evaluate a Whisper seq2seq checkpoint."""
    try:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError:
        print("[ERROR] pip install transformers torch")
        raise SystemExit(1)

    processor = WhisperProcessor.from_pretrained(str(model_dir))
    model = WhisperForConditionalGeneration.from_pretrained(str(model_dir)).to(device)
    model.eval()

    # Force Bengali transcription via generation_config. The older
    # forced_decoder_ids API is deprecated in transformers v5 (it can be ignored
    # or raise a mutual-exclusivity error with language/task).
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = "bengali"
    model.generation_config.task = "transcribe"

    predictions = []
    for i, clip in enumerate(clips):
        audio_np = load_audio_np(clip["audio_path"])
        inputs = processor(
            audio_np,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).input_features.to(device=device, dtype=model.dtype)

        with torch.no_grad():
            ids = model.generate(inputs)

        pred_text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        predictions.append(prediction_record(clip, pred_text))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(clips)}] done")

    return predictions


def run_ctc(model_dir: Path, clips: list[dict], device: str) -> list[dict]:
    """Evaluate a CTC checkpoint saved by 03_finetune_wav2vec2.py."""
    try:
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    except ImportError:
        print("[ERROR] pip install transformers torch")
        raise SystemExit(1)

    processor = Wav2Vec2Processor.from_pretrained(str(model_dir))
    model = Wav2Vec2ForCTC.from_pretrained(str(model_dir)).to(device)
    model.eval()

    predictions = []
    for i, clip in enumerate(clips):
        audio_np = load_audio_np(clip["audio_path"])
        inputs = processor(
            audio_np,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        ).input_values.to(device=device, dtype=model.dtype)

        with torch.no_grad():
            logits = model(inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)
        pred_text = processor.batch_decode(pred_ids)[0].replace("|", " ")
        predictions.append(prediction_record(clip, pred_text))

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(clips)}] done")

    return predictions


def run_finetuned(model_key: str, clips: list[dict]):
    """Evaluate a single fine-tuned model and save predictions."""
    family = model_family(model_key)
    if family == "lm":
        print(f"[SKIP] {model_key} is an LM-fusion result. Run 09_lm_shallow_fusion.py instead.")
        return
    if family == "unknown":
        print(f"[ERROR] Unsupported model key: {model_key}")
        raise SystemExit(1)

    try:
        import torch
    except ImportError:
        print("[ERROR] pip install torch")
        raise SystemExit(1)

    model_dir = resolve_model_dir(model_key)
    if not model_dir.exists():
        print(f"[ERROR] {model_dir} not found.")
        if family == "whisper":
            subset = model_key.removeprefix("ft_whisper_")
            print(f"        Run: python fine_tuning/02_finetune_whisper.py --subset {subset}")
        else:
            arch, _, subset = model_key.removeprefix("ft_").partition("_")
            subset = subset or "all"
            print(f"        Run: python fine_tuning/03_finetune_wav2vec2.py --arch {arch} --subset {subset}")
        raise SystemExit(1)

    label = FINETUNED.get(model_key, model_key)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*60}")
    print(f"Evaluating: {label}")
    print(f"  Model dir: {model_dir}")
    print(f"  Family   : {family}")
    print(f"  Device   : {device}")
    print(f"{'='*60}")

    if family == "whisper":
        predictions = run_whisper(model_dir, clips, device)
    else:
        predictions = run_ctc(model_dir, clips, device)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{model_key}_predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(predictions)} predictions -> {out_path}")


def main(model_key: str | None = None):
    clips = load_test_manifest()
    keys = [model_key] if model_key else checkpoint_model_keys()
    for key in keys:
        run_finetuned(key, clips)
    print("\nFine-tuned model evaluation complete.")
    print("Next: python 03_compute_metrics.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(FINETUNED.keys()),
        default=None,
        help="Which fine-tuned model to evaluate (default: all checkpoint-backed models)",
    )
    args = parser.parse_args()
    main(args.model)
