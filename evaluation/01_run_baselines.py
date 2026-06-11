"""
Evaluation Step 1: Run zero-shot baseline models on the test set.

Baselines:
  - whisper_large_v3    : OpenAI Whisper large-v3 (zero-shot)
  - tugstugi_whisper_bn : Tugstugi fine-tuned on Ben-10 dialect data
  - wav2vec2_bn         : wav2vec2-XLS-R fine-tuned on standard Bengali

Output: evaluation/results/<model_key>_predictions.json

Usage:
    python 01_run_baselines.py                           # all baselines
    python 01_run_baselines.py --model whisper_large_v3  # one model
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add project root to path to resolve cross-directory imports like 'config'
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fine_tuning.config import BASELINES, DATASET_DIR, RESULTS_DIR

SAMPLE_RATE = 16_000


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
    # Only keep rows where audio file exists
    rows = [r for r in rows if Path(str(r["audio_path"])).exists()]
    print(f"Test set: {len(rows)} clips")
    return rows


def load_audio_np(audio_path: str) -> np.ndarray:
    """Load wav file as float32 numpy array at 16 kHz."""
    from pydub import AudioSegment

    audio = (
        AudioSegment.from_file(audio_path)
        .set_frame_rate(SAMPLE_RATE)
        .set_channels(1)
        .set_sample_width(2)
    )
    return np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32) / 32768.0


# ── Whisper (openai-whisper) ───────────────────────────────────────────────────

def transcribe_whisper_openai(clips: list[dict], model_size: str) -> list[dict]:
    """Zero-shot Whisper via HuggingFace transformers — BATCHED, with capped
    generation.

    NOTE: We deliberately do NOT use the openai-whisper `transcribe()` API here.
    On out-of-distribution dialectal Bengali it is extremely slow because of
    (a) temperature fallback (each hard clip is re-decoded 5-6x) and
    (b) hallucination to max length. Batched HF generate with max_new_tokens is
    1-2 orders of magnitude faster and gives the same zero-shot baseline.
    """
    try:
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
    except ImportError:
        print("[ERROR] pip install transformers torch")
        raise SystemExit(1)

    hf_id = {
        "tiny":   "openai/whisper-tiny",
        "base":   "openai/whisper-base",
        "small":  "openai/whisper-small",
        "medium": "openai/whisper-medium",
        "large":  "openai/whisper-large-v3",
        "large-v3": "openai/whisper-large-v3",
    }.get(model_size, f"openai/whisper-{model_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading {hf_id} on {device} ...")
    processor = WhisperProcessor.from_pretrained(hf_id)
    model     = WhisperForConditionalGeneration.from_pretrained(hf_id).to(device)
    model.eval()
    # Force Bengali transcription via generation_config (v5-safe).
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = "bengali"
    model.generation_config.task = "transcribe"

    BATCH = 16
    predictions = []
    for start in range(0, len(clips), BATCH):
        batch  = clips[start:start + BATCH]
        audios = [load_audio_np(c["audio_path"]) for c in batch]
        inputs = processor(
            audios, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(device=device, dtype=model.dtype)

        with torch.no_grad():
            ids = model.generate(inputs, max_new_tokens=128)

        texts = processor.batch_decode(ids, skip_special_tokens=True)
        for clip, pred_text in zip(batch, texts):
            predictions.append({
                "clip_id":          clip["clip_id"],
                "audio_path":       clip["audio_path"],
                "reference":        str(clip.get("transcript", "")).strip(),
                "hypothesis":       pred_text.strip(),
                "dialect":          clip.get("dialect", "unknown"),
                "domain":           clip.get("domain", "General"),
                "is_code_switched": bool(clip.get("is_code_switched", False)),
                "bn_ratio":         float(clip.get("bn_ratio", 0.0)),
                "en_ratio":         float(clip.get("en_ratio", 0.0)),
                "duration_sec":     float(clip.get("duration_sec", 0.0)),
            })

        done = min(start + BATCH, len(clips))
        if (start // BATCH) % 5 == 0 or done == len(clips):
            print(f"  [{done}/{len(clips)}] done")

    return predictions


# ── Whisper (HuggingFace transformers) ────────────────────────────────────────

def transcribe_whisper_hf(clips: list[dict], model_name: str) -> list[dict]:
    """Whisper model loaded from HuggingFace hub (e.g. Tugstugi fine-tune)."""
    try:
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
    except ImportError:
        print("[ERROR] pip install transformers torch")
        raise SystemExit(1)

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading {model_name} on {device} ...")
    processor = WhisperProcessor.from_pretrained(model_name)
    model     = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
    model.eval()

    # Force Bengali transcription via generation_config (forced_decoder_ids is
    # deprecated in transformers v5 and can be ignored or conflict).
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = "bengali"
    model.generation_config.task = "transcribe"

    predictions = []
    for i, clip in enumerate(clips):
        audio_np = load_audio_np(clip["audio_path"])
        inputs   = processor(
            audio_np,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).input_features.to(device=device, dtype=model.dtype)

        with __import__("torch").no_grad():
            ids = model.generate(inputs)

        pred_text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        ref_text  = str(clip.get("transcript", "")).strip()

        predictions.append({
            "clip_id":         clip["clip_id"],
            "audio_path":      clip["audio_path"],
            "reference":       ref_text,
            "hypothesis":      pred_text,
            "dialect":         clip.get("dialect", "unknown"),
            "domain":          clip.get("domain", "General"),
            "is_code_switched": bool(clip.get("is_code_switched", False)),
            "bn_ratio":        float(clip.get("bn_ratio", 0.0)),
            "en_ratio":        float(clip.get("en_ratio", 0.0)),
            "duration_sec":    float(clip.get("duration_sec", 0.0)),
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(clips)}] done")

    return predictions


# ── wav2vec2 ──────────────────────────────────────────────────────────────────

def transcribe_wav2vec2(clips: list[dict], model_name: str) -> list[dict]:
    """wav2vec2 CTC model for Bengali."""
    try:
        import torch
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    except ImportError:
        print("[ERROR] pip install transformers torch")
        raise SystemExit(1)

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading {model_name} on {device} ...")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model     = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    model.eval()

    predictions = []
    for i, clip in enumerate(clips):
        audio_np = load_audio_np(clip["audio_path"])
        inputs   = processor(
            audio_np,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        ).input_values.to(device=device, dtype=model.dtype)

        with __import__("torch").no_grad():
            logits = model(inputs).logits

        pred_ids  = __import__("torch").argmax(logits, dim=-1)
        pred_text = processor.batch_decode(pred_ids)[0].strip()
        ref_text  = str(clip.get("transcript", "")).strip()

        predictions.append({
            "clip_id":         clip["clip_id"],
            "audio_path":      clip["audio_path"],
            "reference":       ref_text,
            "hypothesis":      pred_text,
            "dialect":         clip.get("dialect", "unknown"),
            "domain":          clip.get("domain", "General"),
            "is_code_switched": bool(clip.get("is_code_switched", False)),
            "bn_ratio":        float(clip.get("bn_ratio", 0.0)),
            "en_ratio":        float(clip.get("en_ratio", 0.0)),
            "duration_sec":    float(clip.get("duration_sec", 0.0)),
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(clips)}] done")

    return predictions


# ── MMS (Meta Massively Multilingual Speech) ──────────────────────────────────

def transcribe_mms(clips: list[dict], model_name: str, lang: str = "ben") -> list[dict]:
    """
    Meta MMS-1B: CTC model covering 1100+ languages.
    Set the target language via processor.tokenizer.set_target_lang(lang).
    lang = ISO 639-3 code: "ben" = Bengali
    """
    try:
        import torch
        from transformers import Wav2Vec2ForCTC, AutoProcessor
    except ImportError:
        print("[ERROR] pip install transformers torch")
        raise SystemExit(1)

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading {model_name} (lang={lang}) on {device} ...")
    processor = AutoProcessor.from_pretrained(model_name)
    model     = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    model.eval()
    processor.tokenizer.set_target_lang(lang)
    model.load_adapter(lang)

    predictions = []
    for i, clip in enumerate(clips):
        audio_np = load_audio_np(clip["audio_path"])
        inputs   = processor(
            audio_np,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
        ).input_values.to(device=device, dtype=model.dtype)

        with __import__("torch").no_grad():
            logits = model(inputs).logits

        pred_ids  = __import__("torch").argmax(logits, dim=-1)
        pred_text = processor.decode(pred_ids[0]).strip()
        ref_text  = str(clip.get("transcript", "")).strip()

        predictions.append({
            "clip_id":          clip["clip_id"],
            "audio_path":       clip["audio_path"],
            "reference":        ref_text,
            "hypothesis":       pred_text,
            "dialect":          clip.get("dialect", "unknown"),
            "domain":           clip.get("domain", "General"),
            "is_code_switched": bool(clip.get("is_code_switched", False)),
            "bn_ratio":         float(clip.get("bn_ratio", 0.0)),
            "en_ratio":         float(clip.get("en_ratio", 0.0)),
            "duration_sec":     float(clip.get("duration_sec", 0.0)),
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(clips)}] done")

    return predictions


# ── Driver ────────────────────────────────────────────────────────────────────

def run_baseline(model_key: str, clips: list[dict]):
    cfg = BASELINES[model_key]
    print(f"\n{'='*60}")
    print(f"Running baseline: {cfg['label']}")
    print(f"{'='*60}")

    if cfg["type"] == "whisper_openai":
        preds = transcribe_whisper_openai(clips, cfg["model"])
    elif cfg["type"] == "whisper_hf":
        preds = transcribe_whisper_hf(clips, cfg["model"])
    elif cfg["type"] == "wav2vec2":
        preds = transcribe_wav2vec2(clips, cfg["model"])
    elif cfg["type"] == "mms":
        preds = transcribe_mms(clips, cfg["model"], cfg.get("lang", "ben"))
    else:
        raise ValueError(f"Unknown model type: {cfg['type']}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{model_key}_predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(preds)} predictions → {out_path}")


def main(model_key: str | None = None):
    clips = load_test_manifest()

    keys = [model_key] if model_key else list(BASELINES.keys())
    for key in keys:
        run_baseline(key, clips)

    print("\nAll baselines done.")
    print("Next: python 03_compute_metrics.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(BASELINES.keys()),
        default=None,
        help="Run one specific baseline (default: all)",
    )
    args = parser.parse_args()
    main(args.model)
