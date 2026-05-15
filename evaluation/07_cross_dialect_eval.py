"""
Evaluation Step 7: Cross-dialect generalization experiment.

Tests whether the model trained on ALL dialects generalizes to a held-out dialect
it has never seen during training.

Experiment design:
  For each dialect D:
    - Train on all clips EXCEPT dialect D  (leave-one-dialect-out)
    - Test on dialect D only
    - Compare WER to model trained WITH dialect D included

This answers the key question:
  "Does your model work on a new dialect it was never trained on?"
  — critical for real-world deployment and for your paper's impact claim.

NOTE: Full leave-one-out training is expensive (one full training run per dialect).
      This script provides TWO options:
        Option A (fast):  re-evaluate existing ft_whisper_all model on each
                          dialect's test clips separately — shows per-dialect
                          performance without retraining.
        Option B (full):  actually retrain with one dialect held out.
                          Set RUN_FULL_TRAINING = True below.

Usage:
    python 07_cross_dialect_eval.py              # Option A (fast, no retraining)
    python 07_cross_dialect_eval.py --full       # Option B (full retraining)
    python 07_cross_dialect_eval.py --dialect Sylhet --full  # one dialect only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fine_tuning.config import DATASET_DIR, MODELS_DIR, RESULTS_DIR

CROSS_DIALECT_DIR = RESULTS_DIR / "cross_dialect"
SAMPLE_RATE = 16_000


def load_test_manifest() -> list[dict]:
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pip install pandas"); raise SystemExit(1)

    csv_path = DATASET_DIR / "test" / "manifest.csv"
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found. Run 05_build_dataset.py first.")
        raise SystemExit(1)

    df   = pd.read_csv(csv_path, encoding="utf-8")
    rows = df.to_dict("records")
    rows = [r for r in rows if Path(str(r["audio_path"])).exists()]
    return rows


def load_train_manifest() -> list[dict]:
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pip install pandas"); raise SystemExit(1)

    csv_path = DATASET_DIR / "train" / "manifest.csv"
    df = pd.read_csv(csv_path, encoding="utf-8")
    return df.to_dict("records")


def get_dialects(clips: list[dict]) -> list[str]:
    return sorted(set(c.get("dialect", "unknown") for c in clips))


# ── Option A: evaluate existing model per dialect ─────────────────────────────

def eval_per_dialect_fast(model_key: str = "ft_whisper_all"):
    """
    Load predictions from an already-run model and compute WER per dialect.
    Fast — no retraining needed. Shows within-distribution per-dialect performance.
    """
    pred_path = RESULTS_DIR / f"{model_key}_predictions.json"
    if not pred_path.exists():
        print(f"[ERROR] {pred_path} not found.")
        print(f"        Run: python 02_run_finetuned.py --model {model_key}")
        raise SystemExit(1)

    with open(pred_path, encoding="utf-8") as f:
        predictions = json.load(f)

    try:
        from jiwer import wer as jiwer_wer, cer as jiwer_cer
    except ImportError:
        print("[ERROR] pip install jiwer"); raise SystemExit(1)

    from collections import defaultdict
    by_dialect = defaultdict(list)
    for p in predictions:
        by_dialect[p.get("dialect", "unknown")].append(p)

    results = {}
    print(f"\n{'='*60}")
    print(f"Per-dialect evaluation: {model_key}")
    print(f"{'='*60}")
    print(f"{'Dialect':<18} {'N':>5} {'WER':>7} {'CER':>7} {'WER-CS':>8} {'CS clips':>8}")
    print("-" * 60)

    for dialect in sorted(by_dialect.keys()):
        preds    = by_dialect[dialect]
        refs     = [p["reference"]  for p in preds if p["reference"].strip()]
        hyps     = [p["hypothesis"] for p in preds if p["reference"].strip()]
        cs_preds = [p for p in preds if p.get("is_code_switched")]
        refs_cs  = [p["reference"]  for p in cs_preds if p["reference"].strip()]
        hyps_cs  = [p["hypothesis"] for p in cs_preds if p["reference"].strip()]

        if not refs:
            continue

        wer_val = jiwer_wer(refs, hyps)
        cer_val = jiwer_cer(refs, hyps)
        wer_cs  = jiwer_wer(refs_cs, hyps_cs) if refs_cs else None

        results[dialect] = {
            "n_clips":    len(preds),
            "n_cs_clips": len(cs_preds),
            "wer":        round(wer_val, 4),
            "cer":        round(cer_val, 4),
            "wer_cs":     round(wer_cs, 4) if wer_cs else None,
        }
        print(
            f"{dialect:<18} {len(preds):>5} "
            f"{wer_val*100:>6.1f}% "
            f"{cer_val*100:>6.1f}% "
            f"{wer_cs*100 if wer_cs else 0:>7.1f}% "
            f"{len(cs_preds):>8}"
        )

    CROSS_DIALECT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CROSS_DIALECT_DIR / f"{model_key}_per_dialect.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")
    return results


# ── Option B: leave-one-dialect-out retraining ────────────────────────────────

def load_audio_np(audio_path: str):
    import numpy as np
    from pydub import AudioSegment
    audio = (
        AudioSegment.from_file(audio_path)
        .set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
    )
    return np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe_clips(model, processor, clips: list[dict], device: str) -> list[dict]:
    import torch
    predictions = []
    for clip in clips:
        if not Path(str(clip["audio_path"])).exists():
            continue
        audio_np = load_audio_np(clip["audio_path"])
        inputs   = processor(
            audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(device)
        with torch.no_grad():
            ids = model.generate(inputs)
        pred_text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        predictions.append({
            "clip_id":          clip.get("clip_id", ""),
            "reference":        str(clip.get("transcript", "")).strip(),
            "hypothesis":       pred_text,
            "dialect":          clip.get("dialect", "unknown"),
            "is_code_switched": bool(clip.get("is_code_switched", False)),
        })
    return predictions


def run_leave_one_out(held_out_dialect: str):
    """
    Full leave-one-dialect-out experiment:
      1. Filter training data to exclude held_out_dialect
      2. Fine-tune Whisper on the remaining dialects
      3. Evaluate on held_out_dialect test clips
    """
    try:
        import torch
        import pandas as pd
        from transformers import (
            WhisperProcessor, WhisperForConditionalGeneration,
            Seq2SeqTrainer, Seq2SeqTrainingArguments,
        )
        from datasets import Dataset, Audio
        import evaluate
    except ImportError:
        print("[ERROR] pip install transformers datasets evaluate torch pandas")
        raise SystemExit(1)

    from fine_tuning.config import BASE_MODEL, LANGUAGE, TASK, TRAINING
    from dataclasses import dataclass
    from typing import Any

    print(f"\n{'='*60}")
    print(f"Leave-one-out: held-out dialect = {held_out_dialect}")
    print(f"{'='*60}")

    # Load train manifest, exclude held-out dialect
    train_rows = load_train_manifest()
    train_rows = [r for r in train_rows if r.get("dialect") != held_out_dialect]
    print(f"Training clips (excluding {held_out_dialect}): {len(train_rows)}")

    # Load test clips for held-out dialect only
    test_rows  = load_test_manifest()
    test_rows  = [r for r in test_rows if r.get("dialect") == held_out_dialect]
    print(f"Test clips for {held_out_dialect}: {len(test_rows)}")

    if not train_rows or not test_rows:
        print(f"[SKIP] Insufficient data for dialect: {held_out_dialect}")
        return None

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16  = TRAINING["fp16"] if device == "cuda" else False
    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=LANGUAGE, task=TASK)
    model     = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task=TASK
    )
    model.config.suppress_tokens = []

    # Build HF dataset from filtered rows
    def _to_hf(rows):
        records = []
        for row in rows:
            ap = str(row.get("audio_path", ""))
            tx = str(row.get("transcript", "")).strip()
            if not Path(ap).exists() or not tx:
                continue
            records.append({"audio": ap, "sentence": tx})
        ds = Dataset.from_list(records)
        ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

        def _map(batch):
            labels = processor.tokenizer(batch["sentence"], return_tensors="np").input_ids
            return {"labels": labels[0].tolist()}

        return ds.map(_map)

    train_ds = _to_hf(train_rows)

    # Minimal training — fewer epochs for leave-one-out runs
    out_dir = MODELS_DIR / f"lodo_{held_out_dialect}"
    out_dir.mkdir(parents=True, exist_ok=True)

    wer_metric = evaluate.load("wer")

    @dataclass
    class Collator:
        processor: Any
        def __call__(self, features):
            import torch
            inp = []
            for f in features:
                inputs = self.processor.feature_extractor(
                    f["audio"]["array"], sampling_rate=16000, return_tensors="pt"
                )
                inp.append({"input_features": inputs.input_features[0]})
            lbl = [{"input_ids": f["labels"]} for f in features]
            batch  = self.processor.feature_extractor.pad(inp, return_tensors="pt")
            labels = self.processor.tokenizer.pad(lbl, return_tensors="pt")
            ids = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
            if (ids[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                ids = ids[:, 1:]
            batch["labels"] = ids
            return batch

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        p = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        l = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": round(wer_metric.compute(predictions=p, references=l), 4)}

    args = Seq2SeqTrainingArguments(
        output_dir                  = str(out_dir),
        num_train_epochs            = 5,          # fewer epochs for leave-one-out
        per_device_train_batch_size = TRAINING["per_device_train_batch_size"],
        gradient_accumulation_steps = TRAINING["gradient_accumulation_steps"],
        learning_rate               = TRAINING["learning_rate"],
        warmup_steps                = 200,
        fp16                        = use_fp16,
        predict_with_generate       = True,
        generation_max_length       = 225,
        save_steps                  = 500,
        eval_steps                  = 500,
        eval_strategy               = "no",       # no eval during leave-one-out
        logging_steps               = 50,
        push_to_hub                 = False,
        report_to                   = "none",
        remove_unused_columns       = False,
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        data_collator   = Collator(processor=processor),
        compute_metrics = compute_metrics,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))

    # Evaluate on held-out dialect
    model.eval()
    preds = transcribe_clips(model, processor, test_rows, device)

    from jiwer import wer as jiwer_wer, cer as jiwer_cer
    refs  = [p["reference"]  for p in preds if p["reference"]]
    hyps  = [p["hypothesis"] for p in preds if p["reference"]]
    wer   = jiwer_wer(refs, hyps) if refs else None
    cer   = jiwer_cer(refs, hyps) if refs else None

    result = {
        "held_out_dialect": held_out_dialect,
        "n_test_clips":     len(preds),
        "wer":              round(wer, 4) if wer else None,
        "cer":              round(cer, 4) if cer else None,
        "model_dir":        str(out_dir),
    }

    CROSS_DIALECT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CROSS_DIALECT_DIR / f"lodo_{held_out_dialect}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nHeld-out WER ({held_out_dialect}): {wer:.4f}")
    print(f"Saved: {out_path}")
    return result


def main(full: bool = False, dialect: str | None = None):
    CROSS_DIALECT_DIR.mkdir(parents=True, exist_ok=True)

    if not full:
        # Fast option: just re-evaluate existing model per dialect
        print("Option A: Per-dialect evaluation of existing model (no retraining)")
        eval_per_dialect_fast("ft_whisper_all")
        print("\nTo run full leave-one-out retraining: python 07_cross_dialect_eval.py --full")
    else:
        # Full leave-one-out retraining
        test_rows = load_test_manifest()
        dialects  = [dialect] if dialect else get_dialects(test_rows)
        print(f"Option B: Full leave-one-out for dialects: {dialects}")

        all_results = {}
        for d in dialects:
            result = run_leave_one_out(d)
            if result:
                all_results[d] = result

        summary_path = CROSS_DIALECT_DIR / "lodo_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"\nLeave-one-out summary:")
        for d, r in sorted(all_results.items()):
            print(f"  {d:<18}: WER={r['wer']}  CER={r['cer']}")
        print(f"Saved: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full leave-one-dialect-out retraining (slow). Default: fast eval only.",
    )
    parser.add_argument(
        "--dialect",
        default=None,
        help="Run leave-one-out for a single dialect only (use with --full).",
    )
    args = parser.parse_args()
    main(full=args.full, dialect=args.dialect)
