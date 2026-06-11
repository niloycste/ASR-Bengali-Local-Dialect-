"""
Fine-tuning Step 3: Fine-tune wav2vec2-XLS-R or WavLM on your Bengali CS dataset.

These are CTC-based (encoder-only) models — different architecture from Whisper.
Including them gives your paper a strong CTC baseline vs your seq2seq Whisper model.

Models supported (set in config.py CTC_MODELS):
  wav2vec2 : facebook/wav2vec2-xls-r-300m  — 300M params, multilingual SSL
  wavlm    : microsoft/wavlm-large          — 316M params, strong on noisy audio

Key difference from Whisper:
  - Whisper: seq2seq, generates one token at a time, handles mixed-script naturally
  - wav2vec2/WavLM: CTC, predicts one character/subword per frame in parallel
  - CTC needs a vocabulary that covers BOTH Bengali Unicode and Latin characters
  - This script builds that mixed vocabulary automatically from your training data

Usage:
    python 03_finetune_wav2vec2.py --arch wav2vec2    # fine-tune wav2vec2-XLS-R
    python 03_finetune_wav2vec2.py --arch wavlm        # fine-tune WavLM-Large
    python 03_finetune_wav2vec2.py --arch wav2vec2 --subset cs_only  # ablation

Requirements:
    pip install -r requirements.txt
    GPU with 16GB+ VRAM recommended (use --arch wav2vec2 on smaller GPUs)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CTC_MODELS, DATASET_DIR, MODELS_DIR, SUBSETS, TRAINING

try:
    import pandas as pd
    from datasets import Dataset, Audio, load_dataset
except ImportError:
    print("[ERROR] pip install datasets pandas")
    raise SystemExit(1)

SAMPLE_RATE   = 16_000
BLANK_TOKEN   = "[PAD]"
UNK_TOKEN     = "[UNK]"
WORD_DELIM    = "|"    # replaces space in CTC vocabulary


def load_audio_array(path: str, retries: int = 3):
    """Load a wav as 16 kHz mono float32. Retries transient read errors
    (e.g. Google Drive I/O hiccups). Returns None if it ultimately fails, so a
    single unreadable file doesn't crash a long training run."""
    import time
    import soundfile as sf
    for attempt in range(retries):
        try:
            audio, sr = sf.read(path, dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != SAMPLE_RATE:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            return audio
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
            else:
                print(f"    [WARN] skipping unreadable audio: {path} ({e})")
    return None


# ── Data collator ──────────────────────────────────────────────────────────────

@dataclass
class DataCollatorCTCWithPadding:
    """
    Pads CTC input_values and labels independently to the longest in the batch.

    Acoustic features are extracted on-the-fly from the lazily-decoded audio so
    that raw waveforms are never stored in the Arrow cache (which would blow past
    PyArrow's memory limits on the full 160h "all" subset).
    """
    processor: Any
    padding: bool = True

    def __call__(self, features: list[dict]) -> dict:
        input_features = []
        kept = []
        for f in features:
            wav = load_audio_array(f["audio"])
            if wav is None:
                continue
            extracted = self.processor.feature_extractor(
                wav, sampling_rate=SAMPLE_RATE
            )
            input_features.append({"input_values": extracted.input_values[0]})
            kept.append(f)

        batch = self.processor.feature_extractor.pad(
            input_features, padding=self.padding, return_tensors="pt"
        )
        label_features = [{"input_ids": f["labels"]} for f in kept]
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=self.padding, return_tensors="pt"
        )
        # Replace padding with -100 so it is ignored by the CTC loss.
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


# ── Vocabulary ─────────────────────────────────────────────────────────────────

def build_vocab(train_csv: str, dev_csv: str) -> dict[str, int]:
    """
    Build a character-level vocabulary covering Bengali Unicode + Latin ASCII.
    This is required for CTC models — they need an explicit character set.

    Bengali script: U+0980–U+09FF
    Latin (English): a-z A-Z
    Digits: 0-9 (both Bengali ০-৯ and ASCII)
    """

    chars = set()
    for csv_path in [train_csv, dev_csv]:
        df = pd.read_csv(csv_path, encoding="utf-8")
        for text in df["transcript"].dropna():
            text = str(text).strip()
            for ch in text:
                # Include Bengali Unicode block, Latin, digits, punctuation
                if (
                    "\u0980" <= ch <= "\u09FF"   # Bengali
                    or "a"   <= ch.lower() <= "z" # Latin
                    or "0"   <= ch <= "9"          # ASCII digits
                    or ch in "।,-.'?! "            # common punctuation
                ):
                    chars.add(ch)

    # Replace space with word delimiter token for CTC
    chars.discard(" ")
    vocab = {BLANK_TOKEN: 0, UNK_TOKEN: 1, WORD_DELIM: 2}
    for i, ch in enumerate(sorted(chars), start=3):
        vocab[ch] = i

    return vocab


# ── Dataset preparation ───────────────────────────────────────────────────────

def prepare_dataset(csv_path: str, processor, subset: str) -> Dataset:
    """
    Load manifest CSV and convert to HuggingFace Dataset with CTC features.
    """
    ds = Dataset.from_csv(str(csv_path))

    # Filter by subset
    if subset == "bn_only":
        ds = ds.filter(lambda x: not x.get("is_code_switched", False))
    elif subset == "cs_only":
        ds = ds.filter(lambda x: x.get("is_code_switched", False))

    # Drop rows with missing audio or transcript
    ds = ds.filter(lambda x: x and Path(str(x["audio_path"])).exists() and x["transcript"] and str(x["transcript"]).strip() != "")
    print(f"  {len(ds)} clips after filtering for subset='{subset}'")

    # Keep audio as a plain path STRING — do NOT cast to Audio(). Casting a
    # large_string column (which Dataset.from_csv produces) to Audio fails on
    # some pyarrow versions ("Unsupported cast from large_string to struct").
    # The collator loads each wav from its path on-the-fly, so waveforms never
    # get materialized into the Arrow cache.
    original_columns = list(ds.column_names)

    def _prepare(example):
        # Tokenize transcript into CTC label ids (space → word delimiter token).
        # Modern API — processor.as_target_processor() was removed in transformers 4.x.
        text   = str(example["transcript"]).strip().replace(" ", WORD_DELIM)
        labels = processor.tokenizer(text).input_ids
        # Cheap length proxy from the manifest duration so group_by_length can
        # sort batches without decoding audio.
        input_length = int(float(example.get("duration_sec") or 0.0) * SAMPLE_RATE)
        return {
            "audio":        example["audio_path"],
            "labels":       labels,
            "input_length": input_length,
        }

    ds = ds.map(_prepare, remove_columns=original_columns)
    return ds


# ── WER metric ─────────────────────────────────────────────────────────────────

def make_compute_metrics(processor):
    try:
        import evaluate
    except ImportError:
        print("[ERROR] pip install evaluate")
        raise SystemExit(1)

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids    = pred_logits.argmax(axis=-1)
        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        # Restore word delimiter to space for WER computation
        pred_str  = [p.replace(WORD_DELIM, " ").strip() for p in pred_str]
        label_str = [l.replace(WORD_DELIM, " ").strip() for l in label_str]

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer, 4)}

    return compute_metrics


# ── Training ───────────────────────────────────────────────────────────────────

def finetune_ctc(arch: str, subset: str):
    cfg = CTC_MODELS.get(arch)
    if cfg is None:
        print(f"[ERROR] Unknown arch '{arch}'. Choose: {list(CTC_MODELS.keys())}")
        raise SystemExit(1)

    try:
        import torch
        from transformers import (
            Wav2Vec2CTCTokenizer,
            Wav2Vec2FeatureExtractor,
            Wav2Vec2Processor,
            AutoModelForCTC,
            TrainingArguments,
            Trainer,
        )
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError:
        print("[ERROR] pip install transformers torch accelerate")
        raise SystemExit(1)

    model_name = cfg["model"]
    out_key    = f"ft_{arch}_{subset}" if subset != "all" else cfg["out_key"]
    model_dir  = MODELS_DIR / out_key
    vocab_dir  = MODELS_DIR / f"{out_key}_vocab"
    model_dir.mkdir(parents=True, exist_ok=True)
    vocab_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Fine-tuning {cfg['label']} (CTC)")
    print(f"  Base model : {model_name}")
    print(f"  Subset     : {subset}  — {SUBSETS[subset]}")
    print(f"  Output     : {model_dir}")
    print(f"{'='*60}\n")

    # Build vocabulary
    train_csv = str(DATASET_DIR / "train" / "manifest.csv")
    dev_csv   = str(DATASET_DIR / "dev"   / "manifest.csv")
    vocab     = build_vocab(train_csv, dev_csv)

    vocab_path = vocab_dir / "vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print(f"Vocabulary: {len(vocab)} characters → {vocab_path}")

    # Tokenizer and processor
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token=UNK_TOKEN,
        pad_token=BLANK_TOKEN,
        word_delimiter_token=WORD_DELIM,
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=SAMPLE_RATE,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )

    # Load and prepare datasets
    print("Preparing training data ...")
    train_ds = prepare_dataset(train_csv, processor, subset)
    dev_ds   = prepare_dataset(dev_csv,   processor, subset)
    print(f"Train: {len(train_ds)} | Dev: {len(dev_ds)}")

    # Load model
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device == "cuda" and not use_bf16
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True   # faster matmuls on A100/H100
        torch.backends.cudnn.allow_tf32 = True
    print(f"Loading {model_name} on {device} | bf16={use_bf16} fp16={use_fp16}")
    model = AutoModelForCTC.from_pretrained(
        model_name,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(vocab),
        ignore_mismatched_sizes=True,   # new vocab head size differs from pretrained
    )

    # Freeze feature encoder — only fine-tune transformer layers
    model.freeze_feature_encoder()

    collator        = DataCollatorCTCWithPadding(processor=processor, padding=True)
    compute_metrics = make_compute_metrics(processor)

    training_args = TrainingArguments(
        output_dir                  = str(model_dir),
        num_train_epochs            = TRAINING["num_train_epochs"],
        per_device_train_batch_size = TRAINING["per_device_train_batch_size"],
        per_device_eval_batch_size  = TRAINING["per_device_eval_batch_size"],
        gradient_accumulation_steps = TRAINING["gradient_accumulation_steps"],
        learning_rate               = 1e-4,    # CTC needs higher LR than seq2seq
        warmup_steps                = TRAINING["warmup_steps"],
        fp16                        = use_fp16,
        bf16                        = use_bf16,
        dataloader_num_workers      = TRAINING.get("dataloader_num_workers", 0),
        save_steps                  = TRAINING["save_steps"],
        eval_steps                  = TRAINING["eval_steps"],
        eval_strategy               = "steps",
        save_strategy               = "steps",
        logging_steps               = TRAINING["logging_steps"],
        load_best_model_at_end      = True,
        metric_for_best_model       = "wer",
        greater_is_better           = False,
        push_to_hub                 = False,
        report_to                   = "none",
        remove_unused_columns       = False,   # keep the "audio" path column for the collator
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = dev_ds,
        data_collator   = collator,
        compute_metrics = compute_metrics,
    )

    last_checkpoint = get_last_checkpoint(str(model_dir))
    if last_checkpoint is not None:
        print(f"\n[INFO] Resuming training from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
        
    trainer.save_model(str(model_dir))
    processor.save_pretrained(str(model_dir))

    print(f"\nModel saved: {model_dir}")
    print(f"Next: python ../evaluation/02_run_finetuned.py --model {out_key}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        choices=list(CTC_MODELS.keys()),
        default=None,
        help="CTC model architecture: wav2vec2 | wavlm. If omitted, runs both.",
    )
    parser.add_argument(
        "--subset",
        choices=list(SUBSETS.keys()),
        default="all",
        help="Training subset (default: all)",
    )
    args = parser.parse_args()

    if args.arch:
        finetune_ctc(args.arch, args.subset)
    else:
        # No arch given — run both wav2vec2 and wavlm on the chosen subset
        for arch in CTC_MODELS:
            finetune_ctc(arch, args.subset)
