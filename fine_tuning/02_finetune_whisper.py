"""
Fine-tuning Step 2: Fine-tune Whisper large-v3 on your Bengali CS dataset.

Runs four experiments depending on --subset:
  all     → Experiment 4 / your proposed full model
  bn_only → Experiment 2 / ablation A (dialect only, no CS)
  cs_only → Experiment 3 / ablation B (CS only, no BN_ONLY)

Experiment 1 (zero-shot Whisper) requires no training — see evaluation/01_run_baselines.py.

Usage:
    python 02_finetune_whisper.py --subset all
    python 02_finetune_whisper.py --subset bn_only
    python 02_finetune_whisper.py --subset cs_only

Requirements:
    pip install transformers datasets evaluate jiwer accelerate
    GPU strongly recommended (16 GB+ VRAM for large-v3; use small/medium on <16 GB)
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BASE_MODEL, HF_DATA_DIR, LANGUAGE, MODELS_DIR, SUBSETS, TASK, TRAINING
)

SAMPLE_RATE = 16_000


# ── Data collator ──────────────────────────────────────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Pads input_features and labels to the longest sequence in the batch.
    Replaces padding token ids with -100 so they are ignored in the loss.
    """
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        import torch

        # Extract audio features on-the-fly
        input_features = []
        for f in features:
            inputs = self.processor.feature_extractor(
                f["audio"]["array"],
                sampling_rate=16000,
                return_tensors="pt"
            )
            input_features.append({"input_features": inputs.input_features[0]})

        # Pad extracted features
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Truncate labels to Whisper's absolute max length (448) to prevent crashes
        for f in features:
            if len(f["labels"]) > 448:
                f["labels"] = f["labels"][:448]

        # Pad labels
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch   = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        # Replace padding with -100 → ignored in cross-entropy loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        # Strip BOS token if present (added by the tokenizer automatically)
        if (
            labels[:, 0] == self.processor.tokenizer.bos_token_id
        ).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ── WER metric ─────────────────────────────────────────────────────────────────

def make_compute_metrics(processor):
    """Return a compute_metrics function that calculates WER during training."""
    try:
        import evaluate
    except ImportError:
        print("[ERROR] evaluate not installed. Run: pip install evaluate")
        raise SystemExit(1)

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids   = pred.predictions
        label_ids  = pred.label_ids

        # Replace -100 back to padding token
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.tokenizer.batch_decode(pred_ids,   skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids,  skip_special_tokens=True)

        # Strip leading/trailing whitespace
        pred_str  = [p.strip() for p in pred_str]
        label_str = [l.strip() for l in label_str]

        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer, 4)}

    return compute_metrics


# ── Training ───────────────────────────────────────────────────────────────────

def finetune(subset: str):
    try:
        import torch
        from datasets import load_from_disk
        from transformers import (
            WhisperForConditionalGeneration,
            WhisperProcessor,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
        from transformers.trainer_utils import get_last_checkpoint
    except ImportError:
        print("[ERROR] transformers/datasets not installed.")
        print("        Run: pip install transformers datasets evaluate jiwer accelerate")
        raise SystemExit(1)

    data_dir  = HF_DATA_DIR / subset
    model_dir = MODELS_DIR / f"ft_whisper_{subset}"
    model_dir.mkdir(parents=True, exist_ok=True)

    # Check data exists
    for split in ["train", "dev"]:
        if not (data_dir / split).exists():
            print(f"[ERROR] {data_dir / split} not found.")
            print(f"        Run: python 01_prepare_hf_dataset.py --subset {subset}")
            raise SystemExit(1)

    print(f"\n{'='*60}")
    print(f"Fine-tuning Whisper — subset: {subset}")
    print(f"  Description : {SUBSETS[subset]}")
    print(f"  Base model  : {BASE_MODEL}")
    print(f"  Output dir  : {model_dir}")
    print(f"{'='*60}\n")

    # Detect device and auto-disable FP16 if running on CPU to prevent crashes
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16 = TRAINING["fp16"] if device == "cuda" else False

    # Load processor and model
    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL, language=LANGUAGE, task=TASK
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16 if use_fp16 else torch.float32
    )

    # Set forced decoder ids for Bengali transcription
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task=TASK
    )
    model.config.suppress_tokens = []

    # Load datasets
    train_ds = load_from_disk(str(data_dir / "train"))
    dev_ds   = load_from_disk(str(data_dir / "dev"))

    print(f"Train: {len(train_ds)} examples")
    print(f"Dev:   {len(dev_ds)} examples\n")

    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics = make_compute_metrics(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = str(model_dir),
        num_train_epochs            = TRAINING["num_train_epochs"],
        per_device_train_batch_size = TRAINING["per_device_train_batch_size"],
        per_device_eval_batch_size  = TRAINING["per_device_eval_batch_size"],
        gradient_accumulation_steps = TRAINING["gradient_accumulation_steps"],
        learning_rate               = TRAINING["learning_rate"],
        warmup_steps                = TRAINING["warmup_steps"],
        fp16                        = use_fp16,
        predict_with_generate       = TRAINING["predict_with_generate"],
        generation_max_length       = TRAINING["generation_max_length"],
        save_steps                  = TRAINING["save_steps"],
        eval_steps                  = TRAINING["eval_steps"],
        eval_strategy               = "steps",
        save_strategy               = "steps",
        logging_steps               = TRAINING["logging_steps"],
        load_best_model_at_end      = TRAINING["load_best_model_at_end"],
        metric_for_best_model       = TRAINING["metric_for_best_model"],
        greater_is_better           = TRAINING["greater_is_better"],
        push_to_hub                 = TRAINING["push_to_hub"],
        report_to                   = TRAINING["report_to"],
        remove_unused_columns       = False,
    )

    trainer = Seq2SeqTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = dev_ds,
        data_collator    = collator,
        compute_metrics  = compute_metrics,
    )

    last_checkpoint = get_last_checkpoint(str(model_dir))
    if last_checkpoint is not None:
        print(f"\n[INFO] Resuming training from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    # Save final model and processor
    trainer.save_model(str(model_dir))
    processor.save_pretrained(str(model_dir))

    print(f"\nModel saved to: {model_dir}")
    print(f"Next: python ../evaluation/02_run_finetuned.py --model ft_whisper_{subset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        choices=list(SUBSETS.keys()),
        default=None,
        help="Training subset: all | bn_only | cs_only. If omitted, runs all three.",
    )
    args = parser.parse_args()

    if args.subset:
        finetune(args.subset)
    else:
        # No argument — run all three experiments sequentially
        for subset in SUBSETS:
            finetune(subset)
