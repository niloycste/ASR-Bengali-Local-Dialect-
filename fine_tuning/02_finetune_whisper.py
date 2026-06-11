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
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Pads input_features and labels to the longest sequence in the batch.
    Replaces padding token ids with -100 so they are ignored in the loss.
    """
    processor: Any

    def __call__(self, features: list[dict]) -> dict:
        import torch

        # Load audio on-the-fly via soundfile (robust to flaky mounts). Samples
        # whose audio can't be read are skipped so one bad file won't crash training.
        input_features = []
        kept = []
        for f in features:
            wav = load_audio_array(f["audio"])
            if wav is None:
                continue
            inputs = self.processor.feature_extractor(
                wav, sampling_rate=16000, return_tensors="pt"
            )
            input_features.append({"input_features": inputs.input_features[0]})
            kept.append(f)

        # Pad extracted features
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad labels (truncate to Whisper's 448 max), aligned to kept samples
        label_features = []
        for f in kept:
            lab = f["labels"][:448] if len(f["labels"]) > 448 else f["labels"]
            label_features.append({"input_ids": lab})
        labels_batch = self.processor.tokenizer.pad(
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

    # Normalize both sides before WER so the training-time metric reflects the
    # real WER (not punctuation/danda/spacing noise) and best-model selection
    # uses the true metric. Falls back to plain strip() if the import fails.
    import sys
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from dataset_pipeline.text_normalization import full_normalize
    except Exception:
        full_normalize = lambda x: x.strip()

    def compute_metrics(pred):
        pred_ids   = pred.predictions
        label_ids  = pred.label_ids

        # Replace -100 back to padding token
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str  = processor.tokenizer.batch_decode(pred_ids,   skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids,  skip_special_tokens=True)

        # Normalize (strip punctuation/danda, fix Unicode, map pronunciation variants)
        pred_str  = [full_normalize(p) for p in pred_str]
        label_str = [full_normalize(l) for l in label_str]

        # Drop pairs whose reference is empty after normalization
        paired = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
        if not paired:
            return {"wer": 1.0}
        ps, ls = zip(*paired)

        wer = wer_metric.compute(predictions=list(ps), references=list(ls))
        return {"wer": round(wer, 4)}

    return compute_metrics


# ── Training ───────────────────────────────────────────────────────────────────

def finetune(subset: str):
    try:
        import torch
        from datasets import load_from_disk, Audio
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

    # Detect device and pick precision. Prefer bf16 on GPUs that support it
    # (A100/H100) — more numerically stable than fp16. Fall back to fp16 on older
    # GPUs (T4/V100), and full precision on CPU.
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    use_fp16 = device == "cuda" and not use_bf16
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True   # faster matmuls on A100/H100
        torch.backends.cudnn.allow_tf32 = True
    print(f"Device: {device} | bf16={use_bf16} fp16={use_fp16}")

    # Load processor and model
    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL, language=LANGUAGE, task=TASK
    )
    # Load in float32 and let the trainer's fp16 flag handle mixed precision
    # (loading weights directly in float16 can make training unstable).
    model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL)

    # Set forced decoder ids for Bengali transcription
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task=TASK
    )
    model.config.suppress_tokens = []

    # Load datasets
    train_ds = load_from_disk(str(data_dir / "train"))
    dev_ds   = load_from_disk(str(data_dir / "dev"))

    # Audio stays as a file-path string; the collator loads it on-the-fly via
    # soundfile (with retries). This is more robust than the datasets/torchcodec
    # decoder when reading from flaky mounts like Google Drive.

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
        bf16                        = use_bf16,
        dataloader_num_workers      = TRAINING.get("dataloader_num_workers", 0),
        predict_with_generate       = TRAINING["predict_with_generate"],
        generation_max_length       = TRAINING["generation_max_length"],
        save_steps                  = TRAINING["save_steps"],
        eval_steps                  = TRAINING["eval_steps"],
        save_total_limit            = TRAINING.get("save_total_limit", 3),
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
