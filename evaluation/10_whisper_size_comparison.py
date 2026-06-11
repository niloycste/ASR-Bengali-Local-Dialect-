"""
Experiment: Whisper model size comparison (tiny → base → small → medium → large-v3).

Two experiments:
  A. Zero-shot:   evaluate each size on your test set without fine-tuning
  B. Fine-tuned:  fine-tune each size on your "all" training set, then evaluate

This shows the accuracy vs compute tradeoff and answers:
  "Do practitioners need the full large-v3, or is medium/small sufficient?"

Outputs:
  evaluation/results/size_comparison/
    {size}_zeroshot_predictions.json
    {size}_finetuned_predictions.json
    size_comparison_summary.json
  evaluation/plots/
    whisper_size_comparison.pdf/png

Usage:
  python 10_whisper_size_comparison.py --mode zeroshot
  python 10_whisper_size_comparison.py --mode finetune
  python 10_whisper_size_comparison.py --mode both     (default)
  python 10_whisper_size_comparison.py --size small    (one size only)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fine_tuning.config import (
    DATASET_DIR, HF_DATA_DIR, LANGUAGE, LM_DIR,
    MODELS_DIR, PLOTS_DIR, RESULTS_DIR, TASK, TRAINING, WHISPER_SIZES,
)

SIZE_DIR    = RESULTS_DIR / "size_comparison"
SAMPLE_RATE = 16_000


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_audio_np(audio_path: str):
    import numpy as np
    from pydub import AudioSegment
    audio = (
        AudioSegment.from_file(audio_path)
        .set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
    )
    return np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32) / 32768.0


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
    return [r for r in rows if Path(str(r["audio_path"])).exists()]


# ── Experiment A: Zero-shot evaluation ────────────────────────────────────────

def run_zeroshot(size_key: str, clips: list[dict]) -> dict:
    cfg = WHISPER_SIZES[size_key]
    out_path = SIZE_DIR / f"{size_key}_zeroshot_predictions.json"
    SIZE_DIR.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"  [CACHED] {out_path.name} already exists. Loading.")
        with open(out_path) as f:
            return json.load(f)

    try:
        import whisper
        import torch
    except ImportError:
        print("[ERROR] pip install openai-whisper torch"); raise SystemExit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Loading whisper/{cfg['model']} ({cfg['params']}) on {device} ...")

    t0    = time.time()
    model = whisper.load_model(cfg["model"], device=device)
    load_time = time.time() - t0

    predictions = []
    total_audio_sec = 0.0
    t_start = time.time()

    for i, clip in enumerate(clips):
        audio_np          = load_audio_np(clip["audio_path"])
        total_audio_sec  += float(clip.get("duration_sec", 0))
        result            = model.transcribe(
            audio_np,
            language="bn",
            task="transcribe",
            verbose=False,
            condition_on_previous_text=False,
        )
        predictions.append({
            "clip_id":          clip.get("clip_id", ""),
            "audio_path":       clip["audio_path"],
            "reference":        str(clip.get("transcript", "")).strip(),
            "hypothesis":       result.get("text", "").strip(),
            "dialect":          clip.get("dialect", "unknown"),
            "domain":           clip.get("domain", "General"),
            "is_code_switched": bool(clip.get("is_code_switched", False)),
            "bn_ratio":         float(clip.get("bn_ratio", 0.0)),
            "en_ratio":         float(clip.get("en_ratio", 0.0)),
            "duration_sec":     float(clip.get("duration_sec", 0.0)),
        })
        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(clips)}]")

    decode_time = time.time() - t_start
    rtf = decode_time / total_audio_sec if total_audio_sec > 0 else 0  # real-time factor

    result_data = {
        "size_key":         size_key,
        "model":            cfg["model"],
        "params":           cfg["params"],
        "mode":             "zeroshot",
        "n_clips":          len(predictions),
        "model_load_sec":   round(load_time, 2),
        "decode_time_sec":  round(decode_time, 2),
        "real_time_factor": round(rtf, 4),
        "predictions":      predictions,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"    RTF={rtf:.3f}  Saved: {out_path}")
    return result_data


# ── Experiment B: Fine-tune each size then evaluate ───────────────────────────

@dataclass
class DataCollatorWhisper:
    processor: Any
    def __call__(self, features):
        import torch
        input_features = []
        for f in features:
            inputs = self.processor.feature_extractor(
                f["audio"]["array"], sampling_rate=16000, return_tensors="pt"
            )
            input_features.append({"input_features": inputs.input_features[0]})
        lbl = [{"input_ids": f["labels"]} for f in features]
        batch  = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        labels = self.processor.tokenizer.pad(lbl, return_tensors="pt")
        ids    = labels["input_ids"].masked_fill(labels.attention_mask.ne(1), -100)
        if (ids[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            ids = ids[:, 1:]
        batch["labels"] = ids
        return batch


def finetune_whisper_size(size_key: str) -> Path:
    """Fine-tune one Whisper size on the 'all' training subset."""
    cfg       = WHISPER_SIZES[size_key]
    model_dir = MODELS_DIR / f"ft_size_{size_key}"

    if model_dir.exists() and any(model_dir.iterdir()):
        print(f"  [CACHED] Model already trained: {model_dir}")
        return model_dir

    try:
        import torch
        from datasets import load_from_disk, Audio
        from transformers import (
            WhisperProcessor, WhisperForConditionalGeneration,
            Seq2SeqTrainer, Seq2SeqTrainingArguments,
        )
        from transformers.trainer_utils import get_last_checkpoint
        import evaluate
    except ImportError:
        print("[ERROR] pip install transformers datasets evaluate torch accelerate")
        raise SystemExit(1)

    data_dir = HF_DATA_DIR / "all"
    if not (data_dir / "train").exists():
        print(f"[ERROR] HF dataset not found at {data_dir}.")
        print("        Run: python fine_tuning/01_prepare_hf_dataset.py")
        raise SystemExit(1)

    print(f"\n  Fine-tuning whisper/{cfg['model']} ({cfg['params']}) ...")
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16  = TRAINING["fp16"] if device == "cuda" else False
    processor = WhisperProcessor.from_pretrained(cfg["hf_id"], language=LANGUAGE, task=TASK)
    model     = WhisperForConditionalGeneration.from_pretrained(cfg["hf_id"])
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=LANGUAGE, task=TASK
    )
    model.config.suppress_tokens = []

    train_ds = load_from_disk(str(data_dir / "train"))
    dev_ds   = load_from_disk(str(data_dir / "dev"))
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    dev_ds   = dev_ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    print(f"    Train: {len(train_ds)}  Dev: {len(dev_ds)}")

    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        p = processor.tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)
        l = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": round(wer_metric.compute(predictions=p, references=l), 4)}

    model_dir.mkdir(parents=True, exist_ok=True)
    args = Seq2SeqTrainingArguments(
        output_dir                  = str(model_dir),
        num_train_epochs            = TRAINING["num_train_epochs"],
        per_device_train_batch_size = TRAINING["per_device_train_batch_size"],
        per_device_eval_batch_size  = TRAINING["per_device_eval_batch_size"],
        gradient_accumulation_steps = TRAINING["gradient_accumulation_steps"],
        learning_rate               = TRAINING["learning_rate"],
        warmup_steps                = TRAINING["warmup_steps"],
        fp16                        = use_fp16,
        predict_with_generate       = True,
        generation_max_length       = 225,
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
        remove_unused_columns       = False,
    )

    trainer = Seq2SeqTrainer(
        model           = model,
        args            = args,
        train_dataset   = train_ds,
        eval_dataset    = dev_ds,
        data_collator   = DataCollatorWhisper(processor=processor),
        compute_metrics = compute_metrics,
    )

    last_checkpoint = get_last_checkpoint(str(model_dir))
    if last_checkpoint is not None:
        print(f"    Resuming from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()
        
    trainer.save_model(str(model_dir))
    processor.save_pretrained(str(model_dir))
    print(f"    Saved: {model_dir}")
    return model_dir


def run_finetuned_eval(size_key: str, model_dir: Path, clips: list[dict]) -> dict:
    cfg      = WHISPER_SIZES[size_key]
    out_path = SIZE_DIR / f"{size_key}_finetuned_predictions.json"
    SIZE_DIR.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        print(f"  [CACHED] {out_path.name} already exists. Loading.")
        with open(out_path) as f:
            return json.load(f)

    try:
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
    except ImportError:
        print("[ERROR] pip install transformers torch"); raise SystemExit(1)

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    processor = WhisperProcessor.from_pretrained(str(model_dir))
    model     = WhisperForConditionalGeneration.from_pretrained(str(model_dir)).to(device)
    model.eval()
    # Force language via generation_config (forced_decoder_ids deprecated in v5)
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK

    predictions = []
    total_audio_sec = 0.0
    t_start = time.time()

    for i, clip in enumerate(clips):
        audio_np         = load_audio_np(clip["audio_path"])
        total_audio_sec += float(clip.get("duration_sec", 0))
        inputs           = processor(
            audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt"
        ).input_features.to(device)
        with torch.no_grad():
            ids = model.generate(inputs)
        pred_text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

        predictions.append({
            "clip_id":          clip.get("clip_id", ""),
            "audio_path":       clip["audio_path"],
            "reference":        str(clip.get("transcript", "")).strip(),
            "hypothesis":       pred_text,
            "dialect":          clip.get("dialect", "unknown"),
            "domain":           clip.get("domain", "General"),
            "is_code_switched": bool(clip.get("is_code_switched", False)),
            "bn_ratio":         float(clip.get("bn_ratio", 0.0)),
            "en_ratio":         float(clip.get("en_ratio", 0.0)),
            "duration_sec":     float(clip.get("duration_sec", 0.0)),
        })
        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(clips)}]")

    decode_time = time.time() - t_start
    rtf = decode_time / total_audio_sec if total_audio_sec > 0 else 0

    result_data = {
        "size_key":         size_key,
        "model":            cfg["model"],
        "params":           cfg["params"],
        "mode":             "finetuned",
        "n_clips":          len(predictions),
        "decode_time_sec":  round(decode_time, 2),
        "real_time_factor": round(rtf, 4),
        "predictions":      predictions,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)
    print(f"    RTF={rtf:.3f}  Saved: {out_path}")
    return result_data


# ── Compute WER/CER for a result_data dict ────────────────────────────────────

def compute_wer_cer(result_data: dict) -> dict:
    try:
        from jiwer import wer as jiwer_wer, cer as jiwer_cer
    except ImportError:
        return {}

    preds  = result_data.get("predictions", [])
    refs   = [p["reference"]  for p in preds if p["reference"].strip()]
    hyps   = [p["hypothesis"] for p in preds if p["reference"].strip()]
    if not refs:
        return {}

    cs_preds = [p for p in preds if p.get("is_code_switched") and p["reference"].strip()]
    refs_cs  = [p["reference"]  for p in cs_preds]
    hyps_cs  = [p["hypothesis"] for p in cs_preds]

    return {
        "wer":        round(jiwer_wer(refs, hyps), 4),
        "cer":        round(jiwer_cer(refs, hyps), 4),
        "wer_cs":     round(jiwer_wer(refs_cs, hyps_cs), 4) if refs_cs else None,
        "n_clips":    len(refs),
        "n_cs_clips": len(cs_preds),
        "rtf":        result_data.get("real_time_factor"),
        "params":     result_data.get("params"),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_size_comparison(summary: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [SKIP] pip install matplotlib")
        return

    sizes     = list(WHISPER_SIZES.keys())
    params    = [int(WHISPER_SIZES[s]["params"].replace("M", "")) for s in sizes]
    zs_wers   = [summary.get(f"{s}_zeroshot",  {}).get("wer")  for s in sizes]
    ft_wers   = [summary.get(f"{s}_finetuned", {}).get("wer")  for s in sizes]
    zs_cers   = [summary.get(f"{s}_zeroshot",  {}).get("cer")  for s in sizes]
    ft_cers   = [summary.get(f"{s}_finetuned", {}).get("cer")  for s in sizes]
    rtfs      = [summary.get(f"{s}_finetuned", {}).get("rtf")
                 or summary.get(f"{s}_zeroshot", {}).get("rtf") for s in sizes]

    labels = [WHISPER_SIZES[s]["model"] for s in sizes]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: WER vs model size
    x = np.arange(len(sizes))
    w = 0.35
    ax = axes[0]
    zs_vals = [v * 100 if v else 0 for v in zs_wers]
    ft_vals = [v * 100 if v else 0 for v in ft_wers]
    b1 = ax.bar(x - w/2, zs_vals, w, label="Zero-shot",   color="#9E9E9E", alpha=0.85)
    b2 = ax.bar(x + w/2, ft_vals, w, label="Fine-tuned",  color="#1565C0", alpha=0.85)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.3,
                        f"{h:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("WER (%)")
    ax.set_title("WER vs Whisper Model Size\n(lower is better)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Plot 2: CER vs model size
    ax = axes[1]
    zs_c = [v * 100 if v else 0 for v in zs_cers]
    ft_c = [v * 100 if v else 0 for v in ft_cers]
    ax.bar(x - w/2, zs_c, w, label="Zero-shot",  color="#9E9E9E", alpha=0.85)
    ax.bar(x + w/2, ft_c, w, label="Fine-tuned", color="#E65100", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("CER (%)")
    ax.set_title("CER vs Whisper Model Size\n(lower is better)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Plot 3: WER vs parameter count (scatter — accuracy vs efficiency)
    ax = axes[2]
    valid_zs = [(p, w*100) for p, w in zip(params, zs_wers) if w]
    valid_ft = [(p, w*100) for p, w in zip(params, ft_wers)  if w]
    if valid_zs:
        xs, ys = zip(*valid_zs)
        ax.scatter(xs, ys, s=100, color="#9E9E9E", label="Zero-shot", zorder=3)
        ax.plot(xs, ys, color="#9E9E9E", alpha=0.5)
    if valid_ft:
        xs, ys = zip(*valid_ft)
        ax.scatter(xs, ys, s=100, color="#1565C0", label="Fine-tuned", zorder=3)
        ax.plot(xs, ys, color="#1565C0", alpha=0.5)
        for xi, yi, lbl in zip(xs, ys, labels):
            ax.annotate(lbl, (xi, yi), textcoords="offset points",
                        xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Model parameters (M)")
    ax.set_ylabel("WER (%)")
    ax.set_title("Accuracy vs Model Size\n(efficiency frontier)")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(PLOTS_DIR / "whisper_size_comparison.pdf"), bbox_inches="tight", dpi=300)
    plt.savefig(str(PLOTS_DIR / "whisper_size_comparison.png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Plot saved: {PLOTS_DIR / 'whisper_size_comparison.pdf'}")


# ── Driver ────────────────────────────────────────────────────────────────────

def main(mode: str = "both", size_key: str | None = None):
    SIZE_DIR.mkdir(parents=True, exist_ok=True)
    clips   = load_test_manifest()
    sizes   = [size_key] if size_key else list(WHISPER_SIZES.keys())
    summary = {}

    for sk in sizes:
        cfg = WHISPER_SIZES[sk]
        print(f"\n{'='*60}")
        print(f"Whisper size: {cfg['model']}  ({cfg['params']} params)")
        print(f"{'='*60}")

        if mode in ("zeroshot", "both"):
            print(f"  [Zero-shot]")
            zs_data    = run_zeroshot(sk, clips)
            zs_metrics = compute_wer_cer(zs_data)
            summary[f"{sk}_zeroshot"] = {**zs_metrics, "mode": "zeroshot", "model": cfg["model"]}
            print(f"    WER={zs_metrics.get('wer')}  CER={zs_metrics.get('cer')}")

        if mode in ("finetune", "both"):
            print(f"  [Fine-tuned]")
            model_dir  = finetune_whisper_size(sk)
            ft_data    = run_finetuned_eval(sk, model_dir, clips)
            ft_metrics = compute_wer_cer(ft_data)
            summary[f"{sk}_finetuned"] = {**ft_metrics, "mode": "finetuned", "model": cfg["model"]}
            print(f"    WER={ft_metrics.get('wer')}  CER={ft_metrics.get('cer')}")

    # Save summary
    summary_path = SIZE_DIR / "size_comparison_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved: {summary_path}")

    # Print table
    print(f"\n{'Model':<14} {'Params':>7} {'WER-ZS':>8} {'WER-FT':>8} {'CER-ZS':>8} {'CER-FT':>8} {'RTF':>6}")
    print("-" * 65)
    for sk in sizes:
        cfg = WHISPER_SIZES[sk]
        zs  = summary.get(f"{sk}_zeroshot",  {})
        ft  = summary.get(f"{sk}_finetuned", {})
        wz  = f"{zs.get('wer', 0)*100:.1f}%" if zs.get("wer") else "—"
        wf  = f"{ft.get('wer', 0)*100:.1f}%" if ft.get("wer") else "—"
        cz  = f"{zs.get('cer', 0)*100:.1f}%" if zs.get("cer") else "—"
        cf  = f"{ft.get('cer', 0)*100:.1f}%" if ft.get("cer") else "—"
        rtf = f"{ft.get('rtf') or zs.get('rtf') or 0:.3f}"
        print(f"{cfg['model']:<14} {cfg['params']:>7} {wz:>8} {wf:>8} {cz:>8} {cf:>8} {rtf:>6}")

    # Plot
    plot_size_comparison(summary)
    print("\nNext: python 03_compute_metrics.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["zeroshot", "finetune", "both"],
        default="both",
        help="Which experiment to run (default: both)",
    )
    parser.add_argument(
        "--size",
        choices=list(WHISPER_SIZES.keys()),
        default=None,
        help="Run one model size only (default: all sizes)",
    )
    args = parser.parse_args()
    main(mode=args.mode, size_key=args.size)
