"""
Evaluation Step 3: compute raw and normalization-aware ASR metrics.

Reads:
  evaluation/results/*_predictions.json

Writes:
  evaluation/results/*_metrics.json
  evaluation/results/all_metrics_summary.json

Metrics:
  - WER overall
  - WER on code-switched clips
  - WER on Bengali-only clips
  - WER on Bengali-script words
  - WER on English/Latin words
  - Pronunciation-normalized versions of the WER metrics above
  - CER overall and on code-switched clips
  - MIX-ER
  - CS detection precision / recall / F1
  - Per-dialect WER
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fine_tuning.config import BASELINES, FINETUNED, RESULTS_DIR
from dataset_pipeline.text_normalization import (
    normalize_pronunciation_variants,
    normalize_transcript,
)


def is_bengali_word(word: str) -> bool:
    return any("\u0980" <= c <= "\u09FF" for c in word)


def is_english_word(word: str) -> bool:
    word = word.strip(".,!?;:\"'()-")
    return bool(word) and word.isascii() and word.isalpha() and len(word) > 1


def extract_bn_words(text: str) -> list[str]:
    return [word for word in text.split() if is_bengali_word(word)]


def extract_en_words(text: str) -> list[str]:
    return [word for word in text.split() if is_english_word(word)]


def compute_wer_from_lists(refs: list[str], hyps: list[str]) -> float | None:
    """Compute WER across paired reference/hypothesis lists."""
    try:
        from jiwer import wer as jiwer_wer
    except ImportError:
        print("[ERROR] pip install jiwer")
        raise SystemExit(1)

    paired = [(ref, hyp) for ref, hyp in zip(refs, hyps) if ref.strip()]
    if not paired:
        return None

    refs_final, hyps_final = zip(*paired)
    return round(jiwer_wer(list(refs_final), list(hyps_final)), 4)


def compute_cer_from_lists(refs: list[str], hyps: list[str]) -> float | None:
    """Compute CER across paired reference/hypothesis lists."""
    try:
        from jiwer import cer as jiwer_cer
    except ImportError:
        return None

    paired = [(ref, hyp) for ref, hyp in zip(refs, hyps) if ref.strip()]
    if not paired:
        return None

    refs_final, hyps_final = zip(*paired)
    return round(jiwer_cer(list(refs_final), list(hyps_final)), 4)


def compute_error_breakdown(refs: list[str], hyps: list[str]) -> dict:
    """
    Compute substitution, deletion, and insertion breakdown.

    Also reports English-word deletion rate because English tokens are a common
    failure mode in code-switched ASR.
    """
    try:
        from jiwer import process_words
    except ImportError:
        return {}

    paired = [(ref, hyp) for ref, hyp in zip(refs, hyps) if ref.strip()]
    if not paired:
        return {}

    refs_final = [pair[0] for pair in paired]
    hyps_final = [pair[1] for pair in paired]
    measures = process_words(refs_final, hyps_final)

    total_ref_words = sum(len(ref.split()) for ref in refs_final)
    if total_ref_words == 0:
        return {}

    en_refs: list[str] = []
    en_hyps: list[str] = []
    for ref, hyp in paired:
        ref_en = " ".join(extract_en_words(ref)) or "<empty>"
        hyp_en = " ".join(extract_en_words(hyp)) or "<empty>"
        if ref_en != "<empty>":
            en_refs.append(ref_en)
            en_hyps.append(hyp_en)

    en_deletion_rate = None
    if en_refs:
        en_measures = process_words(en_refs, en_hyps)
        en_total = sum(len(ref.split()) for ref in en_refs)
        en_deletion_rate = round(en_measures.deletions / en_total, 4) if en_total else None

    return {
        "substitution_rate": round(measures.substitutions / total_ref_words, 4),
        "deletion_rate": round(measures.deletions / total_ref_words, 4),
        "insertion_rate": round(measures.insertions / total_ref_words, 4),
        "en_word_deletion_rate": en_deletion_rate,
    }


def compute_mix_er(predictions: list[dict]) -> float | None:
    """
    Compute a simple code-switch aware error rate on CS clips.

    Bengali-word WER and English-word WER are averaged with equal weight.
    """
    cs_preds = [prediction for prediction in predictions if prediction.get("is_code_switched")]
    if not cs_preds:
        return None

    refs_bn = [" ".join(extract_bn_words(prediction["reference"])) for prediction in cs_preds]
    hyps_bn = [" ".join(extract_bn_words(prediction["hypothesis"])) for prediction in cs_preds]
    refs_en = [" ".join(extract_en_words(prediction["reference"])) for prediction in cs_preds]
    hyps_en = [" ".join(extract_en_words(prediction["hypothesis"])) for prediction in cs_preds]

    wer_bn = compute_wer_from_lists(refs_bn, hyps_bn) or 0.0
    wer_en = compute_wer_from_lists(refs_en, hyps_en) or 0.0
    return round((wer_bn + wer_en) / 2, 4)


def compute_wer_bn(predictions: list[dict]) -> float | None:
    """Compute WER using only Bengali-script words."""
    refs: list[str] = []
    hyps: list[str] = []
    for prediction in predictions:
        ref_bn = " ".join(extract_bn_words(prediction["reference"]))
        hyp_bn = " ".join(extract_bn_words(prediction["hypothesis"]))
        if ref_bn.strip():
            refs.append(ref_bn)
            hyps.append(hyp_bn if hyp_bn.strip() else "<empty>")
    return compute_wer_from_lists(refs, hyps)


def compute_wer_en(predictions: list[dict]) -> float | None:
    """Compute WER using only English/Latin words."""
    refs: list[str] = []
    hyps: list[str] = []
    for prediction in predictions:
        ref_en = " ".join(extract_en_words(prediction["reference"]))
        hyp_en = " ".join(extract_en_words(prediction["hypothesis"]))
        if ref_en.strip():
            refs.append(ref_en.lower())
            hyps.append(hyp_en.lower() if hyp_en.strip() else "<empty>")
    return compute_wer_from_lists(refs, hyps)


def surface_normalize_predictions(predictions: list[dict]) -> list[dict]:
    """
    Build a surface-cleaned evaluation view.

    Applies normalize_transcript() (strips punctuation, danda, emojis, ZWNJ and
    normalizes Unicode/whitespace) to BOTH reference and hypothesis. This is the
    standard pre-WER cleaning step — without it, trivial formatting differences
    (e.g. "ভালো।" vs "ভালো") are wrongly counted as word errors and inflate WER.

    The stored prediction files are not modified; this only creates a parallel
    in-memory view used for metric computation.
    """
    cleaned = []
    for prediction in predictions:
        cleaned.append(
            {
                **prediction,
                "reference": normalize_transcript(prediction.get("reference", "")),
                "hypothesis": normalize_transcript(prediction.get("hypothesis", "")),
            }
        )
    return cleaned


def normalize_prediction_texts(predictions: list[dict]) -> list[dict]:
    """
    Build a pronunciation-normalized evaluation view.

    This does not modify stored transcripts. It only creates a parallel view
    where known borrowed-word variants map to a single canonical form.
    Apply on top of surface-normalized predictions to get full normalization.
    """
    normalized_predictions = []
    for prediction in predictions:
        normalized_predictions.append(
            {
                **prediction,
                "reference": normalize_pronunciation_variants(prediction["reference"]),
                "hypothesis": normalize_pronunciation_variants(prediction["hypothesis"]),
            }
        )
    return normalized_predictions


def detect_cs_from_text(text: str) -> bool:
    """Detect code-switching from hypothesis text using script-based heuristics."""
    en_words = extract_en_words(text)
    bn_words = extract_bn_words(text)
    total = len(en_words) + len(bn_words)
    if total < 4:
        return False
    en_ratio = len(en_words) / total
    return len(bn_words) >= 2 and len(en_words) >= 1 and 0.05 <= en_ratio <= 0.80


def compute_cs_f1(predictions: list[dict]) -> dict:
    """
    Compute precision, recall, and F1 for CS detection.

    Reference labels come from the manifest. Predictions are inferred from the
    hypothesis transcript using script analysis.
    """
    tp = fp = fn = tn = 0
    for prediction in predictions:
        ref_cs = bool(prediction["is_code_switched"])
        pred_cs = detect_cs_from_text(prediction["hypothesis"])
        if ref_cs and pred_cs:
            tp += 1
        elif not ref_cs and pred_cs:
            fp += 1
        elif ref_cs and not pred_cs:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        "cs_precision": round(precision, 4),
        "cs_recall": round(recall, 4),
        "cs_f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_per_dialect_wer(predictions: list[dict]) -> dict[str, dict]:
    """Compute per-dialect WER and clip counts."""
    by_dialect: dict[str, list[dict]] = defaultdict(list)
    for prediction in predictions:
        by_dialect[prediction.get("dialect", "unknown")].append(prediction)

    result: dict[str, dict] = {}
    for dialect, dialect_predictions in sorted(by_dialect.items()):
        refs = [prediction["reference"] for prediction in dialect_predictions]
        hyps = [prediction["hypothesis"] for prediction in dialect_predictions]
        result[dialect] = {
            "wer": compute_wer_from_lists(refs, hyps),
            "n_clips": len(dialect_predictions),
        }
    return result


def compute_all_metrics(predictions: list[dict]) -> dict:
    """Compute the full evaluation bundle for one model."""
    # Surface-clean reference and hypothesis before any scoring. All base metrics
    # are computed on this cleaned view; the *_normalized metrics additionally map
    # pronunciation variants on top of it.
    predictions = surface_normalize_predictions(predictions)

    cs_preds = [prediction for prediction in predictions if prediction.get("is_code_switched")]
    bn_preds = [prediction for prediction in predictions if not prediction.get("is_code_switched")]

    normalized_preds = normalize_prediction_texts(predictions)
    normalized_cs_preds = [
        prediction for prediction in normalized_preds if prediction.get("is_code_switched")
    ]
    normalized_bn_preds = [
        prediction for prediction in normalized_preds if not prediction.get("is_code_switched")
    ]

    refs_all = [prediction["reference"] for prediction in predictions]
    hyps_all = [prediction["hypothesis"] for prediction in predictions]
    refs_cs = [prediction["reference"] for prediction in cs_preds]
    hyps_cs = [prediction["hypothesis"] for prediction in cs_preds]
    refs_bn = [prediction["reference"] for prediction in bn_preds]
    hyps_bn = [prediction["hypothesis"] for prediction in bn_preds]

    refs_all_norm = [prediction["reference"] for prediction in normalized_preds]
    hyps_all_norm = [prediction["hypothesis"] for prediction in normalized_preds]
    refs_cs_norm = [prediction["reference"] for prediction in normalized_cs_preds]
    hyps_cs_norm = [prediction["hypothesis"] for prediction in normalized_cs_preds]
    refs_bn_norm = [prediction["reference"] for prediction in normalized_bn_preds]
    hyps_bn_norm = [prediction["hypothesis"] for prediction in normalized_bn_preds]

    return {
        "n_clips": len(predictions),
        "n_cs_clips": len(cs_preds),
        "n_bn_clips": len(bn_preds),
        "wer_overall": compute_wer_from_lists(refs_all, hyps_all),
        "wer_cs_subset": compute_wer_from_lists(refs_cs, hyps_cs),
        "wer_bn_subset": compute_wer_from_lists(refs_bn, hyps_bn),
        "wer_bn_words": compute_wer_bn(predictions),
        "wer_en_words": compute_wer_en(predictions),
        "wer_overall_normalized": compute_wer_from_lists(refs_all_norm, hyps_all_norm),
        "wer_cs_subset_normalized": compute_wer_from_lists(refs_cs_norm, hyps_cs_norm),
        "wer_bn_subset_normalized": compute_wer_from_lists(refs_bn_norm, hyps_bn_norm),
        "wer_bn_words_normalized": compute_wer_bn(normalized_preds),
        "wer_en_words_normalized": compute_wer_en(normalized_preds),
        "cer_overall": compute_cer_from_lists(refs_all, hyps_all),
        "cer_cs_subset": compute_cer_from_lists(refs_cs, hyps_cs),
        "mix_er": compute_mix_er(predictions),
        "error_breakdown": compute_error_breakdown(refs_all, hyps_all),
        "cs_detection": compute_cs_f1(predictions),
        "per_dialect_wer": compute_per_dialect_wer(predictions),
    }


def fmt_metric(value: float | None) -> str:
    return f"{value:.4f}" if isinstance(value, (int, float)) else "None"


def process_model(model_key: str):
    pred_path = RESULTS_DIR / f"{model_key}_predictions.json"
    if not pred_path.exists():
        print(f"  [SKIP] {pred_path} not found.")
        return None

    with open(pred_path, encoding="utf-8") as file:
        predictions = json.load(file)

    print(f"\n{'=' * 60}")
    print(f"Computing metrics: {model_key}  ({len(predictions)} clips)")
    print(f"{'=' * 60}")

    metrics = compute_all_metrics(predictions)

    print(f"  WER overall      : {fmt_metric(metrics['wer_overall'])}")
    print(f"  WER norm         : {fmt_metric(metrics['wer_overall_normalized'])}")
    print(f"  WER (CS clips)   : {fmt_metric(metrics['wer_cs_subset'])}")
    print(f"  WER CS norm      : {fmt_metric(metrics['wer_cs_subset_normalized'])}")
    print(f"  WER (BN clips)   : {fmt_metric(metrics['wer_bn_subset'])}")
    print(f"  WER BN norm      : {fmt_metric(metrics['wer_bn_subset_normalized'])}")
    print(f"  WER-BN (words)   : {fmt_metric(metrics['wer_bn_words'])}")
    print(f"  WER-BN norm      : {fmt_metric(metrics['wer_bn_words_normalized'])}")
    print(f"  WER-EN (words)   : {fmt_metric(metrics['wer_en_words'])}")
    print(f"  WER-EN norm      : {fmt_metric(metrics['wer_en_words_normalized'])}")
    print(f"  CER overall      : {fmt_metric(metrics['cer_overall'])}")
    print(f"  CER (CS clips)   : {fmt_metric(metrics['cer_cs_subset'])}")
    print(f"  MIX-ER           : {fmt_metric(metrics['mix_er'])}")

    error_breakdown = metrics.get("error_breakdown", {})
    if error_breakdown:
        print(f"  Substitutions    : {fmt_metric(error_breakdown.get('substitution_rate'))}")
        print(f"  Deletions        : {fmt_metric(error_breakdown.get('deletion_rate'))}")
        print(f"  Insertions       : {fmt_metric(error_breakdown.get('insertion_rate'))}")
        print(f"  EN word deletions: {fmt_metric(error_breakdown.get('en_word_deletion_rate'))}")

    cs_detection = metrics["cs_detection"]
    print(
        "  CS F1            : "
        f"{cs_detection['cs_f1']:.4f}  "
        f"(P={cs_detection['cs_precision']:.3f} R={cs_detection['cs_recall']:.3f})"
    )
    print("  Per-dialect WER:")
    for dialect, dialect_metrics in metrics["per_dialect_wer"].items():
        print(f"    {dialect:<18}: {dialect_metrics['wer']}  (n={dialect_metrics['n_clips']})")

    out_path = RESULTS_DIR / f"{model_key}_metrics.json"
    with open(out_path, "w", encoding="utf-8") as file:
        json.dump({"model": model_key, **metrics}, file, ensure_ascii=False, indent=2)
    print(f"\n  Metrics saved: {out_path}")
    return metrics


def main(model_key: str | None = None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_keys = list(BASELINES.keys()) + list(FINETUNED.keys())
    keys = [model_key] if model_key else all_keys

    summary = {}
    for key in keys:
        metrics = process_model(key)
        if metrics is not None:
            summary[key] = metrics

    summary_path = RESULTS_DIR / "all_metrics_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"\nSummary saved: {summary_path}")
    print("Next: python 04_ablation_summary.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=list(BASELINES.keys()) + list(FINETUNED.keys()),
        default=None,
    )
    args = parser.parse_args()
    main(args.model)
