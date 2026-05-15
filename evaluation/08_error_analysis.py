"""
Evaluation Step 8: Qualitative error analysis for the paper.

Produces:
  - Error type breakdown (substitution / deletion / insertion per model)
  - English word survival rate in hypotheses (are CS words preserved?)
  - Most common error patterns (what Bengali words get confused most?)
  - Sample error table (25 worst clips) for paper appendix
  - Error analysis plots

Output: evaluation/results/error_analysis/

Usage:
    python 08_error_analysis.py
    python 08_error_analysis.py --model ft_whisper_all
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fine_tuning.config import BASELINES, FINETUNED, RESULTS_DIR, TABLE_ORDER
from dataset_pipeline.code_mixing_utils import classify_code_mixing, detect_word_tags

ERROR_DIR = RESULTS_DIR / "error_analysis"


def is_bengali_word(w: str) -> bool:
    return any("\u0980" <= c <= "\u09FF" for c in w)


def is_english_word(w: str) -> bool:
    w = w.strip(".,!?;:\"'()-")
    return bool(w) and w.isascii() and w.isalpha() and len(w) > 1


def per_clip_edits(ref: str, hyp: str) -> dict:
    """
    Compute word-level edit operations for a single clip.
    Returns counts of substitutions, deletions, insertions.
    """
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    n, m = len(ref_words), len(hyp_words)

    if not ref_words:
        return {"sub": 0, "del": 0, "ins": len(hyp_words), "ref_len": 0}

    # DP edit distance with backtrace
    dp = [[(0, "")] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = (i, "D" * i)
    for j in range(m + 1):
        dp[0][j] = (j, "I" * j)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                candidates = [
                    (dp[i - 1][j][0] + 1,     "D"),   # deletion
                    (dp[i][j - 1][0] + 1,     "I"),   # insertion
                    (dp[i - 1][j - 1][0] + 1, "S"),   # substitution
                ]
                best = min(candidates, key=lambda x: x[0])
                dp[i][j] = best

    # Count operations from the alignment (approximate from DP)
    # Faster approximation: use edit distance decomposition
    edit_dist = dp[n][m][0]
    # Approximate breakdown (exact backtrace is expensive at scale)
    matched    = sum(1 for rw, hw in zip(ref_words, hyp_words) if rw == hw)
    subs       = min(n, m) - matched
    deletions  = max(0, n - m)
    insertions = max(0, m - n)

    return {
        "sub":     max(0, subs),
        "del":     deletions,
        "ins":     insertions,
        "ref_len": n,
    }


def english_word_analysis(predictions: list[dict]) -> dict:
    """
    For CS clips: how many English words in the reference survive in the hypothesis?
    Key finding: do models silently delete English CS words?
    """
    cs_preds = [p for p in predictions if p.get("is_code_switched")]
    if not cs_preds:
        return {}

    total_en_ref  = 0
    found_en_hyp  = 0
    deleted_en    = 0
    transliterated = 0   # English word replaced by Bengali characters

    common_en_deletions = Counter()

    for p in cs_preds:
        ref_en_words = [w for w in p["reference"].split()  if is_english_word(w)]
        hyp_en_words = set(w.lower() for w in p["hypothesis"].split() if is_english_word(w))

        total_en_ref += len(ref_en_words)
        for w in ref_en_words:
            if w.lower() in hyp_en_words:
                found_en_hyp += 1
            else:
                deleted_en += 1
                common_en_deletions[w.lower()] += 1
                # Check if model produced Bengali characters where EN word was
                if is_bengali_word(p["hypothesis"]):
                    transliterated += 1

    return {
        "total_en_words_in_ref":   total_en_ref,
        "en_words_preserved":      found_en_hyp,
        "en_words_deleted":        deleted_en,
        "en_word_survival_rate":   round(found_en_hyp / total_en_ref, 4) if total_en_ref else 0,
        "en_deletion_rate":        round(deleted_en   / total_en_ref, 4) if total_en_ref else 0,
        "top_deleted_en_words":    common_en_deletions.most_common(20),
    }


def most_common_substitutions(predictions: list[dict], top_n: int = 20) -> list[tuple]:
    """Find the most common (reference_word → hypothesis_word) substitution pairs."""
    sub_counter = Counter()
    for p in predictions:
        ref_words = p["reference"].split()
        hyp_words = p["hypothesis"].split()
        # Align by position (approximate — exact alignment needs edit backtrace)
        for rw, hw in zip(ref_words, hyp_words):
            if rw != hw and rw.strip() and hw.strip():
                sub_counter[(rw, hw)] += 1
    return sub_counter.most_common(top_n)


def worst_clips(predictions: list[dict], n: int = 25) -> list[dict]:
    """Return the N clips with highest WER for paper appendix / error inspection."""
    def clip_wer(p):
        ref = p["reference"].strip().split()
        hyp = p["hypothesis"].strip().split()
        if not ref:
            return 0.0
        edits = per_clip_edits(p["reference"], p["hypothesis"])
        return (edits["sub"] + edits["del"] + edits["ins"]) / len(ref)

    scored = [(clip_wer(p), p) for p in predictions if p["reference"].strip()]
    scored.sort(key=lambda x: -x[0])
    return [
        {
            "clip_id":          p["clip_id"],
            "dialect":          p["dialect"],
            "is_cs":            p.get("is_code_switched", False),
            "wer":              round(wer, 4),
            "reference":        p["reference"],
            "hypothesis":       p["hypothesis"],
        }
        for wer, p in scored[:n]
    ]


def run_error_analysis(model_key: str):
    pred_path = RESULTS_DIR / f"{model_key}_predictions.json"
    if not pred_path.exists():
        print(f"  [SKIP] {pred_path} not found.")
        return

    with open(pred_path, encoding="utf-8") as f:
        predictions = json.load(f)

    print(f"\n{'='*60}")
    print(f"Error analysis: {model_key}  ({len(predictions)} clips)")
    print(f"{'='*60}")

    # 1. Error type breakdown
    total_sub = total_del = total_ins = total_ref = 0
    for p in predictions:
        edits       = per_clip_edits(p["reference"], p["hypothesis"])
        total_sub  += edits["sub"]
        total_del  += edits["del"]
        total_ins  += edits["ins"]
        total_ref  += edits["ref_len"]

    error_breakdown = {}
    if total_ref:
        error_breakdown = {
            "substitution_rate": round(total_sub / total_ref, 4),
            "deletion_rate":     round(total_del / total_ref, 4),
            "insertion_rate":    round(total_ins / total_ref, 4),
        }
        print(f"  Substitution rate : {error_breakdown['substitution_rate']:.4f}")
        print(f"  Deletion rate     : {error_breakdown['deletion_rate']:.4f}")
        print(f"  Insertion rate    : {error_breakdown['insertion_rate']:.4f}")

    # 2. English word survival analysis
    en_analysis = english_word_analysis(predictions)
    if en_analysis:
        print(f"\n  English word survival (CS clips only):")
        print(f"    Total EN words in reference : {en_analysis['total_en_words_in_ref']}")
        print(f"    EN words preserved in hyp   : {en_analysis['en_words_preserved']}")
        print(f"    EN word survival rate       : {en_analysis['en_word_survival_rate']:.4f}")
        print(f"    EN word deletion rate       : {en_analysis['en_deletion_rate']:.4f}")
        print(f"    Top deleted EN words        : {en_analysis['top_deleted_en_words'][:5]}")

    # 3. Most common substitutions
    common_subs = most_common_substitutions(predictions, top_n=15)
    print(f"\n  Top substitution errors (ref → hyp):")
    for (ref_w, hyp_w), cnt in common_subs[:10]:
        print(f"    '{ref_w}' → '{hyp_w}' : {cnt}x")

    # 4. Worst clips
    worst = worst_clips(predictions, n=25)

    # Save everything
    ERROR_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "model":              model_key,
        "n_clips":            len(predictions),
        "error_breakdown":    error_breakdown,
        "en_word_analysis":   en_analysis,
        "top_substitutions":  [(f"{r}→{h}", c) for (r, h), c in common_subs],
        "worst_clips":        worst,
    }

    out_path = ERROR_DIR / f"{model_key}_error_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  Saved: {out_path}")

    # Save worst clips as TSV for easy inspection
    tsv_path = ERROR_DIR / f"{model_key}_worst_clips.tsv"
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=list(worst[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(worst)
    print(f"  Worst clips TSV: {tsv_path}")

    return result


def plot_error_comparison(all_results: dict):
    """Bar chart comparing error type breakdown across models."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  [SKIP] matplotlib not installed.")
        return

    models = [k for k in TABLE_ORDER if k in all_results]
    labels = [k.replace("ft_whisper_", "ours_").replace("whisper_large_v3", "whisper_zeroshot") for k in models]

    subs = [all_results[k]["error_breakdown"].get("substitution_rate", 0) * 100 for k in models]
    dels = [all_results[k]["error_breakdown"].get("deletion_rate",     0) * 100 for k in models]
    ins  = [all_results[k]["error_breakdown"].get("insertion_rate",    0) * 100 for k in models]
    surv = [
        all_results[k].get("en_word_analysis", {}).get("en_word_survival_rate", 0) * 100
        for k in models
    ]

    x     = np.arange(len(models))
    width = 0.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    # Error breakdown
    ax1.bar(x - width, subs, width, label="Substitutions", color="#EF5350")
    ax1.bar(x,         dels, width, label="Deletions",     color="#FF7043")
    ax1.bar(x + width, ins,  width, label="Insertions",    color="#FFA726")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Rate (% of reference words)")
    ax1.set_title("Error Type Breakdown per Model")
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # English word survival
    colors = ["#9E9E9E" if "ft" not in m else "#2E7D32" for m in models]
    ax2.bar(x, surv, color=colors, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("English word survival rate (%)")
    ax2.set_title("English CS Word Survival Rate\n(higher = model preserves English words better)")
    ax2.set_ylim(0, 105)
    ax2.axhline(100, color="grey", linestyle="--", alpha=0.4)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plots_dir = Path(__file__).resolve().parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    plt.savefig(str(plots_dir / "error_analysis.pdf"), bbox_inches="tight", dpi=300)
    plt.savefig(str(plots_dir / "error_analysis.png"), bbox_inches="tight", dpi=300)
    plt.close()
    print(f"\nPlot saved: {plots_dir / 'error_analysis.pdf'}")


def main(model_key: str | None = None):
    ERROR_DIR.mkdir(parents=True, exist_ok=True)

    all_keys    = list(BASELINES.keys()) + list(FINETUNED.keys())
    keys        = [model_key] if model_key else all_keys
    all_results = {}

    for key in keys:
        result = run_error_analysis(key)
        if result:
            all_results[key] = result

    if len(all_results) > 1:
        plot_error_comparison(all_results)

    print("\nError analysis complete.")
    print(f"Results in: {ERROR_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Analyse one model only (default: all)")
    args = parser.parse_args()
    main(args.model)
