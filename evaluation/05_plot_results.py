"""
Evaluation Step 5: Generate all plots for the paper.

Produces (saved to evaluation/plots/):
  1. wer_comparison.pdf/png       — bar chart: all models vs WER (overall, BN, EN, CS)
  2. dialect_wer_heatmap.pdf/png  — heatmap: model × dialect WER
  3. cs_f1_comparison.pdf/png     — bar chart: CS detection F1 per model
  4. cs_vs_bn_wer.pdf/png         — grouped bars: WER on CS clips vs BN_ONLY clips
  5. dialect_distribution.pdf/png — pie/bar: clip counts per dialect in dataset

Usage:
    python 05_plot_results.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fine_tuning.config import PLOTS_DIR, RESULTS_DIR, TABLE_ORDER, MODEL_LABELS, MODEL_COLORS

# Short multi-line labels for plot axes (derived from MODEL_LABELS)
MODEL_SHORT = {k: v.replace(" (", "\n(").replace(", ", ",\n") for k, v in MODEL_LABELS.items()}


def load_summary() -> dict:
    path = RESULTS_DIR / "all_metrics_summary.json"
    if not path.exists():
        print(f"[ERROR] {path} not found. Run 03_compute_metrics.py first.")
        raise SystemExit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def setup_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend — safe on all platforms
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        print("[ERROR] pip install matplotlib")
        raise SystemExit(1)
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "axes.titlesize":   13,
        "axes.labelsize":   11,
        "figure.dpi":       150,
        "savefig.bbox":     "tight",
        "savefig.dpi":      300,
    })
    return plt


def save_fig(plt, name: str):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = PLOTS_DIR / f"{name}.pdf"
    png_path = PLOTS_DIR / f"{name}.png"
    plt.savefig(str(pdf_path))
    plt.savefig(str(png_path))
    print(f"  Saved: {pdf_path}")
    plt.close()


# ── Plot 1: WER comparison bar chart ─────────────────────────────────────────

def plot_wer_comparison(summary: dict):
    plt = setup_matplotlib()
    import numpy as np

    keys    = [k for k in TABLE_ORDER if k in summary]
    labels  = [MODEL_SHORT[k] for k in keys]
    colors  = [MODEL_COLORS[k] for k in keys]

    metrics_to_plot = [
        ("wer_overall",  "WER Overall"),
        ("wer_cs_subset","WER (CS clips)"),
        ("wer_bn_words", "WER-BN (words)"),
        ("wer_en_words", "WER-EN (words)"),
    ]

    x      = np.arange(len(keys))
    width  = 0.20
    n_bars = len(metrics_to_plot)
    offsets = np.linspace(-(n_bars - 1) * width / 2,
                           (n_bars - 1) * width / 2, n_bars)

    fig, ax = plt.subplots(figsize=(14, 6))

    bar_colors = ["#1565C0", "#42A5F5", "#EF6C00", "#FFA726"]
    for i, (metric_key, metric_label) in enumerate(metrics_to_plot):
        vals = [
            summary[k].get(metric_key) or 0.0
            for k in keys
        ]
        vals_pct = [v * 100 for v in vals]
        bars = ax.bar(
            x + offsets[i], vals_pct, width,
            label=metric_label,
            color=bar_colors[i],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
        # Value labels on bars
        for bar, v in zip(bars, vals_pct):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{v:.1f}",
                ha="center", va="bottom",
                fontsize=7, rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Word Error Rate (%)")
    ax.set_title("ASR Performance: WER Comparison Across Models\n"
                 "(Bengali Dialect + Code-Switching Test Set)")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, min(110, ax.get_ylim()[1] * 1.15))
    ax.grid(axis="y", alpha=0.3)

    # Shade the proposed system column
    if "ft_whisper_all" in keys:
        idx = keys.index("ft_whisper_all")
        ax.axvspan(idx - 0.4, idx + 0.4, alpha=0.07, color="green")

    save_fig(plt, "wer_comparison")


# ── Plot 2: Per-dialect WER heatmap ─────────────────────────────────────────

def plot_dialect_heatmap(summary: dict):
    plt = setup_matplotlib()
    import numpy as np

    keys = [k for k in TABLE_ORDER if k in summary]

    # Collect all dialects
    all_dialects = set()
    for k in keys:
        all_dialects.update(summary[k].get("per_dialect_wer", {}).keys())
    dialects = sorted(all_dialects)

    data = np.zeros((len(keys), len(dialects)))
    for i, k in enumerate(keys):
        pdw = summary[k].get("per_dialect_wer", {})
        for j, d in enumerate(dialects):
            wer = pdw.get(d, {}).get("wer")
            data[i, j] = wer * 100 if wer is not None else float("nan")

    fig, ax = plt.subplots(
        figsize=(max(10, len(dialects) * 0.9), max(5, len(keys) * 0.8))
    )

    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)
    fig.colorbar(im, ax=ax, label="WER (%)")

    ax.set_xticks(range(len(dialects)))
    ax.set_xticklabels(dialects, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([MODEL_SHORT[k].replace("\n", " ") for k in keys], fontsize=9)

    # Annotate cells
    for i in range(len(keys)):
        for j in range(len(dialects)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                        fontsize=8, color="black")

    ax.set_title("Per-Dialect WER (%) — Model × Dialect\n(lower = better)")
    plt.tight_layout()
    save_fig(plt, "dialect_wer_heatmap")


# ── Plot 3: CS detection F1 ───────────────────────────────────────────────────

def plot_cs_f1(summary: dict):
    plt = setup_matplotlib()
    import numpy as np

    keys   = [k for k in TABLE_ORDER if k in summary]
    labels = [MODEL_SHORT[k] for k in keys]
    colors = [MODEL_COLORS[k] for k in keys]

    precision = [summary[k].get("cs_detection", {}).get("cs_precision", 0) for k in keys]
    recall    = [summary[k].get("cs_detection", {}).get("cs_recall",    0) for k in keys]
    f1        = [summary[k].get("cs_detection", {}).get("cs_f1",        0) for k in keys]

    x     = np.arange(len(keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - width, [v * 100 for v in precision], width, label="Precision", color="#1565C0", alpha=0.85)
    b2 = ax.bar(x,          [v * 100 for v in recall],   width, label="Recall",    color="#42A5F5", alpha=0.85)
    b3 = ax.bar(x + width,  [v * 100 for v in f1],       width, label="F1",        color="#2E7D32", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score (%)")
    ax.set_title("Code-Switching Detection: Precision / Recall / F1 per Model")
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    save_fig(plt, "cs_f1_comparison")


# ── Plot 4: CS vs BN WER ─────────────────────────────────────────────────────

def plot_cs_vs_bn_wer(summary: dict):
    plt = setup_matplotlib()
    import numpy as np

    keys    = [k for k in TABLE_ORDER if k in summary]
    labels  = [MODEL_SHORT[k] for k in keys]

    wer_cs = [summary[k].get("wer_cs_subset") or 0.0 for k in keys]
    wer_bn = [summary[k].get("wer_bn_subset") or 0.0 for k in keys]

    x     = np.arange(len(keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    b1 = ax.bar(x - width / 2, [v * 100 for v in wer_cs], width,
                label="WER — CS clips (dialect+English)", color="#EF6C00", alpha=0.85)
    b2 = ax.bar(x + width / 2, [v * 100 for v in wer_bn], width,
                label="WER — BN_ONLY clips (dialect only)", color="#42A5F5", alpha=0.85)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                    f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("WER (%)")
    ax.set_title("WER on Code-Switched vs Bengali-Only Clips per Model\n"
                 "(Shows extra difficulty of code-switching on top of dialect)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_fig(plt, "cs_vs_bn_wer")


# ── Plot 5: Dataset dialect distribution ─────────────────────────────────────

def plot_dataset_distribution():
    """Plot dialect distribution from the segments manifest."""
    plt = setup_matplotlib()

    manifest_path = (
        Path(__file__).resolve().parent.parent
        / "dataset_pipeline" / "segments" / "segments_manifest.json"
    )
    if not manifest_path.exists():
        print(f"  [SKIP] Manifest not found: {manifest_path}")
        return

    with open(manifest_path, encoding="utf-8") as f:
        clips = json.load(f)

    from collections import Counter
    dialect_counts = Counter(c.get("dialect", "unknown") for c in clips)
    dialect_hrs    = {}
    for c in clips:
        d = c.get("dialect", "unknown")
        dialect_hrs[d] = dialect_hrs.get(d, 0.0) + c.get("duration_sec", 0.0) / 3600

    dialects = sorted(dialect_counts, key=lambda d: -dialect_counts[d])
    counts   = [dialect_counts[d] for d in dialects]
    hours    = [dialect_hrs[d] for d in dialects]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Clip count bar chart
    bars = ax1.bar(range(len(dialects)), counts, color="#1565C0", alpha=0.85)
    ax1.set_xticks(range(len(dialects)))
    ax1.set_xticklabels(dialects, rotation=45, ha="right")
    ax1.set_ylabel("Number of clips")
    ax1.set_title("Clips per Dialect")
    for bar, v in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                 str(v), ha="center", fontsize=8)

    # Hours bar chart
    bars2 = ax2.bar(range(len(dialects)), hours, color="#2E7D32", alpha=0.85)
    ax2.set_xticks(range(len(dialects)))
    ax2.set_xticklabels(dialects, rotation=45, ha="right")
    ax2.set_ylabel("Duration (hours)")
    ax2.set_title("Duration per Dialect (hours)")
    for bar, v in zip(bars2, hours):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                 f"{v:.1f}h", ha="center", fontsize=8)

    fig.suptitle(
        f"Bengali Dialect CS Dataset — {sum(counts):,} clips / "
        f"{sum(hours):.1f} hrs across {len(dialects)} dialects",
        fontsize=13
    )
    plt.tight_layout()
    save_fig(plt, "dialect_distribution")


# ── Driver ────────────────────────────────────────────────────────────────────

def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()

    print("\nGenerating plots ...")

    print("1. WER comparison bar chart")
    plot_wer_comparison(summary)

    print("2. Per-dialect WER heatmap")
    plot_dialect_heatmap(summary)

    print("3. CS detection F1")
    plot_cs_f1(summary)

    print("4. CS vs BN_ONLY WER")
    plot_cs_vs_bn_wer(summary)

    print("5. Dataset dialect distribution")
    plot_dataset_distribution()

    print(f"\nAll plots saved to: {PLOTS_DIR}")
    print("Next: python 06_statistical_analysis.py")


if __name__ == "__main__":
    main()
