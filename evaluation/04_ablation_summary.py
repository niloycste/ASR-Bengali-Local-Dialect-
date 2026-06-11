"""
Evaluation Step 4: Build the ablation comparison table for your paper.

Reads:  evaluation/results/all_metrics_summary.json
Writes: evaluation/results/ablation_table.csv
        evaluation/results/ablation_table.tex   (LaTeX table for paper)
        evaluation/results/ablation_table.txt   (plain text for quick reading)

Usage:
    python 04_ablation_summary.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fine_tuning.config import BASELINES, FINETUNED, RESULTS_DIR, TABLE_ORDER, MODEL_LABELS


def fmt(value, pct: bool = True) -> str:
    """Format a metric value for display."""
    if value is None:
        return "—"
    if pct:
        return f"{value * 100:.1f}%"
    return f"{value:.4f}"


def load_summary() -> dict:
    path = RESULTS_DIR / "all_metrics_summary.json"
    if not path.exists():
        print(f"[ERROR] {path} not found. Run 03_compute_metrics.py first.")
        raise SystemExit(1)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_rows(summary: dict) -> list[dict]:
    rows = []
    for key in TABLE_ORDER:
        if key not in summary:
            continue
        m   = summary[key]
        cs  = m.get("cs_detection", {})
        row = {
            "model":         MODEL_LABELS.get(key, key),
            "wer_overall":   m.get("wer_overall"),
            "wer_cs":        m.get("wer_cs_subset"),
            "wer_bn_words":  m.get("wer_bn_words"),
            "wer_en_words":  m.get("wer_en_words"),
            "cs_f1":         cs.get("cs_f1"),
            "n_clips":       m.get("n_clips", 0),
            "n_cs":          m.get("n_cs_clips", 0),
        }
        rows.append(row)
    return rows


def print_plain_table(rows: list[dict]) -> str:
    header = (
        f"{'Model':<45} {'WER':>7} {'WER-CS':>7} "
        f"{'WER-BN':>7} {'WER-EN':>7} {'CS-F1':>6}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in rows:
        lines.append(
            f"{r['model']:<45} "
            f"{fmt(r['wer_overall']):>7} "
            f"{fmt(r['wer_cs']):>7} "
            f"{fmt(r['wer_bn_words']):>7} "
            f"{fmt(r['wer_en_words']):>7} "
            f"{fmt(r['cs_f1']):>6}"
        )
    lines.append(sep)
    lines.append(
        "WER = Word Error Rate (lower is better). "
        "WER-CS = WER on code-switched clips. "
        "WER-BN/EN = WER on Bengali/English words only. "
        "CS-F1 = F1 for detecting code-switched clips."
    )
    return "\n".join(lines)


def build_csv(rows: list[dict]) -> str:
    import csv, io
    buf = io.StringIO()
    cols = ["model", "wer_overall", "wer_cs", "wer_bn_words", "wer_en_words", "cs_f1", "n_clips", "n_cs"]
    writer = csv.DictWriter(buf, fieldnames=cols)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k, "") for k in cols})
    return buf.getvalue()


def build_latex(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{ASR Results on Bengali Dialect Code-Switching Test Set}",
        r"\label{tab:asr_results}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Model & WER$\downarrow$ & WER-CS$\downarrow$ & WER-BN$\downarrow$ "
        r"& WER-EN$\downarrow$ & CS-F1$\uparrow$ \\",
        r"\midrule",
    ]
    for i, r in enumerate(rows):
        # Draw a midrule before fine-tuned models
        if i == 3:
            lines.append(r"\midrule")
        model = r["model"].replace("&", r"\&").replace("_", r"\_")
        if i == len(rows) - 1:
            model = r"\textbf{" + model + r"}"  # bold the proposed system
        line = (
            f"{model} & "
            f"{fmt(r['wer_overall'])} & "
            f"{fmt(r['wer_cs'])} & "
            f"{fmt(r['wer_bn_words'])} & "
            f"{fmt(r['wer_en_words'])} & "
            f"{fmt(r['cs_f1'])} \\\\"
        )
        lines.append(line)
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    rows    = build_rows(summary)

    if not rows:
        print("[WARNING] No model results found in summary. Run 03_compute_metrics.py first.")
        return

    plain = print_plain_table(rows)
    print("\n" + plain + "\n")

    # Save files
    txt_path = RESULTS_DIR / "ablation_table.txt"
    csv_path = RESULTS_DIR / "ablation_table.csv"
    tex_path = RESULTS_DIR / "ablation_table.tex"

    txt_path.write_text(plain, encoding="utf-8")
    csv_path.write_text(build_csv(rows), encoding="utf-8")
    tex_path.write_text(build_latex(rows), encoding="utf-8")

    print(f"Saved: {txt_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {tex_path}  <- paste directly into your LaTeX paper")
    print("\nNext: python 05_plot_results.py")


if __name__ == "__main__":
    main()
