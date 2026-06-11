"""
Evaluation Step 14: Plot training curves from HuggingFace trainer_state.json.

Reads the trainer_state.json saved inside each fine-tuned model's checkpoints
and plots training loss and development WER vs. training step for all fine-tuned
models on shared axes. Useful as a convergence figure for the paper.

Note: the eval_wer logged during training is the RAW (un-normalised) WER and is
higher than the final corrected WER from 03_compute_metrics.py (which applies
surface normalisation + repetition collapse). It is shown here only to
illustrate convergence, not as the reported accuracy.

Output: evaluation/plots/training_curves.pdf (+ .png)

Usage:
    python 14_plot_training_curves.py
    # if checkpoints live elsewhere (e.g. a backup folder), point --search at it:
    python 14_plot_training_curves.py --search "fine_tuning/models" "fine_tuning/models/_messy_download_backup"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from fine_tuning.config import MODELS_DIR, PLOTS_DIR, MODEL_LABELS, MODEL_COLORS

# Fine-tuned model keys we expect training logs for.
TRAIN_KEYS = ["ft_whisper_all", "ft_whisper_bn_only", "ft_whisper_cs_only", "ft_wav2vec2_all"]


def find_states(search_roots: list[Path]) -> dict[str, Path]:
    """For each known model key, find the trainer_state.json with the most steps
    (the latest checkpoint, whose log_history is cumulative)."""
    best: dict[str, tuple[int, Path]] = {}
    for root in search_roots:
        if not root.exists():
            continue
        for ts in root.rglob("trainer_state.json"):
            path_str = str(ts).replace("\\", "/")
            key = next((k for k in TRAIN_KEYS if k in path_str), None)
            if key is None:
                continue
            try:
                step = json.load(open(ts, encoding="utf-8")).get("global_step", 0)
            except Exception:
                continue
            if key not in best or step > best[key][0]:
                best[key] = (step, ts)
    return {k: v[1] for k, v in best.items()}


def parse_curve(path: Path):
    """Return (train_steps, train_loss, eval_steps, eval_wer)."""
    log = json.load(open(path, encoding="utf-8")).get("log_history", [])
    ts, tl, es, ew = [], [], [], []
    for e in log:
        if "loss" in e and "eval_loss" not in e:
            ts.append(e["step"]); tl.append(e["loss"])
        if "eval_wer" in e:
            es.append(e["step"]); ew.append(e["eval_wer"])
    return ts, tl, es, ew


def main(search_roots):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[ERROR] pip install matplotlib"); raise SystemExit(1)

    states = find_states([Path(r) for r in search_roots])
    if not states:
        print("[ERROR] No trainer_state.json found under:", search_roots)
        print("        Point --search at the folder containing your checkpoints.")
        raise SystemExit(1)

    print("Found training logs:")
    for k, p in states.items():
        print(f"  {k}: {p}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    for key in TRAIN_KEYS:
        if key not in states:
            continue
        ts, tl, es, ew = parse_curve(states[key])
        label = MODEL_LABELS.get(key, key)
        color = MODEL_COLORS.get(key)
        if tl:
            ax1.plot(ts, tl, label=label, color=color, linewidth=1.5)
        if ew:
            ax2.plot(es, ew, label=label, color=color, marker="o",
                     markersize=3, linewidth=1.5)

    ax1.set_xlabel("Training step"); ax1.set_ylabel("Training loss")
    ax1.set_title("Training loss"); ax1.grid(True, alpha=0.3); ax1.legend(fontsize=8)
    ax2.set_xlabel("Training step"); ax2.set_ylabel("Dev WER (raw, training-time)")
    ax2.set_title("Development WER during training"); ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = PLOTS_DIR / f"training_curves.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--search", nargs="+",
                   default=[str(MODELS_DIR)],
                   help="Directories searched recursively for trainer_state.json")
    args = p.parse_args()
    main(args.search)
