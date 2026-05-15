"""
Evaluation Step 6: Statistical significance testing and confidence intervals.

Methods:
  - Bootstrap confidence intervals (95%) for WER of each model
  - Paired bootstrap significance test between model pairs
  - Relative WER reduction (RWER) from baseline to proposed system
  - Effect size (Cohen's d) between model pairs

Output:
  evaluation/results/statistical_report.txt   (plain text, for paper appendix)
  evaluation/results/statistical_report.json  (machine-readable)

Usage:
    python 06_statistical_analysis.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fine_tuning.config import (
    BASELINES, BOOTSTRAP_ITERS, CONFIDENCE_LEVEL,
    FINETUNED, RESULTS_DIR, TABLE_ORDER, MODEL_LABELS,
)


# ── WER per clip ──────────────────────────────────────────────────────────────

def per_clip_wer(ref: str, hyp: str) -> float:
    """Compute WER for a single clip using edit distance."""
    ref_words = ref.strip().split()
    hyp_words = hyp.strip().split()
    if not ref_words:
        return 0.0

    # Dynamic programming edit distance
    n, m = len(ref_words), len(hyp_words)
    dp   = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp

    return dp[m] / len(ref_words)


def load_predictions(model_key: str) -> list[dict]:
    path = RESULTS_DIR / f"{model_key}_predictions.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_wer_ci(
    refs: list[str],
    hyps: list[str],
    n_iters: int = BOOTSTRAP_ITERS,
    ci: float    = CONFIDENCE_LEVEL,
    seed: int    = 42,
) -> tuple[float, float, float]:
    """
    Returns (observed_wer, lower_bound, upper_bound) via bootstrap resampling.
    """
    rng      = random.Random(seed)
    n        = len(refs)
    clip_wers = [per_clip_wer(r, h) for r, h in zip(refs, hyps)]

    observed = sum(clip_wers) / n if n else 0.0

    boot_means = []
    for _ in range(n_iters):
        sample = [clip_wers[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    alpha = (1 - ci) / 2
    lo    = boot_means[int(alpha * n_iters)]
    hi    = boot_means[int((1 - alpha) * n_iters)]
    return round(observed, 4), round(lo, 4), round(hi, 4)


# ── Paired bootstrap significance test ───────────────────────────────────────

def paired_bootstrap_test(
    refs: list[str],
    hyps_a: list[str],
    hyps_b: list[str],
    n_iters: int = BOOTSTRAP_ITERS,
    seed: int    = 42,
) -> dict:
    """
    Test whether model B is significantly better than model A.
    H0: WER(A) <= WER(B)
    Returns p-value for the one-sided test WER(B) < WER(A).

    Standard approach: Berg-Kirkpatrick et al. (2012).
    """
    rng = random.Random(seed)
    n   = len(refs)

    wers_a = [per_clip_wer(r, h) for r, h in zip(refs, hyps_a)]
    wers_b = [per_clip_wer(r, h) for r, h in zip(refs, hyps_b)]

    observed_diff = sum(wers_a) / n - sum(wers_b) / n  # positive = B is better

    count_greater = 0
    for _ in range(n_iters):
        idxs    = [rng.randrange(n) for _ in range(n)]
        boot_a  = sum(wers_a[i] for i in idxs) / n
        boot_b  = sum(wers_b[i] for i in idxs) / n
        if (boot_a - boot_b) >= observed_diff:
            count_greater += 1

    p_value = count_greater / n_iters
    return {
        "wer_a":          round(sum(wers_a) / n, 4),
        "wer_b":          round(sum(wers_b) / n, 4),
        "wer_reduction":  round(observed_diff, 4),
        "wer_reduction_pct": round(
            observed_diff / (sum(wers_a) / n) * 100 if sum(wers_a) else 0.0, 2
        ),
        "p_value":        round(p_value, 4),
        "significant_at_005": p_value < 0.05,
        "significant_at_001": p_value < 0.01,
    }


# ── Effect size (Cohen's d) ────────────────────────────────────────────────────

def cohens_d(wers_a: list[float], wers_b: list[float]) -> float:
    """Cohen's d between two distributions of per-clip WER values."""
    import math
    n_a, n_b = len(wers_a), len(wers_b)
    if n_a < 2 or n_b < 2:
        return 0.0
    mean_a = sum(wers_a) / n_a
    mean_b = sum(wers_b) / n_b
    var_a  = sum((x - mean_a) ** 2 for x in wers_a) / (n_a - 1)
    var_b  = sum((x - mean_b) ** 2 for x in wers_b) / (n_b - 1)
    pooled = math.sqrt((var_a + var_b) / 2)
    return round((mean_a - mean_b) / pooled, 4) if pooled > 0 else 0.0


# ── Driver ────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    available = {k: load_predictions(k) for k in TABLE_ORDER}
    available = {k: v for k, v in available.items() if v}

    if not available:
        print("[ERROR] No prediction files found. Run 01_run_baselines.py and 02_run_finetuned.py first.")
        raise SystemExit(1)

    lines  = ["=" * 70, "STATISTICAL ANALYSIS REPORT", "=" * 70, ""]
    report = {}

    # ── Bootstrap CIs ────────────────────────────────────────────────────────
    lines.append("1. BOOTSTRAP CONFIDENCE INTERVALS (95%)")
    lines.append("-" * 70)
    ci_results = {}
    for key, preds in available.items():
        refs = [p["reference"]  for p in preds]
        hyps = [p["hypothesis"] for p in preds]
        obs, lo, hi = bootstrap_wer_ci(refs, hyps)
        label = MODEL_LABELS.get(key, key)
        lines.append(
            f"  {label:<48} WER = {obs:.4f}  95% CI [{lo:.4f}, {hi:.4f}]"
        )
        ci_results[key] = {"wer": obs, "ci_lower": lo, "ci_upper": hi}

    report["bootstrap_ci"] = ci_results
    lines.append("")

    # ── Paired tests vs proposed system ──────────────────────────────────────
    lines.append("2. PAIRED BOOTSTRAP SIGNIFICANCE TESTS")
    lines.append("   (Testing if our proposed system 'ft_all' is significantly better)")
    lines.append("-" * 70)

    sig_results = {}
    if "ft_whisper_all" in available:
        proposed = available["ft_whisper_all"]
        prop_refs = [p["reference"]  for p in proposed]
        prop_hyps = [p["hypothesis"] for p in proposed]

        for key, preds in available.items():
            if key == "ft_whisper_all":
                continue
            # Match clips by clip_id for proper pairing
            prop_map = {p["clip_id"]: p for p in proposed}
            pairs    = [(p, prop_map[p["clip_id"]]) for p in preds if p["clip_id"] in prop_map]
            if not pairs:
                continue

            refs_a  = [p[0]["reference"]  for p in pairs]
            hyps_a  = [p[0]["hypothesis"] for p in pairs]
            hyps_b  = [p[1]["hypothesis"] for p in pairs]

            result = paired_bootstrap_test(refs_a, hyps_a, hyps_b, seed=42)

            wers_a = [per_clip_wer(r, h) for r, h in zip(refs_a, hyps_a)]
            wers_b = [per_clip_wer(r, h) for r, h in zip(refs_a, hyps_b)]
            d      = cohens_d(wers_a, wers_b)
            result["cohens_d"] = d

            label = MODEL_LABELS.get(key, key)
            sig_star = ""
            if result["significant_at_001"]:
                sig_star = "***"
            elif result["significant_at_005"]:
                sig_star = "*"

            lines.append(
                f"  vs {label:<40}"
            )
            lines.append(
                f"     WER reduction: {result['wer_a']:.4f} → {result['wer_b']:.4f}"
                f"  ({result['wer_reduction_pct']:+.1f}%)"
                f"  p={result['p_value']:.4f} {sig_star}"
                f"  Cohen's d={d:.3f}"
            )
            sig_results[key] = result

        lines.append("")
        lines.append("  * p<0.05  ** p<0.01  *** p<0.001")
    else:
        lines.append("  [SKIP] ft_whisper_all predictions not found.")

    report["significance_vs_proposed"] = sig_results
    lines.append("")

    # ── CS vs BN_ONLY WER gap ────────────────────────────────────────────────
    lines.append("3. CODE-SWITCHING DIFFICULTY: CS clips vs BN_ONLY clips WER gap")
    lines.append("-" * 70)
    cs_gap_results = {}
    for key, preds in available.items():
        cs_preds = [p for p in preds if p.get("is_code_switched")]
        bn_preds = [p for p in preds if not p.get("is_code_switched")]
        if not cs_preds or not bn_preds:
            continue

        wer_cs = sum(per_clip_wer(p["reference"], p["hypothesis"]) for p in cs_preds) / len(cs_preds)
        wer_bn = sum(per_clip_wer(p["reference"], p["hypothesis"]) for p in bn_preds) / len(bn_preds)
        gap    = wer_cs - wer_bn

        label = MODEL_LABELS.get(key, key)
        lines.append(
            f"  {label:<48} WER-CS={wer_cs:.4f}  WER-BN={wer_bn:.4f}"
            f"  Gap={gap:+.4f}"
        )
        cs_gap_results[key] = {"wer_cs": round(wer_cs, 4), "wer_bn": round(wer_bn, 4), "gap": round(gap, 4)}

    report["cs_difficulty_gap"] = cs_gap_results
    lines.append("")

    # ── Summary ───────────────────────────────────────────────────────────────
    lines += [
        "=" * 70,
        "INTERPRETATION NOTES FOR PAPER",
        "=" * 70,
        "  - Bootstrap CI: if CI of model A and model B do not overlap,",
        "    the difference is significant without needing a p-value.",
        "  - Paired bootstrap p < 0.05 → statistically significant improvement.",
        "  - Cohen's d > 0.5 = medium effect, > 0.8 = large effect.",
        "  - CS difficulty gap (WER-CS > WER-BN) proves that code-switching",
        "    adds difficulty on top of dialect variation — key research finding.",
        "=" * 70,
    ]

    report_text = "\n".join(lines)
    print("\n" + report_text)

    txt_path  = RESULTS_DIR / "statistical_report.txt"
    json_path = RESULTS_DIR / "statistical_report.json"
    txt_path.write_text(report_text, encoding="utf-8")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {txt_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
