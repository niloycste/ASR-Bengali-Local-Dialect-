"""
Evaluation Step 11 (Phase 3): LLM-as-a-Judge — Ensemble Evaluation of ASR Output.

Uses 3 LLMs running locally via Ollama to evaluate ASR hypotheses beyond WER.
Each model judges independently; scores are aggregated as an ensemble.

WHY ENSEMBLE?
  A single LLM judge can be biased or inconsistent. Using 3 different models
  (Mistral, LLaMA, Gemma) and averaging their scores gives a more reliable
  signal. Inter-model agreement also tells you how confident the ensemble is.

WHAT IS JUDGED (3 dimensions):
  1. Semantic Correctness (0–5)
       Does the hypothesis convey the same meaning as the reference,
       even if wording differs? (e.g. "miting" vs "meeting" = correct)
  2. CS Preservation (0–5)
       Does the hypothesis preserve the code-switching structure?
       Did English CS words survive, or were they silently dropped/transliterated?
  3. Dialect Naturalness (0–5)
       Does the hypothesis sound like natural dialectal Bengali speech,
       or has the model over-corrected to standard Bengali?

ENSEMBLE AGGREGATION:
  - Final score per dimension = mean of 3 model scores
  - Confidence = 1 - (std of 3 scores / 5)   [higher = more agreement]
  - Verdict = "PASS" if ensemble mean >= 3.0, else "FAIL"

MODELS USED (via Ollama):
  - mistral:7b    (strong instruction following, good multilingual)
  - llama3.1:8b   (strong English reasoning, decent Bengali)
  - gemma3:12b    (Google's model, good at structured output)

REQUIREMENTS:
  1. Install Ollama: https://ollama.com/download
  2. Pull models (run once in terminal):
       ollama pull mistral
       ollama pull llama3.1
       ollama pull gemma3
  3. Start Ollama server (it starts automatically on most systems):
       ollama serve
  4. pip install requests  (already in most Python installs)

Usage:
  python 11_llm_judge.py                          # judge all models, all clips
  python 11_llm_judge.py --model ft_whisper_all   # one model only
  python 11_llm_judge.py --sample 100             # random 100 clips (faster)
  python 11_llm_judge.py --cs_only                # CS clips only
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path


# --- Project-level imports ---
# Add project root to path to allow importing 'fine_tuning.config'
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from fine_tuning.config import BASELINES, FINETUNED, RESULTS_DIR, TABLE_ORDER, MODEL_LABELS

OLLAMA_URL  = "http://localhost:11434/api/generate"
JUDGE_DIR   = RESULTS_DIR / "llm_judge"

# ── The 3 judge models ────────────────────────────────────────────────────────
JUDGE_MODELS = [
    "mistral",    # Mistral 7B
    "llama3.1",   # LLaMA 3.1 8B
    "gemma3",     # Gemma 3 12B
]

# ── Scoring dimensions ────────────────────────────────────────────────────────
DIMENSIONS = ["semantic_correctness", "cs_preservation", "dialect_naturalness"]

# ── Prompt template ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """You are an expert evaluator of Automatic Speech Recognition (ASR) systems for Bengali-English code-switched speech.

You will evaluate an ASR hypothesis against a reference transcript.
The speech is from Bangladeshi speakers who naturally mix Bengali dialect with English words.

REFERENCE (ground truth):
{reference}

HYPOTHESIS (model output):
{hypothesis}

Evaluate on exactly these 3 dimensions. Score each from 0 to 5:

1. SEMANTIC_CORRECTNESS (0-5):
   Does the hypothesis convey the same meaning as the reference?
   - 5 = Meaning perfectly preserved (minor spelling/phonetic variants like "miting"/"meeting" are CORRECT)
   - 3 = Core meaning preserved but some words wrong
   - 1 = Meaning significantly changed
   - 0 = Completely wrong or empty

2. CS_PRESERVATION (0-5):
   Are English code-switched words in the reference preserved in the hypothesis?
   - 5 = All English CS words present (exact or phonetic variant)
   - 3 = Some English CS words preserved, some dropped
   - 1 = Most English CS words dropped or replaced with Bengali
   - 0 = All English CS words deleted (model produced Bengali-only output)
   - N/A = Reference has no English CS words → score 5 automatically

3. DIALECT_NATURALNESS (0-5):
   Does the hypothesis sound like natural dialectal Bangladeshi speech?
   - 5 = Natural dialectal Bengali preserved
   - 3 = Partially naturalised to standard Bengali
   - 1 = Heavily over-corrected to standard Bengali
   - 0 = Not Bengali at all

Respond ONLY in this exact JSON format (no explanation, no extra text):
{{
  "semantic_correctness": <integer 0-5>,
  "cs_preservation": <integer 0-5>,
  "dialect_naturalness": <integer 0-5>,
  "brief_reason": "<one sentence max>"
}}"""


# ── Ollama API call ───────────────────────────────────────────────────────────

def call_ollama(model: str, prompt: str, retries: int = 3) -> str | None:
    """Call Ollama API and return raw text response."""
    try:
        import requests
    except ImportError:
        print("[ERROR] pip install requests")
        raise SystemExit(1)

    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,   # low temperature = more consistent scoring
            "num_predict": 150,   # we only need a short JSON response
        },
    }

    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            else:
                print(f"    [WARN] Ollama {model} HTTP {resp.status_code} — retry {attempt+1}")
        except requests.exceptions.ConnectionError:
            if attempt == 0:
                print(f"    [ERROR] Cannot connect to Ollama at {OLLAMA_URL}")
                print(f"            Make sure Ollama is running: ollama serve")
            raise SystemExit(1)
        except Exception as e:
            print(f"    [WARN] {model} error: {e} — retry {attempt+1}")
        time.sleep(1)

    return None


def parse_scores(response: str) -> dict | None:
    """Extract JSON scores from model response."""
    if not response:
        return None

    # Find JSON block in response (model sometimes adds preamble)
    start = response.find("{")
    end   = response.rfind("}") + 1
    if start == -1 or end == 0:
        return None

    try:
        data = json.loads(response[start:end])
        scores = {}
        for dim in DIMENSIONS:
            val = data.get(dim)
            if val is None:
                return None
            scores[dim] = max(0, min(5, int(val)))   # clamp to [0, 5]
        scores["brief_reason"] = str(data.get("brief_reason", ""))[:200]
        return scores
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


# ── Single clip evaluation ────────────────────────────────────────────────────

def judge_clip(clip: dict) -> dict:
    """
    Run all 3 judge models on one clip and compute ensemble scores.
    Returns a dict with per-model scores and ensemble aggregation.
    """
    ref = clip["reference"].strip()
    hyp = clip["hypothesis"].strip()

    if not ref:
        return {"skipped": True, "reason": "empty_reference"}

    prompt = PROMPT_TEMPLATE.format(reference=ref, hypothesis=hyp)

    per_model = {}
    for model in JUDGE_MODELS:
        raw   = call_ollama(model, prompt)
        scores = parse_scores(raw)
        per_model[model] = scores if scores else None

    # ── Ensemble aggregation ──────────────────────────────────────────────────
    valid_models = {m: s for m, s in per_model.items() if s is not None}
    n_valid      = len(valid_models)

    if n_valid == 0:
        return {
            "clip_id":    clip["clip_id"],
            "dialect":    clip.get("dialect", "unknown"),
            "is_cs":      clip.get("is_code_switched", False),
            "reference":  ref,
            "hypothesis": hyp,
            "per_model":  per_model,
            "ensemble":   None,
            "valid_judges": 0,
        }

    ensemble = {}
    for dim in DIMENSIONS:
        dim_scores = [s[dim] for s in valid_models.values() if s and dim in s]
        if not dim_scores:
            ensemble[dim] = {"mean": None, "confidence": None, "verdict": "UNKNOWN"}
            continue
        mean = round(statistics.mean(dim_scores), 2)
        std  = statistics.stdev(dim_scores) if len(dim_scores) > 1 else 0.0
        conf = round(1.0 - (std / 5.0), 3)   # 1.0 = all models agree
        ensemble[dim] = {
            "mean":       mean,
            "std":        round(std, 3),
            "confidence": conf,
            "verdict":    "PASS" if mean >= 3.0 else "FAIL",
        }

    # Overall ensemble score = average across 3 dimensions
    dim_means = [ensemble[d]["mean"] for d in DIMENSIONS if ensemble[d]["mean"] is not None]
    overall   = round(statistics.mean(dim_means), 2) if dim_means else None

    # Agreement rate = fraction of model pairs that agree within 1 point
    agreement_pairs = 0
    total_pairs     = 0
    models_list     = list(valid_models.keys())
    for i in range(len(models_list)):
        for j in range(i + 1, len(models_list)):
            for dim in DIMENSIONS:
                s_i = valid_models[models_list[i]].get(dim, 0)
                s_j = valid_models[models_list[j]].get(dim, 0)
                total_pairs += 1
                if abs(s_i - s_j) <= 1:
                    agreement_pairs += 1
    agreement_rate = round(agreement_pairs / total_pairs, 3) if total_pairs else 0.0

    return {
        "clip_id":        clip["clip_id"],
        "dialect":        clip.get("dialect", "unknown"),
        "is_cs":          clip.get("is_code_switched", False),
        "reference":      ref,
        "hypothesis":     hyp,
        "per_model":      per_model,
        "ensemble":       ensemble,
        "overall_score":  overall,
        "agreement_rate": agreement_rate,
        "valid_judges":   n_valid,
    }


# ── Aggregate results across clips ────────────────────────────────────────────

def aggregate_results(results: list[dict]) -> dict:
    """Compute summary statistics across all judged clips."""
    valid = [r for r in results if r.get("ensemble") and not r.get("skipped")]

    if not valid:
        return {}

    summary = {"n_clips": len(valid)}

    for dim in DIMENSIONS:
        means = [r["ensemble"][dim]["mean"]
                 for r in valid
                 if r["ensemble"].get(dim) and r["ensemble"][dim]["mean"] is not None]
        if means:
            summary[f"{dim}_mean"] = round(statistics.mean(means), 3)
            summary[f"{dim}_std"]  = round(statistics.stdev(means) if len(means) > 1 else 0.0, 3)
            pass_rate = sum(1 for r in valid
                            if r["ensemble"].get(dim, {}).get("verdict") == "PASS") / len(valid)
            summary[f"{dim}_pass_rate"] = round(pass_rate, 3)

    overall_scores = [r["overall_score"] for r in valid if r.get("overall_score") is not None]
    summary["overall_mean"]      = round(statistics.mean(overall_scores), 3) if overall_scores else None
    summary["mean_agreement_rate"] = round(
        statistics.mean(r["agreement_rate"] for r in valid), 3
    )

    # Per-model breakdown
    for model in JUDGE_MODELS:
        model_scores = []
        for r in valid:
            ms = r["per_model"].get(model)
            if ms:
                model_scores.append(statistics.mean(ms[d] for d in DIMENSIONS if d in ms))
        summary[f"model_{model}_mean"] = round(statistics.mean(model_scores), 3) if model_scores else None

    # CS vs BN breakdown
    cs_results  = [r for r in valid if r.get("is_cs")]
    bn_results  = [r for r in valid if not r.get("is_cs")]
    for subset_name, subset in [("cs", cs_results), ("bn", bn_results)]:
        if subset:
            scores = [r["overall_score"] for r in subset if r.get("overall_score") is not None]
            summary[f"overall_{subset_name}_mean"] = round(statistics.mean(scores), 3) if scores else None

    return summary


# ── Driver ────────────────────────────────────────────────────────────────────

def run_judge(model_key: str, sample: int | None = None, cs_only: bool = False):
    pred_path = RESULTS_DIR / f"{model_key}_predictions.json"
    if not pred_path.exists():
        print(f"  [SKIP] {pred_path.name} not found — run 01_run_baselines.py or 02_run_finetuned.py first.")
        return None

    with open(pred_path, encoding="utf-8") as f:
        predictions = json.load(f)

    if cs_only:
        predictions = [p for p in predictions if p.get("is_code_switched")]
        print(f"  CS clips only: {len(predictions)}")

    if sample and sample < len(predictions):
        predictions = random.sample(predictions, sample)
        print(f"  Random sample: {len(predictions)} clips")

    print(f"\n{'='*60}")
    print(f"LLM Judge Ensemble: {model_key}")
    print(f"  ASR predictions : {len(predictions)} clips")
    print(f"  Judge models    : {', '.join(JUDGE_MODELS)}")
    print(f"{'='*60}")

    JUDGE_DIR.mkdir(parents=True, exist_ok=True)

    results      = []
    pass_counts  = {dim: 0 for dim in DIMENSIONS}
    total_scored = 0

    for i, clip in enumerate(predictions):
        print(f"  [{i+1:>4}/{len(predictions)}] {clip['clip_id']}", end=" ... ")

        result = judge_clip(clip)
        results.append(result)

        if result.get("ensemble") and not result.get("skipped"):
            total_scored += 1
            ens = result["ensemble"]
            verdicts = [ens[d]["verdict"] for d in DIMENSIONS if ens.get(d)]
            status   = " | ".join(
                f"{d[:4]}={'✓' if ens[d]['verdict']=='PASS' else '✗'} ({ens[d]['mean']:.1f})"
                for d in DIMENSIONS if ens.get(d)
            )
            print(f"overall={result['overall_score']:.1f}  agree={result['agreement_rate']:.2f}  [{status}]")
            for dim in DIMENSIONS:
                if ens.get(dim) and ens[dim]["verdict"] == "PASS":
                    pass_counts[dim] += 1
        else:
            print("[parse error or skipped]")

        # Save incrementally every 20 clips
        if (i + 1) % 20 == 0:
            _save_results(model_key, results)

    _save_results(model_key, results)

    # Summary
    summary = aggregate_results(results)
    summary_path = JUDGE_DIR / f"{model_key}_judge_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_key, **summary}, f, ensure_ascii=False, indent=2)

    print(f"\n  ── Ensemble Summary ──")
    print(f"  Clips scored          : {total_scored}")
    for dim in DIMENSIONS:
        m = summary.get(f"{dim}_mean")
        p = summary.get(f"{dim}_pass_rate", 0) * 100
        print(f"  {dim:<26}: mean={m:.2f}  pass={p:.1f}%")
    print(f"  Overall mean score    : {summary.get('overall_mean'):.2f} / 5.0")
    print(f"  Inter-model agreement : {summary.get('mean_agreement_rate'):.3f}")
    print(f"  Saved: {JUDGE_DIR / f'{model_key}_judge_results.json'}")
    print(f"  Saved: {summary_path}")

    return summary


def _save_results(model_key: str, results: list[dict]):
    out = JUDGE_DIR / f"{model_key}_judge_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def compare_all_models(all_summaries: dict):
    """Print comparison table across all ASR models judged."""
    if len(all_summaries) < 2:
        return

    print(f"\n{'='*70}")
    print(f"LLM JUDGE ENSEMBLE — COMPARISON ACROSS ASR MODELS")
    print(f"{'='*70}")
    print(f"{'Model':<35} {'Overall':>8} {'Semantic':>9} {'CS-Pres':>8} {'Dialect':>8} {'Agree':>7}")
    print(f"{'-'*70}")

    for key in TABLE_ORDER:
        if key not in all_summaries:
            continue
        s     = all_summaries[key]
        label = MODEL_LABELS.get(key, key)[:34]
        print(
            f"  {label:<33} "
            f"{s.get('overall_mean', 0):>7.2f}  "
            f"{s.get('semantic_correctness_mean', 0):>8.2f}  "
            f"{s.get('cs_preservation_mean', 0):>7.2f}  "
            f"{s.get('dialect_naturalness_mean', 0):>7.2f}  "
            f"{s.get('mean_agreement_rate', 0):>6.3f}"
        )

    # Save combined summary
    combined_path = JUDGE_DIR / "all_llm_judge_summary.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    print(f"\nFull comparison saved: {combined_path}")


def check_ollama_models():
    """Verify all judge models are available in Ollama before starting."""
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code != 200:
            raise ConnectionError
        available = [m["name"].split(":")[0] for m in resp.json().get("models", [])]
    except Exception:
        print("[ERROR] Cannot connect to Ollama.")
        print("        1. Install Ollama: https://ollama.com/download")
        print("        2. Run: ollama serve")
        raise SystemExit(1)

    missing = []
    for model in JUDGE_MODELS:
        model_base = model.split(":")[0]
        if model_base not in available:
            missing.append(model)

    if missing:
        print(f"[ERROR] These judge models are not pulled in Ollama: {missing}")
        print(f"        Pull them with:")
        for m in missing:
            print(f"          ollama pull {m}")
        raise SystemExit(1)

    print(f"  ✓ All judge models available: {', '.join(JUDGE_MODELS)}")


def main(model_key: str | None = None, sample: int | None = None, cs_only: bool = False):
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLLM-as-a-Judge Ensemble Evaluation")
    print(f"Judge models: {', '.join(JUDGE_MODELS)}")
    print("Checking Ollama connection...")
    check_ollama_models()

    all_keys = list(BASELINES.keys()) + list(FINETUNED.keys())
    keys     = [model_key] if model_key else all_keys

    all_summaries = {}
    for key in keys:
        summary = run_judge(key, sample=sample, cs_only=cs_only)
        if summary:
            all_summaries[key] = summary

    if len(all_summaries) > 1:
        compare_all_models(all_summaries)

    print("\nLLM Judge evaluation complete.")
    print(f"Results in: {JUDGE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=None,
                        choices=list(BASELINES.keys()) + list(FINETUNED.keys()),
                        help="Evaluate one ASR model only. Default: all models.")
    parser.add_argument("--sample",  type=int, default=None,
                        help="Randomly sample N clips per model (faster). Default: all clips.")
    parser.add_argument("--cs_only", action="store_true",
                        help="Only judge code-switched clips.")
    args = parser.parse_args()
    main(model_key=args.model, sample=args.sample, cs_only=args.cs_only)
