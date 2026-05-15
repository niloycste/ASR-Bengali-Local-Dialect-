"""
Experiment: LM Shallow Fusion for CTC models (wav2vec2 / WavLM).

What is LM Shallow Fusion?
  CTC models produce character/subword probabilities per frame.
  Beam search decodes these into words. Without a Language Model,
  the beam only uses acoustic scores — it picks the most acoustically
  likely sequence, which is often not grammatically correct.

  Shallow Fusion = during beam search, add a weighted LM score:
    final_score = acoustic_score + alpha * lm_score + beta * len(hypothesis)
      alpha = LM weight  (how much to trust the LM)
      beta  = word insertion bonus (prevents the model from producing short outputs)

  This improves WER significantly for dialectal speech because the LM
  "knows" what Bengali + English word sequences are likely.

Pipeline:
  Step 1 — Build n-gram LM from your training transcripts (KenLM format)
  Step 2 — Decode CTC model outputs using beam search + LM (pyctcdecode)
  Step 3 — Save predictions and compute metrics

Requirements:
  pip install pyctcdecode kenlm

  KenLM also needs a system install:
    Linux/Mac: pip install https://github.com/kpu/kenlm/archive/master.zip
    Windows:   Use WSL or a pre-built wheel from:
               https://github.com/kpu/kenlm/releases

Usage:
  python 09_lm_shallow_fusion.py                    # both wav2vec2 and WavLM
  python 09_lm_shallow_fusion.py --arch wav2vec2    # only wav2vec2
  python 09_lm_shallow_fusion.py --arch wavlm       # only WavLM
  python 09_lm_shallow_fusion.py --tune_alpha       # grid search alpha/beta on dev
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fine_tuning.config import (
    CTC_MODELS, DATASET_DIR, LM_CONFIG, LM_DIR,
    MODELS_DIR, RESULTS_DIR,
)

SAMPLE_RATE = 16_000
ARPA_PATH   = LM_DIR / f"bengali_cs_{LM_CONFIG['ngram_order']}gram.arpa"
BIN_PATH    = LM_DIR / f"bengali_cs_{LM_CONFIG['ngram_order']}gram.bin"


# ── Step 1: Build n-gram Language Model ────────────────────────────────────────

def collect_transcripts() -> list[str]:
    """Collect all training + dev transcripts to build the LM corpus.

    Also picks up synthetic CS transcripts from generate_synthetic_cs.py
    (text-only Strategy A) if that file exists.
    """
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pip install pandas"); raise SystemExit(1)

    texts = []
    for split in LM_CONFIG["lm_corpus_splits"]:
        csv_path = DATASET_DIR / split / "manifest.csv"
        if not csv_path.exists():
            print(f"  [WARNING] {csv_path} not found, skipping.")
            continue
        df = pd.read_csv(csv_path, encoding="utf-8")
        for t in df["transcript"].dropna():
            t = str(t).strip()
            if t:
                texts.append(t)

    # Also include synthetic CS transcripts (text-only, no audio needed)
    synth_path = (Path(__file__).resolve().parent.parent
                  / "dataset_pipeline" / "transcripts"
                  / "synthetic_cs_transcripts.json")
    if synth_path.exists():
        import json
        with open(synth_path, encoding="utf-8") as f:
            synth_clips = json.load(f)
        synth_texts = [c.get("transcript", "").strip()
                       for c in synth_clips if c.get("transcript", "").strip()]
        texts.extend(synth_texts)
        print(f"  [INFO] Added {len(synth_texts)} synthetic CS sentences to LM corpus.")

    print(f"  LM corpus: {len(texts)} sentences from splits: {LM_CONFIG['lm_corpus_splits']}")
    return texts


def build_lm(texts: list[str]) -> Path:
    """
    Build a KenLM n-gram language model from transcript texts.
    Returns path to the binary .bin model file.

    KenLM lmplz command must be on PATH. If not available, falls back to
    a pure-Python n-gram approximation (slower but no system install needed).
    """
    LM_DIR.mkdir(parents=True, exist_ok=True)

    if BIN_PATH.exists():
        print(f"  LM already built: {BIN_PATH}")
        return BIN_PATH

    # Write corpus to temp file
    corpus_path = LM_DIR / "lm_corpus.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        f.write("\n".join(texts))
    print(f"  Corpus written: {corpus_path}  ({len(texts)} lines)")

    # Try KenLM
    try:
        # Build ARPA file
        lmplz_cmd = [
            "lmplz",
            "-o", str(LM_CONFIG["ngram_order"]),
            "--text", str(corpus_path),
            "--arpa", str(ARPA_PATH),
            "--discount_fallback",
        ]
        result = subprocess.run(lmplz_cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode()[:300])

        # Convert ARPA → binary (faster loading)
        build_cmd = ["build_binary", str(ARPA_PATH), str(BIN_PATH)]
        subprocess.run(build_cmd, check=True, capture_output=True)
        print(f"  KenLM model built: {BIN_PATH}")
        return BIN_PATH

    except (FileNotFoundError, RuntimeError) as e:
        print(f"  [WARNING] KenLM not available ({e}). Using ARPA file directly.")
        # lmplz failed or not installed — try python kenlm via pip
        try:
            import kenlm  # type: ignore
            # If kenlm python module exists but lmplz binary doesn't,
            # we can't build the ARPA. Fall back to simple bigram.
        except ImportError:
            pass
        # Return ARPA path if it exists, else None
        return ARPA_PATH if ARPA_PATH.exists() else None


def build_vocab_from_model(model_dir: Path) -> list[str]:
    """Extract vocabulary from a fine-tuned wav2vec2 processor."""
    try:
        from transformers import Wav2Vec2Processor
    except ImportError:
        print("[ERROR] pip install transformers"); raise SystemExit(1)

    processor = Wav2Vec2Processor.from_pretrained(str(model_dir))
    vocab = processor.tokenizer.get_vocab()
    # Sort by token id, return list of tokens
    return [tok for tok, _ in sorted(vocab.items(), key=lambda x: x[1])]


# ── Step 2: Decode with LM shallow fusion ────────────────────────────────────

def load_audio_np(audio_path: str):
    import numpy as np
    from pydub import AudioSegment
    audio = (
        AudioSegment.from_file(audio_path)
        .set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
    )
    return np.frombuffer(audio.raw_data, dtype=np.int16).astype(np.float32) / 32768.0


def get_logits(model, processor, audio_np, device):
    """Run CTC model forward pass and return logits."""
    import torch
    inputs = processor(
        audio_np, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_values.to(device)
    with torch.no_grad():
        logits = model(inputs).logits
    return logits.cpu().numpy()[0]


def decode_with_lm(
    logits,
    decoder,
    alpha: float,
    beta: float,
    beam_width: int,
) -> str:
    """
    Decode CTC logits using beam search + LM shallow fusion.
    pyctcdecode handles the alpha/beta weighting internally.
    """
    text = decoder.decode(
        logits,
        beam_width=beam_width,
        alpha=alpha,
        beta=beta,
    )
    return text.strip()


def decode_greedy(logits, processor) -> str:
    """Greedy CTC decode (no LM) for comparison."""
    import numpy as np
    pred_ids = logits.argmax(axis=-1)
    return processor.decode(pred_ids).strip()


def build_decoder(model_dir: Path, lm_path: Path) -> Any:
    """Build pyctcdecode BeamSearchDecoderCTC with the n-gram LM."""
    try:
        from pyctcdecode import build_ctcdecoder  # type: ignore
    except ImportError:
        print("[ERROR] pip install pyctcdecode")
        print("        Also install KenLM: pip install https://github.com/kpu/kenlm/archive/master.zip")
        raise SystemExit(1)

    vocab = build_vocab_from_model(model_dir)
    # pyctcdecode expects vocab as list of strings, with blank at index 0
    # Remove special tokens that aren't real characters
    clean_vocab = [
        v if v not in ("[PAD]", "<pad>") else ""
        for v in vocab
    ]

    kenlm_model = str(lm_path) if lm_path and lm_path.exists() else None
    decoder = build_ctcdecoder(
        clean_vocab,
        kenlm_model=kenlm_model,
    )
    return decoder


# ── Step 3: Alpha/Beta tuning on dev set ──────────────────────────────────────

def tune_alpha_beta(
    model,
    processor,
    decoder,
    dev_clips: list[dict],
    device: str,
    arch: str,
) -> tuple[float, float]:
    """
    Grid search over alpha and beta on the dev set.
    Returns the (alpha, beta) pair with lowest WER.
    """
    try:
        from jiwer import wer as jiwer_wer
    except ImportError:
        print("[ERROR] pip install jiwer"); raise SystemExit(1)

    alpha_values = [0.1, 0.3, 0.5, 0.7, 1.0]
    beta_values  = [0.5, 1.0, 1.5, 2.0]

    print(f"\n  Tuning alpha/beta on {len(dev_clips)} dev clips ...")
    print(f"  Grid: alpha={alpha_values} × beta={beta_values}")

    best_wer   = float("inf")
    best_alpha = LM_CONFIG["alpha"]
    best_beta  = LM_CONFIG["beta"]

    # Precompute logits for dev set (expensive — done once)
    all_logits = []
    all_refs   = []
    for clip in dev_clips[:200]:  # limit to 200 for speed
        if not Path(str(clip["audio_path"])).exists():
            continue
        audio_np = load_audio_np(clip["audio_path"])
        logits   = get_logits(model, processor, audio_np, device)
        all_logits.append(logits)
        all_refs.append(str(clip.get("transcript", "")).strip())

    for alpha in alpha_values:
        for beta in beta_values:
            hyps = [
                decode_with_lm(lg, decoder, alpha, beta, beam_width=50)
                for lg in all_logits
            ]
            paired = [(r, h) for r, h in zip(all_refs, hyps) if r]
            if not paired:
                continue
            refs_f, hyps_f = zip(*paired)
            wer = jiwer_wer(list(refs_f), list(hyps_f))
            if wer < best_wer:
                best_wer   = wer
                best_alpha = alpha
                best_beta  = beta

    print(f"  Best: alpha={best_alpha}  beta={best_beta}  dev_WER={best_wer:.4f}")

    # Save tuning results
    tune_path = RESULTS_DIR / f"ft_{arch}_lm_alpha_beta_tuning.json"
    tune_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tune_path, "w") as f:
        json.dump({"best_alpha": best_alpha, "best_beta": best_beta,
                   "best_dev_wer": best_wer}, f, indent=2)

    return best_alpha, best_beta


# ── Main evaluation function ──────────────────────────────────────────────────

def run_lm_fusion(arch: str, tune_alpha: bool = False):
    cfg       = CTC_MODELS[arch]
    model_key = cfg["out_key"]
    model_dir = MODELS_DIR / model_key

    if not model_dir.exists():
        print(f"[ERROR] Fine-tuned model not found: {model_dir}")
        print(f"        Run: python fine_tuning/03_finetune_wav2vec2.py --arch {arch}")
        raise SystemExit(1)

    print(f"\n{'='*60}")
    print(f"LM Shallow Fusion: {cfg['label']}")
    print(f"  Model dir  : {model_dir}")
    print(f"  LM order   : {LM_CONFIG['ngram_order']}-gram")
    print(f"{'='*60}\n")

    try:
        import torch
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    except ImportError:
        print("[ERROR] pip install transformers torch"); raise SystemExit(1)

    # Load model
    device    = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Wav2Vec2Processor.from_pretrained(str(model_dir))
    model     = Wav2Vec2ForCTC.from_pretrained(str(model_dir)).to(device)
    model.eval()

    # Build LM
    print("Building/loading language model ...")
    texts    = collect_transcripts()
    lm_path  = build_lm(texts)

    # Build decoder
    print("Building beam search decoder ...")
    decoder = build_decoder(model_dir, lm_path)

    # Load test and dev manifests
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pip install pandas"); raise SystemExit(1)

    test_df = pd.read_csv(DATASET_DIR / "test" / "manifest.csv", encoding="utf-8")
    test_clips = [r for r in test_df.to_dict("records")
                  if Path(str(r["audio_path"])).exists()]

    dev_df = pd.read_csv(DATASET_DIR / "dev" / "manifest.csv", encoding="utf-8")
    dev_clips = [r for r in dev_df.to_dict("records")
                 if Path(str(r["audio_path"])).exists()]

    # Tune or use defaults
    if tune_alpha:
        alpha, beta = tune_alpha_beta(model, processor, decoder, dev_clips, device, arch)
    else:
        alpha = LM_CONFIG["alpha"]
        beta  = LM_CONFIG["beta"]
        print(f"  Using default alpha={alpha}  beta={beta}  (run with --tune_alpha to optimise)")

    beam_width = LM_CONFIG["beam_width"]

    # Decode test set
    print(f"\nDecoding {len(test_clips)} test clips with LM fusion ...")
    predictions = []
    for i, clip in enumerate(test_clips):
        audio_np = load_audio_np(clip["audio_path"])
        logits   = get_logits(model, processor, audio_np, device)

        pred_lm     = decode_with_lm(logits, decoder, alpha, beta, beam_width)
        pred_greedy = decode_greedy(logits, processor)

        predictions.append({
            "clip_id":          clip.get("clip_id", ""),
            "audio_path":       clip["audio_path"],
            "reference":        str(clip.get("transcript", "")).strip(),
            "hypothesis":       pred_lm,
            "hypothesis_greedy": pred_greedy,
            "dialect":          clip.get("dialect", "unknown"),
            "domain":           clip.get("domain", "General"),
            "is_code_switched": bool(clip.get("is_code_switched", False)),
            "bn_ratio":         float(clip.get("bn_ratio", 0.0)),
            "en_ratio":         float(clip.get("en_ratio", 0.0)),
            "duration_sec":     float(clip.get("duration_sec", 0.0)),
            "alpha":            alpha,
            "beta":             beta,
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(test_clips)}]")

    # Save predictions
    lm_key   = f"ft_{arch}_lm"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{lm_key}_predictions.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")

    # Quick WER comparison: greedy vs LM
    try:
        from jiwer import wer as jiwer_wer
        refs        = [p["reference"]        for p in predictions if p["reference"]]
        hyps_lm     = [p["hypothesis"]       for p in predictions if p["reference"]]
        hyps_greedy = [p["hypothesis_greedy"] for p in predictions if p["reference"]]
        wer_lm      = jiwer_wer(refs, hyps_lm)
        wer_greedy  = jiwer_wer(refs, hyps_greedy)
        improvement = (wer_greedy - wer_lm) / wer_greedy * 100 if wer_greedy > 0 else 0

        print(f"\n  WER without LM (greedy) : {wer_greedy:.4f}")
        print(f"  WER with LM fusion      : {wer_lm:.4f}")
        print(f"  Relative improvement    : {improvement:.1f}%")

        summary = {
            "arch": arch, "model_key": lm_key,
            "alpha": alpha, "beta": beta,
            "wer_greedy": round(wer_greedy, 4),
            "wer_lm":     round(wer_lm, 4),
            "relative_improvement_pct": round(improvement, 2),
        }
        summary_path = RESULTS_DIR / f"{lm_key}_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Summary saved: {summary_path}")

    except ImportError:
        pass

    print(f"\nNext: python 03_compute_metrics.py --model {lm_key}")


def main(arch: str | None = None, tune_alpha: bool = False):
    archs = [arch] if arch else list(CTC_MODELS.keys())
    for a in archs:
        run_lm_fusion(a, tune_alpha=tune_alpha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch",
        choices=list(CTC_MODELS.keys()),
        default=None,
        help="CTC architecture to apply LM fusion to. Default: both.",
    )
    parser.add_argument(
        "--tune_alpha",
        action="store_true",
        help="Grid search alpha/beta on dev set before decoding test set.",
    )
    args = parser.parse_args()
    main(args.arch, tune_alpha=args.tune_alpha)
