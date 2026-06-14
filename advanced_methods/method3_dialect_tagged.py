"""
METHOD 3 -- Dialect-tagged (conditioned) fine-tuning
====================================================

THE PROBLEM IT SOLVES
  Per-dialect WER varies enormously (Sylhet ~0.93 and Sandwip ~0.92 vs. central
  dialects ~0.66-0.70). A single monolithic model is trained to ignore the known
  dialect of each clip, so it cannot specialise -- the hardest dialects stay hard.

THE FIX (the method)
  Prepend a dialect tag to the target text, e.g. "[Sylhet] <transcript>". The model
  jointly predicts the dialect and transcribes, conditioning its decoding on the
  dialect (a multitask, dialect-aware objective). No architecture change.
  Report per-dialect WER vs. the untagged model; the gain should concentrate on
  the hard dialects.

EVAL NOTE
  Strip the leading "[Dialect] " prefix from each hypothesis before scoring WER
  (one regex: re.sub(r'^\\s*\\[[^\\]]*\\]\\s*', '', hyp)).

USAGE
  python method3_dialect_tagged.py --out ../fine_tuning/models/ft_whisper_dialect
  # smoke test:
  python method3_dialect_tagged.py --max_train 60 --epochs 1 \
      --out ../fine_tuning/models/_smoketest_dia
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import (DEFAULT_DATASET, build_model_processor, finetune,  # noqa: E402
                     load_eval_split, load_split)

try:
    from fine_tuning.config import BASE_MODEL
except Exception:
    BASE_MODEL = "openai/whisper-small"


def tag_text(example):
    """Prepend a dialect tag to the transcript."""
    dialect = str(example.get("dialect", "unknown")).replace(" ", "_")
    return f"[{dialect}] {example['sentence']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--base_model", default=BASE_MODEL)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--max_train", type=int, default=None)
    a = ap.parse_args()

    print("== METHOD 3 [dialect-tagged] ==")
    model, processor = build_model_processor(a.base_model)
    train_ds = load_split(processor, a.dataset, "train", "all", a.max_train,
                          label_text_fn=tag_text)
    eval_ds = load_eval_split(processor, a.dataset, "all", label_text_fn=tag_text)
    print(f"  train={len(train_ds)}  eval={len(eval_ds)}")

    finetune(model, processor, train_ds, eval_ds, a.out,
             epochs=a.epochs, batch=a.batch, lr=a.lr, warmup=a.warmup)
    print(f"[done] saved to {a.out}")
    print("  REMEMBER: strip the leading '[Dialect] ' from hypotheses before scoring.")


if __name__ == "__main__":
    main()
