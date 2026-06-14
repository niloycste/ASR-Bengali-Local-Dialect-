"""
METHOD 1 -- CS-aware (English-weighted) fine-tuning
===================================================

THE PROBLEM IT SOLVES
  Your error analysis shows fine-tuned models DELETE English words on
  code-switched clips: English-word survival is ~0.00 for MMS / wav2vec2 /
  BN-only Whisper, and only 0.13 for the proposed model. Standard token
  cross-entropy barely penalizes dropping a rare English token, so the model
  learns it is "safe" to omit English. This is the dominant CS failure mode.

THE FIX (the method)
  Up-weight every Latin-script (English) target token in the loss by `en_weight`.
  Dropping English now costs more -> the model keeps English. No architecture
  change; one new hyperparameter.

WHAT TO REPORT
  Run baseline AND proposed, evaluate both, and show CS-aware raises
  CS-F1 / English-word survival at equal-or-better WER (paired bootstrap).

USAGE
  python method1_cs_aware.py --method baseline \
      --out ../fine_tuning/models/ft_whisper_baseline
  python method1_cs_aware.py --method cs_aware --en_weight 3.0 \
      --out ../fine_tuning/models/ft_whisper_csaware
  # quick smoke test first:
  python method1_cs_aware.py --method cs_aware --max_train 50 --epochs 1 \
      --out ../fine_tuning/models/_smoketest
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # project root
from _common import (DEFAULT_DATASET, build_model_processor, build_token_weights,  # noqa: E402
                     finetune, load_eval_split, load_split)

try:
    from fine_tuning.config import BASE_MODEL
except Exception:
    BASE_MODEL = "openai/whisper-small"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["baseline", "cs_aware"], default="cs_aware")
    ap.add_argument("--en_weight", type=float, default=3.0,
                    help="English-token loss weight (cs_aware only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--base_model", default=BASE_MODEL)
    ap.add_argument("--subset", choices=["all", "bn_only", "cs_only"], default="all")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--max_train", type=int, default=None)
    a = ap.parse_args()

    print(f"== METHOD 1 [{a.method}] base={a.base_model} subset={a.subset} ==")
    model, processor = build_model_processor(a.base_model)
    train_ds = load_split(processor, a.dataset, "train", a.subset, a.max_train)
    eval_ds = load_eval_split(processor, a.dataset, a.subset)
    print(f"  train={len(train_ds)}  eval={len(eval_ds)}")

    token_weights = (build_token_weights(processor.tokenizer, a.en_weight)
                     if a.method == "cs_aware" else None)
    finetune(model, processor, train_ds, eval_ds, a.out, token_weights=token_weights,
             epochs=a.epochs, batch=a.batch, lr=a.lr, warmup=a.warmup)
    print(f"[done] saved to {a.out}")


if __name__ == "__main__":
    main()
