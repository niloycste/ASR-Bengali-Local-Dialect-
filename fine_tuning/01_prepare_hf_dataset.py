"""
Fine-tuning Step 1: Convert final_dataset/ CSVs into HuggingFace Dataset format.

Reads:  dataset_pipeline/final_dataset/{train,dev,test}/manifest.csv
Writes: fine_tuning/hf_datasets/{all,bn_only,cs_only}/

Run once before fine-tuning. Re-run only if the dataset changes.

Usage:
    python 01_prepare_hf_dataset.py               # prepares all three subsets
    python 01_prepare_hf_dataset.py --subset all  # only the "all" subset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── resolve imports regardless of working directory ────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATASET_DIR, HF_DATA_DIR, SUBSETS, BASE_MODEL

SAMPLE_RATE = 16_000


def load_manifest(split: str) -> list[dict]:
    """Read manifest.csv for a split and return list of dicts."""
    try:
        import pandas as pd
    except ImportError:
        print("[ERROR] pandas not installed. Run: pip install pandas")
        raise SystemExit(1)

    csv_path = DATASET_DIR / split / "manifest.csv"
    if not csv_path.exists():
        print(f"[ERROR] {csv_path} not found.")
        print("        Run dataset_pipeline/05_build_dataset.py first.")
        raise SystemExit(1)

    df = pd.read_csv(csv_path, encoding="utf-8")
    return df.to_dict("records")


def filter_subset(rows: list[dict], subset: str) -> list[dict]:
    """Keep only rows matching the requested subset."""
    if subset == "bn_only":
        return [r for r in rows if not r.get("is_code_switched", False)]
    if subset == "cs_only":
        return [r for r in rows if r.get("is_code_switched", False)]
    return rows  # "all"


def build_hf_dataset(rows: list[dict]):
    """
    Convert manifest rows to a HuggingFace Dataset with Audio feature.
    Audio is loaded lazily (path stored, decoded on access).
    """
    try:
        from datasets import Dataset, Audio
    except ImportError:
        print("[ERROR] datasets not installed. Run: pip install datasets")
        raise SystemExit(1)

    records = []
    missing = 0
    for row in rows:
        audio_path = row.get("audio_path", "")
        transcript = row.get("transcript", "").strip()
        if not audio_path or not Path(audio_path).exists():
            missing += 1
            continue
        if not transcript:
            missing += 1
            continue
        records.append({
            "audio":           audio_path,
            "sentence":        transcript,
            "clip_id":         row.get("clip_id", ""),
            "dialect":         row.get("dialect", "unknown"),
            "domain":          row.get("domain", "General"),
            "is_code_switched": bool(row.get("is_code_switched", False)),
            "bn_ratio":        float(row.get("bn_ratio", 0.0)),
            "en_ratio":        float(row.get("en_ratio", 0.0)),
            "duration_sec":    float(row.get("duration_sec", 0.0)),
        })

    if missing:
        print(f"  [WARNING] Skipped {missing} rows with missing audio or transcript.")

    ds = Dataset.from_list(records)
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    return ds


def prepare_feature_extraction(ds, processor):
    """
    Tokenise text. Acoustic feature extraction is done on-the-fly during training
    to avoid PyArrow 64GB memory limit crashes.
    """
    def _map(batch):
        labels  = processor.tokenizer(
            batch["sentence"],
            return_tensors="np",
        ).input_ids
        return {
            "labels":         labels[0].tolist(),
        }

    ds = ds.map(
        _map,
    )
    return ds


def main(subset: str = "all"):
    try:
        from transformers import WhisperProcessor
    except ImportError:
        print("[ERROR] transformers not installed. Run: pip install transformers")
        raise SystemExit(1)

    print(f"\n{'='*60}")
    print(f"Preparing HuggingFace dataset — subset: {subset}")
    print(f"  Base model  : {BASE_MODEL}")
    print(f"  Description : {SUBSETS[subset]}")
    print(f"{'='*60}\n")

    processor = WhisperProcessor.from_pretrained(
        BASE_MODEL, language="bengali", task="transcribe"
    )

    out_dir = HF_DATA_DIR / subset
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "dev", "test"]:
        print(f"--- {split} ---")
        rows = load_manifest(split)
        rows = filter_subset(rows, subset)
        print(f"  {len(rows)} clips after filtering.")

        ds = build_hf_dataset(rows)
        ds = prepare_feature_extraction(ds, processor)

        save_path = out_dir / split
        ds.save_to_disk(str(save_path))
        print(f"  Saved to: {save_path}  ({len(ds)} examples)")

    print(f"\nDone. Next: python 02_finetune_whisper.py --subset {subset}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subset",
        choices=list(SUBSETS.keys()),
        default=None,
        help="Which subset to prepare. If omitted, prepares ALL three subsets.",
    )
    args = parser.parse_args()

    if args.subset:
        main(args.subset)
    else:
        # No argument given — run all three subsets automatically
        for subset in SUBSETS:
            main(subset)
