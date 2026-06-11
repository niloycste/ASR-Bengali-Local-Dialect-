"""
Step 13: Score silver-standard label quality from the filled validation sample.

Reads validation_sample.csv (after a human has filled the human_transcript
column), normalises both human and silver transcripts, and reports the WER/CER
between them --- i.e. the human-verified accuracy of the Whisper silver labels.
Reports overall, CS vs BN, and per-dialect. These numbers go into the
"Dataset Quality" subsection of the paper.

Convention: the HUMAN transcript is the reference (ground truth); the SILVER
(Whisper) transcript is the hypothesis being validated. A low WER means the
silver labels closely match human transcription.

Usage:
    python 13_validation_wer.py
"""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
from fine_tuning.config import RESULTS_DIR
from dataset_pipeline.text_normalization import full_normalize


def _score(refs, hyps, metric):
    try:
        from jiwer import wer as jwer, cer as jcer
    except ImportError:
        print("[ERROR] pip install jiwer")
        raise SystemExit(1)
    fn = jwer if metric == "wer" else jcer
    pairs = [(full_normalize(r), full_normalize(h))
             for r, h in zip(refs, hyps) if full_normalize(r).strip()]
    if not pairs:
        return None
    r, h = zip(*pairs)
    return round(fn(list(r), list(h)), 4)


def _load_filled() -> list[dict]:
    """Read the validation sample, preferring the .xlsx (correct Bengali in Excel)."""
    xlsx = RESULTS_DIR / "validation_sample.xlsx"
    csvp = RESULTS_DIR / "validation_sample.csv"
    if xlsx.exists():
        from openpyxl import load_workbook
        ws = load_workbook(xlsx, read_only=True).active
        header = [c.value for c in next(ws.iter_rows(min_row=1, max_row=1))]
        rows = []
        for r in ws.iter_rows(min_row=2, values_only=True):
            rows.append({header[i]: ("" if r[i] is None else str(r[i]))
                         for i in range(len(header))})
        return rows
    if csvp.exists():
        with open(csvp, encoding="utf-8-sig") as f:
            return [dict(r) for r in csv.DictReader(f)]
    print("[ERROR] No validation_sample.xlsx/.csv found. Run 12_make_validation_sample.py first.")
    raise SystemExit(1)


def main():
    rows = [r for r in _load_filled() if str(r.get("human_transcript", "")).strip()]
    if not rows:
        print("[ERROR] No rows have a filled 'human_transcript'. Fill the file first.")
        raise SystemExit(1)

    refs = [r["human_transcript"] for r in rows]    # human = ground truth
    hyps = [r["silver_transcript"] for r in rows]   # silver = being validated

    print(f"\n{'='*56}")
    print(f"Silver-standard validation  (n = {len(rows)} human-transcribed clips)")
    print(f"{'='*56}")
    print(f"  WER (human vs silver labels): {_score(refs, hyps, 'wer')}")
    print(f"  CER (human vs silver labels): {_score(refs, hyps, 'cer')}")

    def subset_wer(flag):
        rr = [r for r in rows
              if (str(r.get('is_code_switched', '')).strip().lower() in ('true', '1')) == flag]
        if not rr:
            return None
        return _score([r['human_transcript'] for r in rr],
                      [r['silver_transcript'] for r in rr], 'wer')

    print(f"\n  CS clips WER : {subset_wer(True)}")
    print(f"  BN clips WER : {subset_wer(False)}")

    by_dialect = defaultdict(list)
    for r in rows:
        by_dialect[r.get('dialect', 'unknown')].append(r)
    print("\n  Per-dialect WER:")
    for d in sorted(by_dialect):
        rr = by_dialect[d]
        w = _score([r['human_transcript'] for r in rr],
                   [r['silver_transcript'] for r in rr], 'wer')
        print(f"    {d:<16}: {w}  (n={len(rr)})")

    print(f"\n{'='*56}")
    print("Report the overall WER as the silver-standard label quality in the paper.")


if __name__ == "__main__":
    main()
