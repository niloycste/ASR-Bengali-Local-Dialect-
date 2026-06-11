"""
Clean (re-)upload of the BanglaMix dataset to the Hugging Face Hub.

WHY THIS SCRIPT
---------------
The Hub previously held raw `save_to_disk` arrow dumps of THREE subsets
(all / bn_only / cs_only). That broke `load_dataset()` and the Viewer, tripled
storage (~99 GB), and published training artifacts. Locally, the prepared arrow
folders were later trimmed (train shards deleted), so we rebuild the clean
dataset directly from the *real source*:

  - metadata + transcripts : dataset_pipeline/transcripts/transcripts_reviewed.json
  - audio                  : dataset_pipeline/segments/<dialect>/<clip>.wav

It keeps only the REAL clips (drops augmented/synthetic), keeps clean columns,
casts audio to a 16 kHz Audio feature, and `push_to_hub()` -> PARQUET. Then it
wipes all old/garbage content from the repo (keeping only the new data/ + README).

USAGE
-----
    huggingface-cli login
    python upload_dataset_to_hf.py                 # clean push: real clips only, with splits;
                                                   #   pushes new parquet, then deletes old files
    python upload_dataset_to_hf.py --fresh-repo    # delete the WHOLE old repo first, then push
                                                   #   (nothing old survives, not even history)
    python upload_dataset_to_hf.py --dry-run       # print what would happen, do NOT upload
    python upload_dataset_to_hf.py --keep-augmented   # also include augmented clips
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

PIPE = SCRIPT_DIR / "dataset_pipeline"
TRANSCRIPTS = PIPE / "transcripts" / "transcripts_reviewed.json"
RESULTS = SCRIPT_DIR / "evaluation" / "results"
DEV_ARROW = SCRIPT_DIR / "fine_tuning" / "hf_datasets" / "all" / "dev"   # surviving dev shard

REPO_ID = "niloycste68/Bangali_local_dialect_ASR_HF_Dataset"

# Public-facing columns (everything else in the json is internal and dropped).
KEEP = ["audio", "sentence", "clip_id", "dialect", "domain",
        "is_code_switched", "cs_confidence", "bn_ratio", "en_ratio", "duration_sec"]

AUG_MARKERS = ("/augmented/", "/synthetic/", "augment", "synthetic")


def is_augmented(audio_path: str) -> bool:
    p = str(audio_path).replace("\\", "/").lower()
    return any(m in p for m in AUG_MARKERS)


def recover_split_ids() -> tuple[set, set]:
    """Best effort: test clip_ids from any predictions JSON; dev clip_ids from the
    surviving dev arrow shard. Returns (dev_ids, test_ids) — possibly empty."""
    test_ids, dev_ids = set(), set()
    for cand in [RESULTS / "ft_whisper_all_predictions.json", *RESULTS.glob("*_predictions.json")]:
        if cand.exists():
            try:
                preds = json.load(open(cand, encoding="utf-8"))
                test_ids = {p.get("clip_id") for p in preds if p.get("clip_id")}
                if test_ids:
                    print(f"  [splits] {len(test_ids)} test clip_ids from {cand.name}")
                    break
            except Exception:
                pass
    # dev clip_ids from the surviving dev arrow shard, read with pyarrow directly
    # (version-independent; avoids datasets' load_from_disk metadata issues)
    try:
        import pyarrow as pa
        shard = next(DEV_ARROW.glob("*.arrow"), None) if DEV_ARROW.exists() else None
        if shard:
            try:
                table = pa.ipc.open_stream(str(shard)).read_all()
            except Exception:
                table = pa.ipc.open_file(str(shard)).read_all()
            if "clip_id" in table.column_names:
                dev_ids = set(table.column("clip_id").to_pylist())
                print(f"  [splits] {len(dev_ids)} dev clip_ids from dev shard (pyarrow)")
    except Exception as e:
        print(f"  [splits] dev shard not usable ({e})")
    return dev_ids, test_ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=REPO_ID)
    ap.add_argument("--transcripts", default=str(TRANSCRIPTS))
    ap.add_argument("--keep-augmented", action="store_true")
    ap.add_argument("--no-clean-old", action="store_true")
    ap.add_argument("--fresh-repo", action="store_true",
                    help="DELETE the entire existing repo and recreate it, so NONE of the "
                         "old data survives (not even in git history). Recommended for a "
                         "guaranteed-clean result.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    tpath = Path(args.transcripts)
    if not tpath.exists():
        print(f"[ERROR] transcripts not found: {tpath}"); sys.exit(1)

    print(f"Loading {tpath.name} ...")
    data = json.load(open(tpath, encoding="utf-8"))
    print(f"  {len(data):,} records")

    dev_ids, test_ids = recover_split_ids()

    records, missing, dropped_aug = [], 0, 0
    for r in data:
        ap_rel = r.get("audio_path", "")
        if not args.keep_augmented and is_augmented(ap_rel):
            dropped_aug += 1
            continue
        # resolve audio: relative to dataset_pipeline/, else as given
        cand = PIPE / str(ap_rel).replace("\\", "/")
        audio = cand if cand.exists() else Path(str(ap_rel))
        if not audio.exists():
            missing += 1
            continue
        transcript = (r.get("transcript") or "").strip()
        if not transcript:
            continue
        cid = r.get("clip_id", "")
        split = "test" if cid in test_ids else ("validation" if cid in dev_ids else "train")
        records.append({
            "audio": str(audio),
            "sentence": transcript,
            "clip_id": cid,
            "dialect": r.get("dialect", "unknown"),
            "domain": r.get("domain", "General"),
            "is_code_switched": bool(r.get("is_code_switched", False)),
            "cs_confidence": r.get("cs_confidence", "NONE"),
            "bn_ratio": float(r.get("bn_ratio", 0.0)),
            "en_ratio": float(r.get("en_ratio", 0.0)),
            "duration_sec": float(r.get("duration_sec", 0.0)),
            "_split": split,
        })

    print(f"  kept {len(records):,} real clips "
          f"(dropped {dropped_aug:,} augmented, {missing:,} with missing audio)")

    from collections import Counter
    by_split = Counter(r["_split"] for r in records)
    print("  split sizes:", dict(by_split))
    if not test_ids and not dev_ids:
        print("  [splits] could not recover dev/test -> single 'train' split "
              "(the official splits live in the GitHub repo / paper).")

    if args.dry_run:
        print("\n[DRY-RUN] No upload performed. Re-run without --dry-run to push.")
        return

    try:
        from datasets import Dataset, DatasetDict, Audio
        from huggingface_hub import HfApi, login
    except ImportError:
        print("[ERROR] pip install -U datasets huggingface_hub"); sys.exit(1)

    try:
        login(); print("Hugging Face login OK.")
    except Exception as e:
        print(f"[ERROR] login failed: {e}\n  run: huggingface-cli login"); sys.exit(1)

    # Optional: nuke the entire existing repo so NOTHING old survives (incl. git/LFS
    # history). The repo is recreated at the same URL by push_to_hub below.
    if args.fresh_repo:
        api = HfApi()
        try:
            api.delete_repo(repo_id=args.repo, repo_type="dataset", missing_ok=True)
            print(f"  [fresh-repo] deleted existing repo '{args.repo}' (recreated on push)")
        except TypeError:  # older huggingface_hub without missing_ok
            try:
                api.delete_repo(repo_id=args.repo, repo_type="dataset")
                print(f"  [fresh-repo] deleted existing repo '{args.repo}'")
            except Exception as e:
                print(f"  [fresh-repo] nothing to delete ({e})")
        except Exception as e:
            print(f"  [fresh-repo] delete skipped ({e})")

    # build DatasetDict (or a single split) with embedded 16 kHz audio.
    # Build fresh dicts that exclude the internal "_split" key (no mutation of
    # `records`, so the per-split selection stays correct across iterations).
    splits = {}
    for sp in ("train", "validation", "test"):
        rows = [{k: v for k, v in r.items() if k != "_split"}
                for r in records if r.get("_split") == sp]
        if not rows:
            continue
        d = Dataset.from_list(rows).cast_column("audio", Audio(sampling_rate=16000))
        splits[sp] = d
    if len(splits) == 1 and "train" in splits:
        ds = splits["train"]            # single split -> still load_dataset-able
    else:
        ds = DatasetDict(splits)

    print(f"\nPushing to https://huggingface.co/datasets/{args.repo} (parquet)...")
    ds.push_to_hub(args.repo, private=args.private)
    print("[SUCCESS] Pushed. load_dataset() and the Viewer will now work.")

    # Wipe all old/garbage content (keep only new data/ + README.md + .gitattributes).
    # Skipped when --fresh-repo was used (the repo was already wiped before the push).
    if not args.no_clean_old and not args.fresh_repo:
        api = HfApi()
        keep_root = {"README.md", ".gitattributes"}
        try:
            existing = api.list_repo_files(repo_id=args.repo, repo_type="dataset")
        except Exception as e:
            existing = []; print(f"  [cleanup] list failed ({e})")
        old_dirs, root_files = set(), []
        for f in existing:
            (old_dirs.add(f.split("/")[0]) if "/" in f else root_files.append(f))
        for d in sorted(old_dirs):
            if d == "data":
                continue
            try:
                api.delete_folder(path_in_repo=d, repo_id=args.repo, repo_type="dataset")
                print(f"  [cleanup] deleted old folder '{d}/'")
            except Exception as e:
                print(f"  [cleanup] could not delete '{d}/' ({e})")
        for f in root_files:
            if f in keep_root:
                continue
            try:
                api.delete_file(path_in_repo=f, repo_id=args.repo, repo_type="dataset")
                print(f"  [cleanup] deleted stale file '{f}'")
            except Exception as e:
                print(f"  [cleanup] could not delete '{f}' ({e})")

    print(f"\nDone. Verify: https://huggingface.co/datasets/{args.repo}")
    print("Test:\n  from datasets import load_dataset\n"
          f"  ds = load_dataset('{args.repo}'); print(ds)")


if __name__ == "__main__":
    main()
