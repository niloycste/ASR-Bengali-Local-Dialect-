# BanglaMix — Dialectal Bengali–English Code-Switching ASR

**The first large-scale dataset and benchmark for ASR on dialectal Bangladeshi speech mixed with English.**

| | |
|---|---|
| **Dataset** | 86,053 clips · 273.46 hours (augmented) · 15 dialects |
| **Task** | Automatic Speech Recognition (ASR) |
| **Languages** | Bengali (15 regional dialects) + English code-switching |
| **Target venues** | INTERSPEECH 2026 / ACL 2026 |

---

## Research Problem

Existing Bengali ASR resources cover either **dialect** or **code-switching (CS)** — never both:

| Dataset | Dialect? | Code-Switching? | Hours |
|---|---|---|---|
| OpenSLR-37 | No | No | 100 |
| MUCS 2021 | No | Yes (standard Bengali only) | 16 |
| Ben-10 (AACL 2025) | Yes (10 dialects) | No | 78 |
| **BanglaMix (ours)** | **Yes (15 dialects)** | **Yes** | **273** |

Young Bangladeshi speakers naturally mix English into regional dialect speech ("চাটগাঁইয়া ভাষায় phone unboxing করতাছি, honestly speaking camera performance অনেক ভালো"). No model or dataset covers this intersection.

---

## Project Structure

```
ASR work bengali/
│
├── dataset_pipeline/          ← Phase 1: Data collection & preparation
│   ├── 01_download_audio.py       Download from YouTube (dialect-name queries)
│   ├── 01b_download_extra_urls.py Download from Facebook / other sites
│   ├── convert_webm.py            Convert .webm/.m4a → .wav (16kHz mono)
│   ├── 02_segment_audio.py        VAD segmentation → 5–15 sec clips
│   ├── 03_auto_transcribe.py      Whisper large-v3 transcription + CS detection
│   ├── 04b_auto_annotate.py       Auto-accept high-confidence transcripts
│   ├── 05_build_dataset.py        Train/dev/test split (source-isolated, dialect-aware)
│   ├── code_mixing_utils.py       CS detection utilities (word-level script tags)
│   ├── pipeline_utils.py          Shared utilities (command checking, imports)
│   └── download_query_profiles.py YouTube query profiles per dialect
│
├── fine_tuning/               ← Phase 2: Model training
│   ├── config.py                  Central config (all paths, hyperparams, model keys)
│   ├── 01_prepare_hf_dataset.py   Convert CSVs → HuggingFace Dataset format
│   ├── 02_finetune_whisper.py     Fine-tune Whisper (3 subsets: all/bn_only/cs_only)
│   └── 03_finetune_wav2vec2.py    Fine-tune wav2vec2 / WavLM (CTC models)
│
├── evaluation/                ← Phase 3: Evaluation & paper results
│   ├── 01_run_baselines.py        Zero-shot: Whisper, Tugstugi, MMS, wav2vec2
│   ├── 02_run_finetuned.py        Run all fine-tuned models on test set
│   ├── 03_compute_metrics.py      WER, CER, MIX-ER, WER-BN, WER-EN, CS-F1
│   ├── 04_ablation_summary.py     Ablation table → .csv + .tex (paste into paper)
│   ├── 05_plot_results.py         5 publication-ready plots (PDF + PNG)
│   ├── 06_statistical_analysis.py Bootstrap CI, p-values, Cohen's d
│   ├── 07_cross_dialect_eval.py   Per-dialect WER + leave-one-out experiment
│   ├── 08_error_analysis.py       Sub/Del/Ins breakdown, EN word survival
│   ├── 09_lm_shallow_fusion.py    5-gram KenLM + beam search for CTC models
│   └── 10_whisper_size_comparison.py  tiny→large-v3 accuracy vs efficiency
│
├── asr_paper_draft/           ← Phase 4: LaTeX paper
│   ├── main.tex                   Master file (compile this)
│   ├── references.bib             All citations
│   ├── Makefile                   Run: make
│   └── sections/
│       ├── abstract.tex
│       ├── 01_introduction.tex
│       ├── 02_related_work.tex
│       ├── 03_dataset.tex
│       ├── 04_methodology.tex
│       ├── 05_experiments.tex
│       ├── 06_results.tex
│       ├── 07_analysis.tex
│       └── 08_conclusion.tex
│
└── requirements.txt           ← Install everything from here
```

---

## Installation

```bash
# 1. Create conda environment
conda create -n ASR python=3.12
conda activate ASR

# 2. Install PyTorch with CUDA (recommended — replace cu121 with your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install everything else
pip install -r requirements.txt
```

**System dependencies** (install once, not via pip):
- **ffmpeg** — required by pydub and Whisper
  ```bash
  # Windows:
  winget install ffmpeg
  # or: choco install ffmpeg
  ```
- **KenLM** — required only for LM shallow fusion (evaluation Step 11)
  ```bash
  pip install https://github.com/kpu/kenlm/archive/master.zip
  pip install pyctcdecode
  ```
- **Ollama** — required only for LLM-as-a-Judge ensemble (evaluation Step 11 / `11_llm_judge.py`)
  ```bash
  # 1. Install Ollama: https://ollama.com/download
  # 2. Pull the 3 judge models (run once):
  ollama pull mistral
  ollama pull llama3.1
  ollama pull gemma2
  # 3. Ollama starts automatically — or manually: ollama serve
  ```
- **gTTS** — required only if you run `generate_synthetic_cs.py --tts` (Strategy B)
  Already included in `requirements.txt`. If missing, install manually:
  ```bash
  pip install gTTS
  ```
  > **Note:** `generate_synthetic_cs.py` without `--tts` (Strategy A, text-only) does **not** need gTTS.

---

## Full Run Order — Step by Step

### Phase 1 — Data Collection & Preparation

```bash
cd dataset_pipeline

# Step 1a: Download from YouTube (dialect-name search queries)
python 01_download_audio.py

# Step 1b: Download from Facebook / other sites (paste URLs in script first)
python 01b_download_extra_urls.py

# Convert any .webm / .m4a files to .wav
python convert_webm.py

# Step 2: VAD segmentation → 5–15 second clips
python 02_segment_audio.py

# Step 3: Whisper transcription + CS detection (GPU strongly recommended)
python 03_auto_transcribe.py

# Step 4b: Auto-accept HIGH confidence transcripts (no_speech_prob < 0.30)
python 04b_auto_annotate.py

# Step 4c: LLM ensemble judges BORDERLINE clips (0.30 ≤ no_speech_prob < 0.60)
#          3 models vote: Mistral, LLaMA 3.1, Gemma 2 — majority wins
#          Requires Ollama running with all 3 models pulled (see Installation)
python 04c_llm_transcript_judge.py

# Step 5: Build source-isolated train/dev/test dataset splits
python 05_build_dataset.py
```

**Output after Phase 1:**
```
dataset_pipeline/
  raw_audio/          ← downloaded audio files
  segments/           ← 5–15 sec clips + segments_manifest.json
  transcripts/        ← transcripts.json, for_human_review.tsv, code_switched_clips.tsv
  final_dataset/
    train/manifest.csv + audio/
    dev/manifest.csv  + audio/
    test/manifest.csv + audio/
```

### Data ASR Worker Merge Status

The four `dataset_pipeline/Data ASR/transcripts_worker*` folders can be merged with:

```bash
python dataset_pipeline/07_merge_data_asr_workers.py
```

The merged output is written to:

```
dataset_pipeline/Data ASR/transcripts_merged_all/
```

Merged files:

```
transcripts.json
for_human_review.tsv
code_switched.tsv
code_switched_clips.tsv
```

Current merge summary:

| Source | Accepted clips |
|---|---:|
| `transcripts_worker1` | 12,524 |
| `transcripts_worker2` | 12,411 |
| `transcripts_worker3` | 12,444 |
| `transcripts_worker4` | 11,393 |
| **Merged total** | **48,772** |

If the expected total clip count is `52,016`, then:

| Metric | Count |
|---|---:|
| Expected clips | 52,016 |
| Accepted merged clips | 48,772 |
| Missing/rejected clips | 3,244 |
| Explicitly logged skipped clips | 242 |
| Unlogged missing/rejected clips | 3,002 |
| Duplicate `clip_id`s across workers | 0 |
| Code-switched clips in merged output | 2,056 |

Dialect distribution in the merged output:

| Dialect | Clips |
|---|---:|
| Sylhet | 9,231 |
| Sandwip | 5,483 |
| Old_Dhaka | 5,460 |
| Barishal | 3,827 |
| Chittagong | 3,483 |
| Khulna | 3,310 |
| Kishoreganj | 2,741 |
| Narail | 2,601 |
| Rangpur | 2,387 |
| Comilla | 2,384 |
| Narsingdi | 1,972 |
| Mymensingh | 1,699 |
| Habiganj | 1,609 |
| Tangail | 1,321 |
| Noakhali | 1,264 |
| **Total** | **48,772** |

---

### Phase 2 — Fine-Tuning

```bash
cd ../fine_tuning
pip install -r requirements.txt

# Prepare HuggingFace datasets (all 3 subsets at once — run once only)
python 01_prepare_hf_dataset.py

# Fine-tune Whisper large-v3 (3 experiments: all / bn_only / cs_only)
# Each run saves to fine_tuning/models/ft_whisper_{subset}/
python 02_finetune_whisper.py

# Fine-tune wav2vec2-XLS-R and WavLM (CTC models)
# Saves to fine_tuning/models/ft_wav2vec2_all/ and ft_wavlm_all/
python 03_finetune_wav2vec2.py
```

**What each fine-tuning experiment produces:**

| Model saved to | Training data | Purpose |
|---|---|---|
| `models/ft_whisper_all/` | All clips (BN_ONLY + CS) | **Proposed system** |
| `models/ft_whisper_bn_only/` | Dialect-only clips | Ablation A |
| `models/ft_whisper_cs_only/` | CS clips only | Ablation B |
| `models/ft_wav2vec2_all/` | All clips | CTC baseline |
| `models/ft_wavlm_all/` | All clips | CTC baseline |

**GPU time estimates (A100 80GB):**
- Whisper large-v3: ~8–12 hours per experiment
- wav2vec2 / WavLM: ~4–6 hours

---

### Phase 3 — Evaluation

```bash
cd ../evaluation
pip install -r requirements.txt

# Zero-shot baselines (Whisper, Tugstugi/Ben-10, MMS, wav2vec2)
python 01_run_baselines.py

# Fine-tuned model predictions
python 02_run_finetuned.py

# LM shallow fusion for CTC models (run --tune_alpha for best results)
python 09_lm_shallow_fusion.py --tune_alpha

# Whisper size comparison (tiny → large-v3)
python 10_whisper_size_comparison.py

# Compute all metrics: WER, CER, MIX-ER, WER-BN, WER-EN, CS-F1
python 03_compute_metrics.py

# Build ablation table → ablation_table.tex (paste into LaTeX paper)
python 04_ablation_summary.py

# Generate 5 publication-ready plots
python 05_plot_results.py

# Statistical significance: bootstrap CI, p-values, Cohen's d
python 06_statistical_analysis.py

# Cross-dialect WER breakdown + leave-one-out experiment
python 07_cross_dialect_eval.py

# Error analysis: Sub/Del/Ins + English word survival
python 08_error_analysis.py
```

**Output after Phase 3:**
```
evaluation/
  results/
    *_predictions.json          ← raw predictions per model
    *_metrics.json              ← metrics per model
    all_metrics_summary.json    ← everything in one file
    ablation_table.tex          ← paste into paper
    ablation_table.csv
    statistical_report.txt
    cross_dialect/
    error_analysis/
  plots/
    wer_comparison.pdf/png
    dialect_wer_heatmap.pdf/png
    cs_f1_comparison.pdf/png
    cs_vs_bn_wer.pdf/png
    dialect_distribution.pdf/png
    whisper_size_comparison.pdf/png
    error_analysis.pdf/png
```

---

### Phase 4 — Paper

```bash
cd ../asr_paper_draft

# Copy plots
cp ../evaluation/plots/*.png figures/

# Compile (requires MiKTeX or TeX Live)
make
# or manually:
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

After running all experiments, replace every `\TODO{X.XX}` in `sections/06_results.tex` and `sections/07_analysis.tex` with numbers from `evaluation/results/all_metrics_summary.json`.

---

## Models & Metrics

### Models evaluated

| Model | Type | Zero-shot | Fine-tuned |
|---|---|---|---|
| Whisper large-v3 | Seq2Seq | ✓ | ✓ (3 variants) |
| Whisper tiny/base/small/medium | Seq2Seq | ✓ | ✓ |
| Tugstugi/whisper-bn (Ben-10) | Seq2Seq | ✓ | — |
| Meta MMS-1B | CTC | ✓ | — |
| wav2vec2-XLS-R | CTC | ✓ | ✓ + LM |
| WavLM-Large | CTC | — | ✓ + LM |

### Metrics computed

| Metric | What it measures |
|---|---|
| **WER** | Overall word error rate |
| **CER** | Character error rate (important for Bengali morphology) |
| **WER-BN** | WER on Bengali-script words only |
| **WER-EN** | WER on English/Latin words only |
| **MIX-ER** | `(WER-BN + WER-EN) / 2` on CS clips — CS-aware metric |
| **CS-F1** | F1 for detecting whether a clip is code-switched |
| **EN-Survival** | Fraction of English reference words preserved in hypothesis |
| **Bootstrap CI** | 95% confidence interval for each model's WER |
| **p-value** | Paired bootstrap significance test (proposed vs each baseline) |
| **Cohen's d** | Effect size |

---

## Dataset Statistics

### After segmentation (Phase 1, Step 2)

| Dialect | Region | Clips | Hours |
|---|---|---|---|
| Sylhet | Northeast Bangladesh + UK diaspora | 9,665 | 31.85 |
| Old_Dhaka | Historic Dhaka city | 5,767 | 17.84 |
| Sandwip | Island, Chittagong coast | 5,744 | 16.31 |
| Barishal | South-central Bangladesh | 4,048 | 13.57 |
| Khulna | Southwest Bangladesh | 3,540 | 11.40 |
| Chittagong | Southeast Bangladesh | 3,661 | 11.16 |
| Kishoreganj | Central Bangladesh | 3,038 | 9.52 |
| Comilla | Southeast Bangladesh | 2,724 | 9.39 |
| Narail | Southwest Bangladesh | 2,714 | 8.93 |
| Rangpur | Northwest Bangladesh | 2,650 | 8.67 |
| Narsingdi | Central Bangladesh | 2,150 | 6.63 |
| Mymensingh | North-central Bangladesh | 1,831 | 5.50 |
| Habiganj | Northeast Bangladesh | 1,694 | 5.28 |
| Tangail | Central Bangladesh | 1,396 | 4.77 |
| Noakhali | Southeast Bangladesh | 1,235 | 3.93 |
| **Total** | **15 dialects** | **51,857** | **164.74** |

### Final Curated Dataset (After Augmentation & Splits)

| Split | Clips | Hours | CS Clips | CS % |
|---|---:|---:|---:|---:|
| **Train** | 78,122 | 247.38 | 6,662 | 8.5% |
| **Dev**   | 2,768  | 8.89   | 185   | 6.7% |
| **Test**  | 5,163  | 17.18  | 236   | 4.6% |
| **Total** | **86,053** | **273.46** | **7,083** | **8.2%** |

*(Note: The Train split includes augmented audio to balance underrepresented dialects and code-switched data. The Dev and Test sets remain strictly 100% real, unaugmented human speech. Roughly 10,000 original clips were safely filtered out during Phase 1 due to severe background noise or alignment errors).*

---

## Key Design Decisions

### Why dialect-name queries, not region-name queries

**Bad:** `"চট্টগ্রামের vlog english"` → finds anyone *from* Chittagong, they speak **standard Bengali** for YouTube audience.

**Good:** `"চাটগাঁইয়া ভাষায় comedy english"` → only surfaces creators who *consciously use* the dialect — a strong signal for genuine dialectal content.

### Why comedy/roast content

Comedy forces natural unscripted dialect. A creator cannot maintain a fake accent through jokes. This is the highest-precision query strategy for dialectal CS.

### Why silver-standard transcription (no human annotators)

Whisper large-v3 with `no_speech_prob < 0.30` produces transcripts with ~3–8% WER on clear speech. CS *detection* (Bengali vs. English words) is near-perfect because the scripts are visually distinct Unicode ranges. We validate on a 200-clip manual sample and report this WER in the paper.

### Why Whisper over CTC for the proposed system

Whisper's seq2seq tokeniser contains both Bengali Unicode and Latin ASCII in one shared vocabulary. It produces mixed-script output (`আমি phone কিনছি`) in a single forward pass. CTC models need an explicit character vocabulary covering both scripts and cannot produce mixed-script output as naturally.

---

## Requirements Summary

```
torch>=2.3.0              # GPU training
transformers>=4.40.0      # Whisper, wav2vec2, WavLM, MMS, HuBERT
datasets>=2.19.0          # HuggingFace dataset format
evaluate>=0.4.2           # WER metric
accelerate>=0.30.0        # Multi-GPU / mixed precision
openai-whisper>=20240930  # Zero-shot Whisper baseline
jiwer>=3.0.3              # WER / CER computation
pydub>=0.25.1             # Audio loading
webrtcvad-wheels>=2.0.10  # VAD segmentation (Python 3.12)
yt-dlp>=2024.11.18        # YouTube download
pandas>=2.2.0             # Dataset manifests
matplotlib>=3.9.0         # Plots
seaborn>=0.13.2           # Heatmaps
scipy>=1.13.0             # Statistical tests
imageio-ffmpeg>=0.5.1     # Bundled ffmpeg binary
pyctcdecode               # CTC beam search with LM (optional)
kenlm                     # n-gram LM (optional, for Step 09)
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `webrtcvad` install fails | Use `pip install webrtcvad-wheels` (pre-built for Python 3.12) |
| Whisper FP16 warning on CPU | Harmless — uses FP32 automatically |
| Facebook download fails | Must be logged in to Facebook in Chrome; cookies extracted automatically |
| `lmplz` not found | KenLM binary not on PATH — see Installation section |
| CUDA out of memory | Reduce `per_device_train_batch_size` in `fine_tuning/config.py` |
| `segments_manifest.json` path errors | Run from `dataset_pipeline/` folder, not project root |
| `gTTS not installed` error | Run `pip install gTTS` — only needed for `generate_synthetic_cs.py --tts` |
| `gTTS` network error during TTS | Needs internet connection — gTTS calls Google's servers |
| `soundfile` / `librosa` missing | Run `pip install -r requirements.txt` again — these are in the file |
| `Cannot connect to Ollama` | Run `ollama serve` in a separate terminal before running `11_llm_judge.py` |
| `model not pulled` error in judge | Run `ollama pull mistral`, `ollama pull llama3.1`, `ollama pull gemma2` |
| Judge parse error / no JSON | Model returned bad format — script retries automatically, then skips |
