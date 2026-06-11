"""
Central configuration for all experiments.
All scripts import from this single file — edit here to change any setting.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
DATASET_DIR           = ROOT / "dataset_pipeline" / "final_dataset"
TRANSCRIPTS_MERGED_DIR = ROOT / "dataset_pipeline" / "Data ASR" / "transcripts_merged_all"
HF_DATA_DIR           = ROOT / "fine_tuning"  / "hf_datasets"
MODELS_DIR            = ROOT / "fine_tuning"  / "models"
RESULTS_DIR           = ROOT / "evaluation"   / "results"
PLOTS_DIR             = ROOT / "evaluation"   / "plots"
LM_DIR                = ROOT / "fine_tuning"  / "language_models"

# ── Training subsets ──────────────────────────────────────────────────────────
SUBSETS = {
    "all":     "All clips (BN_ONLY + CS)   — proposed full model",
    "bn_only": "Bengali-only dialect clips  — ablation A",
    "cs_only": "Code-switched clips only    — ablation B",
}

# ── Whisper fine-tune base ────────────────────────────────────────────────────
BASE_MODEL = "openai/whisper-small"   # fast, feasible on free Colab. Use whisper-medium/large-v3 on an A100.
LANGUAGE   = "bengali"
TASK       = "transcribe"

# ── Training hyperparameters ───────────────────────────────────────────────────
TRAINING = {
    "num_train_epochs":             3,
    "per_device_train_batch_size":  16,    # A100/H100 have plenty of VRAM for these small models
    "per_device_eval_batch_size":   16,
    "gradient_accumulation_steps":  1,     # effective batch 16 (was 8x2); raise batch further on big GPUs
    "learning_rate":                1e-5,
    "warmup_steps":                 500,
    "fp16":                         True,   # auto-upgraded to bf16 on A100/H100 by the train scripts
    "dataloader_num_workers":       2,      # parallel audio decode; keep low when reading from Google Drive
    "predict_with_generate":        True,
    "generation_max_length":        225,
    "save_steps":                   2000,   # Increased to avoid long CPU pauses
    "eval_steps":                   2000,   # Increased to avoid long CPU pauses
    "save_total_limit":             3,      # keep only best + 2 recent checkpoints (saves disk)
    "logging_steps":                50,
    "load_best_model_at_end":       True,
    "metric_for_best_model":        "wer",
    "greater_is_better":            False,
    "push_to_hub":                  False,
    "report_to":                    "none",
}

# ── Whisper model sizes (Experiment 5 — size comparison) ──────────────────────
# All evaluated zero-shot AND fine-tuned on "all" subset.
# Shows accuracy vs compute tradeoff for practitioners.
WHISPER_SIZES = {
    "whisper_tiny":   {"model": "tiny",    "hf_id": "openai/whisper-tiny",    "params": "39M"},
    "whisper_base":   {"model": "base",    "hf_id": "openai/whisper-base",    "params": "74M"},
    "whisper_small":  {"model": "small",   "hf_id": "openai/whisper-small",   "params": "244M"},
    "whisper_medium": {"model": "medium",  "hf_id": "openai/whisper-medium",  "params": "769M"},
    "whisper_large":  {"model": "large-v3","hf_id": "openai/whisper-large-v3","params": "1550M"},
}

# ── Zero-shot baseline models ─────────────────────────────────────────────────
BASELINES = {
    "whisper_small_zs": {
        "type":  "whisper_openai",
        "model": "small",
        "label": "Whisper small (zero-shot)",
    },
    "whisper_large_v3": {
        "type":  "whisper_openai",
        "model": "large-v3",
        "label": "Whisper large-v3 (zero-shot)",
    },
    "mms_bn": {
        "type":  "mms",
        "model": "facebook/mms-1b-all",
        "lang":  "ben",
        "label": "Meta MMS-1B (multilingual zero-shot)",
    },
    "wav2vec2_bn": {
        "type":  "wav2vec2",
        "model": "arijitx/wav2vec2-xls-r-300m-bengali",
        "label": "wav2vec2-XLS-R-300M-BN (standard Bengali)",
    },
}

# ── Fine-tuned models ──────────────────────────────────────────────────────────
FINETUNED = {
    # Whisper (seq2seq) — three training subsets
    "ft_whisper_all":     "Whisper fine-tuned ALL (proposed system)",
    "ft_whisper_bn_only": "Whisper fine-tuned BN_ONLY (ablation A)",
    "ft_whisper_cs_only": "Whisper fine-tuned CS only (ablation B)",
    # CTC models fine-tuned on "all"
    "ft_wav2vec2_all":    "wav2vec2-XLS-R fine-tuned ALL (CTC baseline)",
    "ft_wavlm_all":       "WavLM-large fine-tuned ALL   (CTC baseline)",
    # CTC + LM shallow fusion
    "ft_wav2vec2_lm":     "wav2vec2-XLS-R + LM shallow fusion",
    "ft_wavlm_lm":        "WavLM-large + LM shallow fusion",
}

# ── CTC model configs ──────────────────────────────────────────────────────────
CTC_MODELS = {
    "wav2vec2": {
        "model":   "facebook/wav2vec2-xls-r-300m",
        "label":   "wav2vec2-XLS-R-300M",
        "out_key": "ft_wav2vec2_all",
    },
    "wavlm": {
        "model":   "microsoft/wavlm-large",
        "label":   "WavLM-Large",
        "out_key": "ft_wavlm_all",
    },
}

# ── LM shallow fusion config ──────────────────────────────────────────────────
# n-gram LM trained on your transcript corpus and used with CTC beam search.
LM_CONFIG = {
    "ngram_order":      5,       # 5-gram LM — good balance for Bengali+EN
    "alpha":            0.5,     # LM weight  (tune on dev set: 0.1 – 1.0)
    "beta":             1.5,     # word insertion bonus (tune: 0.5 – 2.0)
    "beam_width":       100,     # beam search width
    "lm_corpus_splits": ["train", "dev"],  # build LM from training transcripts
}

# ── Statistical analysis ──────────────────────────────────────────────────────
BOOTSTRAP_ITERS  = 10_000
CONFIDENCE_LEVEL = 0.95

# ── Display order for all tables and plots ────────────────────────────────────
TABLE_ORDER = [
    "whisper_small_zs",
    "whisper_large_v3",
    "mms_bn",
    "wav2vec2_bn",
    "ft_wav2vec2_all",
    "ft_wav2vec2_lm",
    "ft_wavlm_all",
    "ft_wavlm_lm",
    "ft_whisper_bn_only",
    "ft_whisper_cs_only",
    "ft_whisper_all",
]

MODEL_LABELS = {
    "whisper_small_zs":    "Whisper small (zero-shot)",
    "whisper_large_v3":    "Whisper large-v3 (zero-shot)",
    "mms_bn":              "MMS-1B (zero-shot)",
    "wav2vec2_bn":         "wav2vec2-XLS-R-BN (zero-shot)",
    "ft_wav2vec2_all":     "wav2vec2 fine-tuned (no LM)",
    "ft_wav2vec2_lm":      "wav2vec2 + LM fusion",
    "ft_wavlm_all":        "WavLM fine-tuned (no LM)",
    "ft_wavlm_lm":         "WavLM + LM fusion",
    "ft_whisper_bn_only":  "Whisper FT BN_ONLY (ablation A)",
    "ft_whisper_cs_only":  "Whisper FT CS only (ablation B)",
    "ft_whisper_all":      "Whisper FT ALL (proposed)",
}

# Color scheme for plots:
#   grey tones  = zero-shot baselines
#   blue tones  = CTC fine-tuned
#   light green = Whisper ablations
#   dark green  = proposed system
MODEL_COLORS = {
    "whisper_small_zs":    "#BDBDBD",
    "whisper_large_v3":    "#9E9E9E",
    "mms_bn":              "#BDBDBD",
    "wav2vec2_bn":         "#B0BEC5",
    "ft_wav2vec2_all":     "#90CAF9",
    "ft_wav2vec2_lm":      "#42A5F5",
    "ft_wavlm_all":        "#64B5F6",
    "ft_wavlm_lm":         "#1565C0",
    "ft_whisper_bn_only":  "#A5D6A7",
    "ft_whisper_cs_only":  "#66BB6A",
    "ft_whisper_all":      "#2E7D32",
}
