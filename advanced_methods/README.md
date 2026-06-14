# Advancing BanglaMix: Method Contributions + Empirical Study

This folder turns the BanglaMix **dataset/benchmark** paper into an **empirical
research paper with method novelty**. Both methods are motivated directly by
failures your own analysis revealed, and both reuse the data + models you already
have (only new training runs on your A100 are needed).

## Research questions (the empirical skeleton)
- **RQ1 — diagnosis:** *Why* do fine-tuned models still delete English words on
  code-switched clips? (You already answer this: English-word survival ≈ 0.00.)
- **RQ2 — Method 1:** Does **CS-aware (English-weighted) fine-tuning** reduce
  English deletion (↑ CS-F1, ↑ EN-survival) without hurting WER?
- **RQ3 — Method 2:** Can **LLM-judge-guided self-training** improve accuracy
  under silver-standard labels (↓ WER/CER across iterations)?
- **RQ4 (optional):** Does **dialect conditioning** close the per-dialect gap
  (Sylhet, Sandwip)?

Answering RQ2–RQ3 with a proposed method + ablation + significance is what moves
this from a resource paper toward Interspeech/ICASSP/a Q1 journal.

---

## Method 1 — CS-aware (English-weighted) fine-tuning  ⭐ headline

**Idea.** Standard Whisper fine-tuning uses uniform token cross-entropy, so
dropping a rare English token is barely penalized → the model deletes English.
We **up-weight every Latin-script (English) target token** in the loss by a factor
`en_weight`, directly penalizing English deletion. No architecture change.

**Run the ablation (both):**
```bash
cd advanced_methods

# baseline: standard cross-entropy
python method1_cs_aware.py --method baseline \
    --out ../fine_tuning/models/ft_whisper_baseline

# proposed: English-weighted cross-entropy
python method1_cs_aware.py --method cs_aware --en_weight 3.0 \
    --out ../fine_tuning/models/ft_whisper_csaware
```
Try a small sweep of `--en_weight` (2.0 / 3.0 / 5.0) for a sensitivity curve.

**Evaluate both** exactly like the paper models:
```bash
cd ../evaluation
python 02_run_finetuned.py        # add the two new model keys to config first
python 03_compute_metrics.py
python 06_statistical_analysis.py # significance of CS-aware vs baseline
python 08_error_analysis.py       # EN-survival before/after
```

**Expected result / the claim:** CS-aware ≈ baseline on overall WER but
**clearly higher CS-F1 and English-word survival** — a clean "our method fixes the
documented failure" story with significance.

> To register the models for evaluation, add to `fine_tuning/config.py`:
> ```python
> FINETUNED["ft_whisper_baseline"] = "Whisper FT (baseline CE)"
> FINETUNED["ft_whisper_csaware"]  = "Whisper FT (CS-aware, proposed)"
> ```

---

## Method 2 — LLM-judge-guided self-training (recipe)

**Idea.** Your labels are silver (Whisper-generated). Iteratively refine them:
each round, the *current* fine-tuned model re-transcribes the data, the **Qwen2.5
LLM judge** keeps only high-quality pseudo-labels, and the model is retrained.
The novel twist vs. GigaSpeech-2-style pipelines is the **LLM judge in the loop**.
Report **WER/CER vs. iteration** — a downward curve is the contribution.

**Recipe using your existing scripts (per round `r = 1..N`):**
```bash
# 1. Re-transcribe the training audio with the CURRENT model
#    (point 03_auto_transcribe.py / a Whisper inference at fine_tuning/models/ft_whisper_csaware)

# 2. Judge-filter the new pseudo-labels (keep accepted)
python dataset_pipeline/04c_llm_transcript_judge.py --model qwen2.5

# 3. Rebuild the training set from the refined labels
python dataset_pipeline/05_build_dataset.py
python fine_tuning/01_prepare_hf_dataset.py

# 4. Retrain + evaluate
python advanced_methods/method1_cs_aware.py --method cs_aware \
    --out fine_tuning/models/ft_whisper_selftrain_r{r}
python evaluation/02_run_finetuned.py && python evaluation/03_compute_metrics.py
```
Plot the per-iteration WER/CER → that figure is the empirical evidence for RQ3.

---

## Method 3 (optional) — Dialect conditioning
Prepend a **dialect-ID token** to the input (or add per-dialect **LoRA adapters**)
and show the hardest dialects (Sylhet, Sandwip) improve vs. the monolithic model.
Light to run (LoRA), gives a per-dialect delta table.

---

## Folding results into the paper
Once Method 1 (and ideally Method 2) produce real numbers:
1. Add a **"Proposed Method"** section before Results (motivation → method → why).
2. Add a results row: **baseline vs. CS-aware** (WER/CER/**CS-F1**/**EN-survival**) + p-value.
3. Reframe the abstract/intro from "we release a benchmark" to "we release a
   benchmark **and** propose a CS-aware training method that reduces English
   deletion, validated on it."

Tell the assistant the numbers and it will wire them into `asr_paper_ieee/main.tex`.

## ⚠️ Honest notes
- These are **real experiments** — they need GPU time and produce new numbers; no
  results exist until you run them.
- Do the **human validation first**: it gives a gold reference to confirm the new
  method actually helps (not just chases silver-label noise).
- `method1_cs_aware.py` trains on the **released real clips** by default;
  add augmentation if you want to match the paper's 60,936-clip train pool.
