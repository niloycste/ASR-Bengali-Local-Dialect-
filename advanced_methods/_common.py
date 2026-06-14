"""
Shared, robust training utilities for the BanglaMix advanced-method scripts
(method1_cs_aware.py, method2_self_training.py, method3_dialect_tagged.py).

Design choices for robustness:
  * Log-mel features are extracted ON THE FLY in the collator (the cached dataset
    stays tiny -- no tens-of-GB feature cache).
  * TrainingArguments are version-tolerant (eval_strategy vs evaluation_strategy).
  * Whisper bos handling and 'validation'/'dev' split naming are guarded.
  * A single finetune() function is reused by all three methods.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer

try:
    from fine_tuning.config import LANGUAGE, TASK
except Exception:
    LANGUAGE, TASK = "bengali", "transcribe"

DEFAULT_DATASET = "niloycste68/Bangali_local_dialect_ASR_HF_Dataset"


# ── English-token weights (Method 1) ──────────────────────────────────────────
def build_token_weights(tokenizer, en_weight: float, bn_weight: float = 1.0) -> torch.Tensor:
    """en_weight for vocab tokens containing Latin letters (English sub-words),
    else bn_weight. Whisper's byte-level BPE puts English bytes as [A-Za-z] and
    Bengali bytes outside that range, so this cleanly separates the scripts."""
    latin = re.compile(r"[A-Za-z]")
    n = len(tokenizer)
    weights = torch.full((n,), float(bn_weight))
    for tid, tok in enumerate(tokenizer.convert_ids_to_tokens(list(range(n)))):
        if tok and latin.search(tok):
            weights[tid] = float(en_weight)
    print(f"[weights] {int((weights == en_weight).sum())}/{n} tokens flagged "
          f"English (w={en_weight}); rest w={bn_weight}")
    return weights


# ── Collator (on-the-fly features) ────────────────────────────────────────────
@dataclass
class SpeechCollator:
    processor: Any

    def __call__(self, features):
        fe, tok = self.processor.feature_extractor, self.processor.tokenizer
        feats = [{"input_features": fe(f["audio"]["array"],
                                       sampling_rate=16000).input_features[0]}
                 for f in features]
        batch = fe.pad(feats, return_tensors="pt")
        labs = tok.pad([{"input_ids": f["labels"]} for f in features],
                       return_tensors="pt")
        labels = labs["input_ids"].masked_fill(labs.attention_mask.ne(1), -100)
        bos = tok.bos_token_id
        if bos is not None and (labels[:, 0] == bos).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


# ── Weighted trainer (token_weights=None -> standard CE) ───────────────────────
class WeightedTrainer(Seq2SeqTrainer):
    def __init__(self, *args, token_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.token_weights = token_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        V = logits.size(-1)
        per_tok = F.cross_entropy(logits.reshape(-1, V), labels.reshape(-1),
                                  ignore_index=-100, reduction="none")
        flat = labels.reshape(-1)
        mask = (flat != -100).float()
        w = mask if self.token_weights is None \
            else self.token_weights.to(logits.device)[flat.clamp(min=0)] * mask
        loss = (per_tok * w).sum() / w.sum().clamp(min=1.0)
        return (loss, outputs) if return_outputs else loss


# ── Model / processor ─────────────────────────────────────────────────────────
def build_model_processor(base_model: str):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    processor = WhisperProcessor.from_pretrained(base_model, language=LANGUAGE, task=TASK)
    model = WhisperForConditionalGeneration.from_pretrained(base_model)
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    return model, processor


# ── Data loading ──────────────────────────────────────────────────────────────
def prepare_for_training(ds, processor, label_text_fn: Callable | None = None):
    """Cast audio to 16 kHz and tokenize the (optionally rewritten) target text;
    returns a dataset with only {audio, labels}. Works on a HF-hub split OR on an
    in-memory dataset built during self-training."""
    from datasets import Audio
    try:
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    except Exception:
        pass  # already a plain {array, ...} dict
    tok = processor.tokenizer
    text_fn = label_text_fn or (lambda e: e["sentence"])
    ds = ds.map(lambda e: {"labels": tok(text_fn(e)).input_ids})
    keep = {"audio", "labels"}
    return ds.remove_columns([c for c in ds.column_names if c not in keep])


def load_split(processor, dataset_id, split, subset="all", max_samples=None,
               label_text_fn: Callable | None = None):
    """Returns a dataset with columns {audio, labels}. `label_text_fn(example)`
    lets a method rewrite the target text (e.g. prepend a dialect tag)."""
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split=split)
    if subset == "bn_only":
        ds = ds.filter(lambda x: not x.get("is_code_switched", False))
    elif subset == "cs_only":
        ds = ds.filter(lambda x: x.get("is_code_switched", False))
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return prepare_for_training(ds, processor, label_text_fn)


def load_eval_split(processor, dataset_id, subset="all", label_text_fn=None):
    try:
        return load_split(processor, dataset_id, "validation", subset,
                          label_text_fn=label_text_fn)
    except Exception:
        return load_split(processor, dataset_id, "dev", subset,
                          label_text_fn=label_text_fn)


# ── Reusable fine-tune (used by all three methods) ────────────────────────────
def finetune(model, processor, train_ds, eval_ds, out_dir, token_weights=None,
             epochs=3, batch=16, lr=1e-5, warmup=500):
    from transformers import Seq2SeqTrainingArguments
    import evaluate

    wer = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids, label_ids = pred.predictions, pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        p = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        r = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return {"wer": wer.compute(predictions=p, references=r)}

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ta = dict(
        output_dir=out_dir,
        per_device_train_batch_size=batch, per_device_eval_batch_size=batch,
        gradient_accumulation_steps=1, learning_rate=lr, warmup_steps=warmup,
        num_train_epochs=epochs,
        bf16=bf16, fp16=not bf16 and torch.cuda.is_available(),
        save_strategy="epoch", predict_with_generate=True, generation_max_length=225,
        logging_steps=50, save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="wer", greater_is_better=False, report_to="none",
    )
    try:
        targs = Seq2SeqTrainingArguments(eval_strategy="epoch", **ta)
    except TypeError:
        targs = Seq2SeqTrainingArguments(evaluation_strategy="epoch", **ta)

    trainer = WeightedTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=SpeechCollator(processor), compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor, token_weights=token_weights,
    )
    trainer.train()
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)
    return trainer


# ── Batched transcription (used by Method 2 self-training) ────────────────────
@torch.no_grad()
def transcribe_clips(model, processor, hf_split, batch_size=16):
    """Return a list of hypothesis strings for a HF split with an 'audio' column."""
    from datasets import Audio
    hf_split = hf_split.cast_column("audio", Audio(sampling_rate=16000))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    fe, tok = processor.feature_extractor, processor.tokenizer
    hyps = []
    for i in range(0, len(hf_split), batch_size):
        rows = hf_split[i:i + batch_size]
        feats = fe([a["array"] for a in rows["audio"]], sampling_rate=16000,
                   return_tensors="pt").input_features.to(device)
        gen = model.generate(feats, max_length=225)
        hyps.extend(tok.batch_decode(gen, skip_special_tokens=True))
    return hyps
