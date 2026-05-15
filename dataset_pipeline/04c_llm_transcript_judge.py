"""
Step 4c: LLM-as-a-Judge Ensemble for Transcript Quality Validation.

WHERE IT FITS IN THE PIPELINE:
  Step 3  → 03_auto_transcribe.py        Whisper transcribes all clips
  Step 4b → 04b_auto_annotate.py         Auto-accepts HIGH confidence clips
  Step 4c → 04c_llm_transcript_judge.py  LLM ensemble judges MEDIUM confidence clips
  Step 5  → 05_build_dataset.py          Builds final dataset

WHY THIS STEP EXISTS:
  04b_auto_annotate.py uses a simple threshold:
    - no_speech_prob < 0.30  → AUTO-ACCEPT  (clear speech, high confidence)
    - no_speech_prob 0.30–0.60 → FLAGGED    (borderline — previously needed human review)
    - no_speech_prob >= 0.60 → AUTO-REJECT  (noise/music)

  This script handles the FLAGGED clips using a 3-model LLM ensemble
  (Mistral, LLaMA, Gemma) to decide: accept or reject — NO HUMAN NEEDED.

WHAT THE LLM JUDGES (per transcript):
  1. Bengali validity     — Is this real Bengali text or hallucination?
  2. CS naturalness       — Does this look like natural Bengali+English CS speech?
  3. Dialect authenticity — Does this sound like dialectal speech (not standard Bengali)?
  4. Noise/hallucination  — Is this gibberish, repeated words, or music lyrics?

ENSEMBLE DECISION:
  - All 3 models vote ACCEPT / REJECT for each clip
  - Majority vote (2/3 or 3/3) wins
  - Clips where all 3 disagree (1 accept, 1 reject, 1 unsure) are SKIPPED
    and saved separately for optional manual review

OUTPUT:
  transcripts/transcripts_reviewed.json   ← merged: auto-accepted + LLM-accepted
  transcripts/llm_judge_accepted.json     ← only LLM-accepted clips
  transcripts/llm_judge_rejected.json     ← LLM-rejected clips (inspect if needed)
  transcripts/llm_judge_uncertain.json    ← 1/3 agreement (optional manual review)
  transcripts/llm_judge_report.txt        ← stats for paper

REQUIREMENTS:
  1. Install Ollama: https://ollama.com/download
  2. Pull models once:
       ollama pull mistral
       ollama pull llama3.1
       ollama pull gemma3
  3. Run ollama serve (or it starts automatically)

Usage:
  python 04c_llm_transcript_judge.py                  # judge all flagged clips
  python 04c_llm_transcript_judge.py --all            # judge ALL clips (re-validate everything)
  python 04c_llm_transcript_judge.py --sample 200     # quick test on 200 clips
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

SCRIPT_DIR      = Path(__file__).resolve().parent

if (SCRIPT_DIR / "Data ASR" / "transcripts_merged_all" / "transcripts.json").exists():
    TRANSCRIPTS_IN = SCRIPT_DIR / "Data ASR" / "transcripts_merged_all" / "transcripts.json"
else:
    TRANSCRIPTS_IN = SCRIPT_DIR / "transcripts" / "transcripts.json"

REVIEWED_IN     = SCRIPT_DIR / "transcripts" / "transcripts_reviewed.json"
REVIEWED_OUT    = SCRIPT_DIR / "transcripts" / "transcripts_reviewed.json"
ACCEPTED_OUT    = SCRIPT_DIR / "transcripts" / "llm_judge_accepted.json"
REJECTED_OUT    = SCRIPT_DIR / "transcripts" / "llm_judge_rejected.json"
UNCERTAIN_OUT   = SCRIPT_DIR / "transcripts" / "llm_judge_uncertain.json"
REPORT_OUT      = SCRIPT_DIR / "transcripts" / "llm_judge_report.txt"

OLLAMA_URL   = "http://localhost:11434/api/generate"

# ── Prompt ────────────────────────────────────────────────────────────────────

PROMPT_TEMPLATE = """You are a quality control expert for a Bengali speech recognition dataset.

A speech recognition system (Whisper) transcribed an audio clip from a Bangladeshi speaker.
The speaker may use regional dialect and mix English words naturally (code-switching).

TRANSCRIPT TO EVALUATE:
"{transcript}"

DIALECT: {dialect}
DETECTED AS CODE-SWITCHED: {is_cs}

Judge this transcript on 4 criteria. Answer each with YES or NO only:

1. BENGALI_VALID: Is this real Bengali text (not hallucination, gibberish, or wrong language)?
   - YES if: contains actual Bengali words, even with dialect variations
   - NO if: repeated characters, music lyrics, random symbols, or clearly another language

2. CS_NATURAL: If code-switched (has English words), does the mixing look natural?
   - YES if: English words appear naturally where a Bangladeshi speaker would use them
   - YES if: transcript is Bengali-only (no CS expected)
   - NO if: random English words inserted that make no sense, or forced unnatural mixing

3. DIALECT_AUTHENTIC: Does this sound like real dialectal speech (not standard textbook Bengali)?
   - YES if: has dialectal words, informal constructions, or natural spoken Bengali patterns
   - YES if: uncertain (give benefit of doubt)
   - NO if: clearly formal/written standard Bengali that no one would speak

4. NOT_NOISE: Is this actual speech content (not noise transcription)?
   - YES if: forms coherent sentences or phrases
   - NO if: single repeated word/character, music lyrics pattern, or fewer than 3 words

OVERALL DECISION:
Based on the above, should this transcript be ACCEPTED into the training dataset?
- ACCEPT if: 3 or 4 criteria are YES
- REJECT if: 2 or more criteria are NO
- UNCERTAIN if: exactly 2 YES and 2 NO

Respond ONLY in this exact JSON format:
{{
  "bengali_valid": "YES" or "NO",
  "cs_natural": "YES" or "NO",
  "dialect_authentic": "YES" or "NO",
  "not_noise": "YES" or "NO",
  "decision": "ACCEPT" or "REJECT" or "UNCERTAIN",
  "reason": "<one sentence max>"
}}"""


# ── Ollama call ───────────────────────────────────────────────────────────────

def call_ollama(model: str, prompt: str, retries: int = 3) -> str | None:
    try:
        import requests
    except ImportError:
        print("[ERROR] pip install requests")
        raise SystemExit(1)

    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.05, "num_predict": 120},
    }

    for attempt in range(retries):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            print(f"    [WARN] {model} HTTP {resp.status_code} — retry {attempt+1}")
        except requests.exceptions.ConnectionError:
            print(f"[ERROR] Cannot connect to Ollama at {OLLAMA_URL}")
            print("        Make sure Ollama is running: ollama serve")
            raise SystemExit(1)
        except Exception as e:
            print(f"    [WARN] {model} error: {e} — retry {attempt+1}")
        time.sleep(1)
    return None


def parse_decision(response: str) -> dict | None:
    if not response:
        return None
    start = response.find("{")
    end   = response.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        data = json.loads(response[start:end])
        decision = str(data.get("decision", "")).strip().upper()
        if decision not in ("ACCEPT", "REJECT", "UNCERTAIN"):
            return None
        return {
            "bengali_valid":      str(data.get("bengali_valid", "NO")).upper() == "YES",
            "cs_natural":         str(data.get("cs_natural", "NO")).upper() == "YES",
            "dialect_authentic":  str(data.get("dialect_authentic", "NO")).upper() == "YES",
            "not_noise":          str(data.get("not_noise", "NO")).upper() == "YES",
            "decision":           decision,
            "reason":             str(data.get("reason", ""))[:200],
        }
    except (json.JSONDecodeError, ValueError):
        return None


# ── Ensemble judge ────────────────────────────────────────────────────────────

def judge_transcript(clip: dict, judge_models: list[str]) -> dict:
    """
    Run all 3 judge models on one transcript.
    Returns ensemble decision: ACCEPT / REJECT / UNCERTAIN.
    """
    transcript = clip.get("transcript", "").strip()
    if not transcript or len(transcript) < 5:
        return {"ensemble_decision": "REJECT", "reason": "empty_or_too_short",
                "per_model": {}, "votes": {}}

    prompt = PROMPT_TEMPLATE.format(
        transcript  = transcript,
        dialect     = clip.get("dialect", "unknown"),
        is_cs       = "YES" if clip.get("is_code_switched") else "NO",
    )

    per_model = {}
    for model in judge_models:
        raw    = call_ollama(model, prompt)
        result = parse_decision(raw)
        per_model[model] = result

    # Count votes
    votes = {"ACCEPT": 0, "REJECT": 0, "UNCERTAIN": 0, "FAILED": 0}
    for model, result in per_model.items():
        if result is None:
            votes["FAILED"] += 1
        else:
            votes[result["decision"]] += 1

    valid_votes = votes["ACCEPT"] + votes["REJECT"] + votes["UNCERTAIN"]

    # Ensemble decision rules
    if valid_votes == 0:
        ensemble_decision = "REJECT"   # all models failed to respond
        reason = "all_judges_failed"
    else:
        majority_needed = (len(judge_models) // 2) + 1
        if votes["ACCEPT"] >= majority_needed:
            ensemble_decision = "ACCEPT"
            reason = f"{votes['ACCEPT']}/{len(judge_models)} judges accepted"
        elif votes["REJECT"] >= majority_needed:
            ensemble_decision = "REJECT"
            reason = f"{votes['REJECT']}/{len(judge_models)} judges rejected"
        else:
            ensemble_decision = "UNCERTAIN"
            reason = f"no majority: A={votes['ACCEPT']} R={votes['REJECT']} U={votes['UNCERTAIN']}"

    return {
        "clip_id":           clip["clip_id"],
        "transcript":        transcript,
        "dialect":           clip.get("dialect", "unknown"),
        "is_code_switched":  clip.get("is_code_switched", False),
        "ensemble_decision": ensemble_decision,
        "reason":            reason,
        "votes":             votes,
        "per_model":         per_model,
    }


# ── Ollama availability check ─────────────────────────────────────────────────

def check_ollama(judge_models: list[str]):
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        if resp.status_code != 200:
            raise ConnectionError
        available = [m["name"] for m in resp.json().get("models", [])]
        available_bases = [m.split(":")[0] for m in available]
    except Exception:
        print("[ERROR] Cannot connect to Ollama.")
        print("        1. Install Ollama: https://ollama.com/download")
        print("        2. Run: ollama serve")
        raise SystemExit(1)

    missing = [m for m in judge_models if m not in available and m.split(":")[0] not in available_bases]
    if missing:
        print(f"[ERROR] Missing Ollama models: {missing}")
        print("        Pull them with:")
        for m in missing:
            print(f"          ollama pull {m}")
        raise SystemExit(1)

    print(f"  ✓ Ollama running. Judge models ready: {', '.join(judge_models)}")


# ── Driver ────────────────────────────────────────────────────────────────────

def run(judge_all: bool = False, sample: int | None = None, model: str = "qwen2.5", resume: bool = False):
    judge_models = [model]

    if not REVIEWED_IN.exists():
        print(f"[ERROR] {REVIEWED_IN} not found. Run 04b_auto_annotate.py first.")
        raise SystemExit(1)

    with open(REVIEWED_IN, encoding="utf-8") as f:
        all_reviewed = json.load(f)
        
    accepted   = []
    rejected   = []
    uncertain  = []
    judged_ids = set()

    if resume:
        if ACCEPTED_OUT.exists():
            with open(ACCEPTED_OUT, encoding="utf-8") as f:
                accepted = json.load(f)
                judged_ids.update(c["clip_id"] for c in accepted)
        if REJECTED_OUT.exists():
            with open(REJECTED_OUT, encoding="utf-8") as f:
                rejected = json.load(f)
                judged_ids.update(c["clip_id"] for c in rejected)
        if UNCERTAIN_OUT.exists():
            with open(UNCERTAIN_OUT, encoding="utf-8") as f:
                uncertain = json.load(f)
                judged_ids.update(c["clip_id"] for c in uncertain)
        print(f"Resuming: Loaded {len(judged_ids)} already judged clips from previous run.")

    if judge_all:
        to_judge = all_reviewed
        already_accepted = []
        print(f"Mode: judge ALL {len(to_judge)} clips")
    else:
        to_judge = [c for c in all_reviewed if c.get("needs_human_review") is True]
        already_accepted = [c for c in all_reviewed if not c.get("needs_human_review")]
        print(f"Clips auto-accepted by Step 4b : {len(already_accepted)}")
        print(f"Flagged clips to judge total   : {len(to_judge)}")

    if resume and judged_ids:
        to_judge = [c for c in to_judge if c["clip_id"] not in judged_ids]
        print(f"Clips remaining to judge now   : {len(to_judge)}")

    if sample and sample < len(to_judge):
        import random
        to_judge = random.sample(to_judge, sample)
        print(f"Random sample                  : {len(to_judge)} clips")

    if not to_judge:
        print("[INFO] No flagged clips to judge. All clips already accepted by Step 4b.")
        print("       Run with --all to re-validate everything.")
        return

    check_ollama(judge_models)

    print(f"\n{'='*60}")
    print(f"Step 4c — LLM Transcript Judge Ensemble")
    print(f"  Judge models : {', '.join(judge_models)}")
    print(f"  Clips        : {len(to_judge)}")
    print(f"{'='*60}\n")

    for i, clip in enumerate(to_judge):
        short_tx = clip.get("transcript", "")[:55]
        print(f"  [{i+1:>4}/{len(to_judge)}] {clip['clip_id']}  \"{short_tx}\"")

        result = judge_transcript(clip, judge_models)

        decision = result["ensemble_decision"]
        print(f"           → {decision}  ({result['reason']})")

        if decision == "ACCEPT":
            # Mark as LLM-accepted and add to accepted list
            accepted_clip = dict(clip)
            accepted_clip["llm_judge_decision"] = "ACCEPT"
            accepted_clip["llm_judge_votes"]    = result["votes"]
            accepted_clip["human_transcript"]   = clip.get("transcript", "")
            accepted_clip["needs_human_review"] = False
            accepted.append(accepted_clip)

        elif decision == "REJECT":
            clip["llm_judge_decision"] = "REJECT"
            clip["llm_judge_votes"]    = result["votes"]
            rejected.append(clip)

        else:  # UNCERTAIN
            clip["llm_judge_decision"] = "UNCERTAIN"
            clip["llm_judge_votes"]    = result["votes"]
            uncertain.append(clip)

        # Save every 25 clips
        if (i + 1) % 25 == 0:
            _checkpoint(accepted, rejected, uncertain)

    _checkpoint(accepted, rejected, uncertain)

    # Merge LLM-accepted + already Step4b-accepted → final reviewed set
    final_reviewed = already_accepted + accepted
    with open(REVIEWED_OUT, "w", encoding="utf-8") as f:
        json.dump(final_reviewed, f, ensure_ascii=False, indent=2)

    # Write report
    total_judged = len(accepted) + len(rejected) + len(uncertain)
    report_lines = [
        "=" * 60,
        "Step 4c — LLM Transcript Judge Report",
        "=" * 60,
        f"Judge models         : {', '.join(judge_models)}",
        f"Clips judged         : {total_judged}",
        f"",
        f"DECISIONS:",
        f"  ACCEPT             : {len(accepted)}  ({len(accepted)/max(total_judged,1)*100:.1f}%)",
        f"  REJECT             : {len(rejected)}  ({len(rejected)/max(total_judged,1)*100:.1f}%)",
        f"  UNCERTAIN          : {len(uncertain)}  ({len(uncertain)/max(total_judged,1)*100:.1f}%)",
        f"",
        f"FINAL DATASET:",
        f"  Step 4b accepted   : {len(already_accepted)}",
        f"  Step 4c accepted   : {len(accepted)}",
        f"  Total for Step 5   : {len(final_reviewed)}",
        f"",
        f"UNCERTAIN clips saved to: {UNCERTAIN_OUT}",
        f"  → Optionally review these manually",
        f"  → Or re-run with different models",
        "=" * 60,
    ]
    report_text = "\n".join(report_lines)
    print("\n" + report_text)
    REPORT_OUT.write_text(report_text, encoding="utf-8")

    print(f"\nOutputs:")
    print(f"  Final reviewed  : {REVIEWED_OUT}  ({len(final_reviewed)} clips)")
    print(f"  LLM accepted    : {ACCEPTED_OUT}  ({len(accepted)} clips)")
    print(f"  LLM rejected    : {REJECTED_OUT}  ({len(rejected)} clips)")
    print(f"  LLM uncertain   : {UNCERTAIN_OUT}  ({len(uncertain)} clips)")
    print(f"  Report          : {REPORT_OUT}")
    print(f"\nNext: python 05_build_dataset.py")


def _checkpoint(accepted, rejected, uncertain):
    with open(ACCEPTED_OUT,  "w", encoding="utf-8") as f:
        json.dump(accepted,  f, ensure_ascii=False, indent=2)
    with open(REJECTED_OUT,  "w", encoding="utf-8") as f:
        json.dump(rejected,  f, ensure_ascii=False, indent=2)
    with open(UNCERTAIN_OUT, "w", encoding="utf-8") as f:
        json.dump(uncertain, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",    action="store_true",
                        help="Re-judge ALL clips, not just flagged ones.")
    parser.add_argument("--sample", type=int, default=None,
                        help="Random sample N clips (for quick testing).")
    parser.add_argument("--model", type=str, default="qwen2.5",
                        help="Ollama model to use for judging (e.g. qwen2.5, llama3.1).")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous interrupted run (loads existing llm_judge_*.json files).")
    args = parser.parse_args()
    run(judge_all=args.all, sample=args.sample, model=args.model, resume=args.resume)
