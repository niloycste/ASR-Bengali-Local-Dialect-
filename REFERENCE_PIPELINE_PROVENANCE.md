# Whisper Large-v3 Reference-Pipeline Provenance

This record distinguishes facts recoverable from the archived AncholVox
artifacts from settings visible only in the current source code. It does not
claim that the historical construction and benchmark runs have been reproduced.

## Paired disagreement measures

All three corpus WER values use the same 4,088 historical test clip IDs and the
paper's primary surface normalization.

| Comparison | Reference | Hypothesis | WER |
|---|---|---|---:|
| Construction vs. stored silver | archived construction-time large-v3 transcript | stored silver reference | 0.0048 |
| Evaluation vs. construction | archived construction-time large-v3 transcript | archived evaluation-time large-v3 output | 0.9271 |
| Evaluation vs. stored silver | stored silver reference | archived evaluation-time large-v3 output | 0.9298 |

The 0.9271 value is pairwise disagreement between two large-v3 transcript sets.
The 0.9298 value is the benchmark WER of the evaluation-time output against the
stored silver reference. They are not the same comparison.

## Historical provenance coverage

The archived construction and evaluation rows do not contain:

- requested or resolved model revisions;
- package versions;
- complete decoding metadata; or
- audio SHA-256 values from the historical remote runs.

Consequently, the historical model revisions and software environments cannot
be recovered. The discrepancy remains unresolved.

## Current source paths

These settings describe the current code, not necessarily the historical runs.

- Label generation (`dataset_pipeline/03_auto_transcribe.py`) defaults to
  faster-whisper/CTranslate2 `large-v3`, Bengali transcription, beam size 1,
  `condition_on_previous_text=False`, word timestamps enabled, and VAD disabled.
- Zero-shot benchmark inference (`evaluation/01_run_baselines.py`) uses
  `openai/whisper-large-v3` through Hugging Face Transformers, batches of 16,
  forced Bengali transcription, and `max_new_tokens=128`.
- New benchmark runs record the requested revision, the resolved commit exposed
  by the model configuration, package versions, decoding settings, and the
  SHA-256 of each input WAV.
- The current local Hugging Face cache resolves the `main` reference for
  `openai/whisper-large-v3` to commit
  `06f233fe06e710322aca913c1bc4249a0d71fce1`. This is a current local
  environment fact and must not be attributed to either historical run.
- The current local benchmark environment reports `transformers==5.3.0`,
  `torch==2.10.0`, `datasets==4.7.0`, and `tokenizers==0.22.2`.
  `faster-whisper` is not installed in that environment. These versions did not
  generate the archived predictions.

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| `dataset_pipeline/Data ASR/transcripts_merged_all/transcripts.json` | `50FA5C5FD84EFA235AA056FBA404466A1ACCB437DBD1FE8A7FDD46735D55C959` |
| `dataset_pipeline/transcripts/transcripts_reviewed.json` | `520B6DBDB4FF05DC7032E0A2F7A8ABD26C212025879D035BE4025387855F4E3F` |
| `evaluation/results/test_set.json` | `F60672E757BE099A8CD773CC345C8CDE98694275C747AA95A9680210962498E7` |
| `evaluation/results/whisper_large_v3_predictions.json` | `70193B44F9F166F075960195E5915A13D78A332892A814035C5613BDAE08094D` |
| `evaluation/results/reference_pipeline_audit.csv` | `577758348F32C369E08B9A1BDEEF722303186DA065BDDB5BCA9962C548DEA2F7` |
| `evaluation/results/audio_sha256_manifest.csv` | `CBD755BE54C15B923FC90975ECA9DD5335083748EED1E67012C172C0C306E3EE` |

The audio manifest hashes the WAV files currently present in the workspace. It
does not prove that historical remote inference used byte-identical audio.

## Reproduction status

`evaluation/19_reference_pipeline_audit.py` reproduces the three comparisons
above from the archived text artifacts. It cannot identify a unique cause for
the disagreement. Any future rerun should pin explicit model commits, lock the
inference environment, retain per-row inference metadata, and archive paired
audio hashes.
