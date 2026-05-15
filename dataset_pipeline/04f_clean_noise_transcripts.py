"""
ASR Noise Token Cleaner for Bengali Transcripts
================================================
Strategies:
  1. Delete pure noise tokens (garbage hallucinations)
  2. BERT masked language model to predict real words from context
  3. Fuzzy match noise tokens to a real word dictionary

Requirements:
    pip install transformers torch rapidfuzz

Usage:
    python asr_noise_cleaner.py
    OR import and use the ASRCleaner class directly.
"""

import re
import json
import argparse
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR / "Data ASR" / "transcripts_merged_all" / "transcripts_reviewed.json").exists():
    TARGET_DIR = SCRIPT_DIR / "Data ASR" / "transcripts_merged_all"
else:
    TARGET_DIR = SCRIPT_DIR / "transcripts"

REVIEWED_PATH = TARGET_DIR / "transcripts_reviewed.json"
NOISE_LEXICON_PATH = SCRIPT_DIR / "lexicons" / "noise_tokens.txt"

TOKEN_RE = re.compile(r"[\u0900-\u09ffA-Za-z][\u0900-\u09ffA-Za-z'\-]*", re.UNICODE)

# ──────────────────────────────────────────────
# 1. LOAD NOISE TOKENS
# ──────────────────────────────────────────────

def load_noise_tokens(filepath: str | Path) -> set:
    """Load noise tokens from file into a set for O(1) lookup."""
    noise = set()
    filepath = Path(filepath)
    if not filepath.exists():
        print(f"[⚠] Noise lexicon not found at {filepath}")
        return noise
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                noise.add(line.casefold())
    print(f"[✔] Loaded {len(noise)} noise tokens from '{filepath.name}'")
    return noise


# ──────────────────────────────────────────────
# 2. STRATEGY 1 — DELETE NOISE TOKENS
# ──────────────────────────────────────────────

def delete_noise_tokens(text: str, noise_set: set) -> str:
    """
    Simply remove every noise token from the text.
    Best for: pure garbage tokens like এইইইই, লললললল, etc.
    """
    def replace_match(m):
        return "" if m.group(0).casefold() in noise_set else m.group(0)
    cleaned = TOKEN_RE.sub(replace_match, text)
    return re.sub(r'\s+', ' ', cleaned).strip()


# ──────────────────────────────────────────────
# 3. STRATEGY 2 — BERT CONTEXT FILL (MASK + PREDICT)
# ──────────────────────────────────────────────

class BERTFiller:
    """
    Uses Bengali BERT to predict what a noise token should be,
    based on the surrounding words (context).

    Model: sagorsarker/bangla-bert-base (HuggingFace)
    """

    def __init__(self, model_name: str = "sagorsarker/bangla-bert-base"):
        print(f"[⏳] Loading BERT model: {model_name} ...")
        from transformers import pipeline
        self.pipe = pipeline(
            "fill-mask",
            model=model_name,
            top_k=1  # return only the best prediction
        )
        print("[✔] BERT model loaded.")

    def predict(self, text_with_mask: str) -> str:
        """
        Replace [MASK] in the text and return the predicted word.

        Example:
            Input:  "আমি [MASK] যাচ্ছি"
            Output: "বাড়ি"
        """
        results = self.pipe(text_with_mask)
        if results:
            return results[0]["token_str"].strip()
        return ""

    def fill_noise_tokens(self, text: str, noise_set: set) -> str:
        """
        For each noise token found in the text, replace it with [MASK]
        and ask BERT to predict the correct word using context.
        """
        tokens = list(TOKEN_RE.finditer(text))
        if not tokens:
            return text
            
        new_text = text
        offset = 0
        
        for match in tokens:
            word = match.group(0)
            if word.casefold() in noise_set:
                start = match.start() + offset
                end = match.end() + offset
                masked_text = new_text[:start] + "[MASK]" + new_text[end:]
                
                predicted = self.predict(masked_text)
                if predicted:
                    print(f"  [BERT] '{word}' → '{predicted}'")
                    new_text = new_text[:start] + predicted + new_text[end:]
                    offset += len(predicted) - len(word)
                else:
                    new_text = new_text[:start] + "" + new_text[end:]
                    offset -= len(word)
                    
        return re.sub(r'\s+', ' ', new_text).strip()


# ──────────────────────────────────────────────
# 4. STRATEGY 3 — FUZZY MATCH TO DICTIONARY
# ──────────────────────────────────────────────

class FuzzyMatcher:
    """
    Matches noise tokens to the closest real word in a dictionary
    using edit distance (Levenshtein).

    Best for: tokens that are close misspellings of real words.
    e.g.  কমপানী  →  কোম্পানি
    """

    def __init__(self, dictionary: list, threshold: int = 80):
        """
        Args:
            dictionary: list of valid Bengali words
            threshold:  minimum similarity score (0-100). 
                        Lower = more aggressive matching.
                        Recommended: 75-85
        """
        from rapidfuzz import process, fuzz
        self.process = process
        self.fuzz = fuzz
        self.dictionary = dictionary
        self.threshold = threshold
        print(f"[✔] Fuzzy matcher ready with {len(dictionary)} words (threshold={threshold})")

    def find_closest(self, token: str) -> Optional[str]:
        """Return the closest dictionary word, or None if below threshold."""
        result = self.process.extractOne(
            token,
            self.dictionary,
            scorer=self.fuzz.ratio
        )
        if result and result[1] >= self.threshold:
            return result[0]
        return None

    def fix_noise_tokens(self, text: str, noise_set: set) -> str:
        """Replace noise tokens with their closest dictionary match."""
        def replace_match(m):
            word = m.group(0)
            if word.casefold() in noise_set:
                match = self.find_closest(word.casefold())
                if match:
                    print(f"  [FUZZY] '{word}' → '{match}'")
                    return match
                return ""
            return word
            
        cleaned = TOKEN_RE.sub(replace_match, text)
        return re.sub(r'\s+', ' ', cleaned).strip()


# ──────────────────────────────────────────────
# 5. COMBINED PIPELINE
# ──────────────────────────────────────────────

class ASRCleaner:
    """
    Full pipeline combining all 3 strategies.

    Flow:
        Raw transcript
            ↓
        [Step 1] Identify noise tokens
            ↓
        [Step 2] Try BERT context fill (if enabled)
            ↓
        [Step 3] Try fuzzy match for remaining (if enabled)
            ↓
        [Step 4] Delete anything still not resolved
            ↓
        Clean transcript ✅
    """

    def __init__(
        self,
        noise_file: str,
        use_bert: bool = True,
        use_fuzzy: bool = True,
        dictionary: Optional[list] = None,
        bert_model: str = "sagorsarker/bangla-bert-base",
        fuzzy_threshold: int = 80,
    ):
        self.noise_set = load_noise_tokens(noise_file)

        self.bert_filler = None
        if use_bert:
            try:
                self.bert_filler = BERTFiller(model_name=bert_model)
            except Exception as e:
                print(f"[⚠] Could not load BERT model: {e}")
                print("    Continuing without BERT fill.")

        self.fuzzy_matcher = None
        if use_fuzzy and dictionary:
            try:
                self.fuzzy_matcher = FuzzyMatcher(dictionary, threshold=fuzzy_threshold)
            except ImportError:
                print("[⚠] rapidfuzz not installed. Run: pip install rapidfuzz")
                print("    Continuing without fuzzy matching.")

    def clean(self, text: str, verbose: bool = True) -> str:
        """
        Clean a single transcript string.

        Args:
            text:    Raw ASR transcript
            verbose: Print what replacements were made

        Returns:
            Cleaned transcript string
        """
        if verbose:
            print(f"\n[INPUT]  {text}")

        # Check if any noise tokens exist in this text
        words = set(text.split())
        has_noise = bool(words & self.noise_set)

        if not has_noise:
            if verbose:
                print("[OUTPUT] (no noise found)")
            return text

        result = text

        # Step 1: BERT fill (uses context to predict real word)
        if self.bert_filler:
            result = self.bert_filler.fill_noise_tokens(result, self.noise_set)

        # Step 2: Fuzzy match (for anything BERT didn't fix)
        if self.fuzzy_matcher:
            result = self.fuzzy_matcher.fix_noise_tokens(result, self.noise_set)

        # Step 3: Delete anything still remaining
        result = delete_noise_tokens(result, self.noise_set)

        if verbose:
            print(f"[OUTPUT] {result}")

        return result

    def clean_pipeline_json(self, json_path: Path):
        """Iterate over transcripts_reviewed.json and clean the texts."""
        if not json_path.exists():
            print(f"[ERROR] '{json_path}' not found. Run 04b_auto_annotate.py first.")
            return
            
        with open(json_path, encoding="utf-8") as f:
            clips = json.load(f)
            
        print(f"\n[⏳] Scanning {len(clips)} clips in '{json_path.name}' ...")
        cleaned_count = 0
        
        for clip in clips:
            text = clip.get("human_transcript", "")
            if text:
                cleaned_text = self.clean(text, verbose=True)
                if cleaned_text != text:
                    clip["human_transcript"] = cleaned_text
                    cleaned_count += 1
                    
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(clips, f, ensure_ascii=False, indent=2)
            
        print(f"\n[✔] Cleaned {cleaned_count} clips. Saved to '{json_path}'")


# ──────────────────────────────────────────────
# 6. EXAMPLE USAGE
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean noise tokens from pipeline transcripts.")
    parser.add_argument("--use-bert", action="store_true", help="Use Bengali BERT to guess missing words")
    parser.add_argument("--use-fuzzy", action="store_true", help="Use Fuzzy Matching for misspelled words")
    args = parser.parse_args()

    print("=" * 60)
    print("PIPELINE ASR NOISE CLEANER")
    print("=" * 60)
    print(f"Target Directory : {TARGET_DIR}")
    print(f"Noise Lexicon    : {NOISE_LEXICON_PATH}")
    print(f"Reviewed JSON    : {REVIEWED_PATH.name}")
    print("-" * 60)

    cleaner = ASRCleaner(
        noise_file=NOISE_LEXICON_PATH,
        use_bert=args.use_bert,
        use_fuzzy=args.use_fuzzy,
        dictionary=[], # Standard dictionary omitted by default for speed
        bert_model="sagorsarker/bangla-bert-base",
        fuzzy_threshold=80,
    )

    cleaner.clean_pipeline_json(REVIEWED_PATH)