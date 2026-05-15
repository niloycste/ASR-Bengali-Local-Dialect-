"""
Step 1: Download audio from YouTube and TikTok.

Target: Bengali LOCAL DIALECT speech mixed with English (code-switching).
Covers all 16 dialect regions:
  Old_Dhaka, Comilla, Chittagong, Sylhet, Barishal, Noakhali,
  Rangpur, Mymensingh, Khulna, Kushtia, Tangail, Kishoreganj, Habiganj,
  Narail, Narsingdi, Sandwip
  + Sylhet_UK_CS (British-Bangladeshi diaspora — very high CS rate)

QUERY STRATEGY (updated for maximum CS yield):
  Each dialect has queries across 4 CS-inducing domains:
    1. Vlog / content creator — young YouTubers speaking their home dialect
    2. Tech / phone review     — English brand/tech terms naturally mixed in
    3. Street interview        — young adults asked about modern topics
    4. Social media / career   — "viral", "trending", "internship", job talk
  Plus Sylhet_UK_CS — British-born Sylheti speakers, heaviest CS in corpus.

Transcript language is forced to Bengali (bn) in 03_auto_transcribe.py.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from download_query_profiles import YOUTUBE_BUCKETS, YOUTUBE_FOLDER_METADATA, write_query_catalog
from pipeline_utils import require_command_runs, require_commands, warn_if_missing_js_runtime

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR   = str(SCRIPT_DIR / "raw_audio")
SAMPLE_RATE  = 16000
AUDIO_FORMAT = "wav"


# ══════════════════════════════════════════════════════════════════════════════
# FACEBOOK — add URLs manually (yt-dlp cannot search Facebook by keyword)
# How: search Facebook for dialect videos, copy URL, paste below
# ══════════════════════════════════════════════════════════════════════════════
FACEBOOK_URLS = {
    "Chittagong": [],
    "Sylhet":     [],
    "Barishal":   [],
    "Noakhali":   [],
    "Rangpur":    [],
    "Mymensingh": [],
    "Khulna":     [],
    "Kushtia":    [],
    "Tangail":    [],
}


# ══════════════════════════════════════════════════════════════════════════════
# YOUTUBE — legacy flat query list kept as a fallback/reference.
# The active CS-first bucket strategy now comes from download_query_profiles.py
# and is wired in near the TikTok section plus the __main__ block below.
#
# Each bucket has TWO types of queries:
#   Type A — dialect + English mixed (code-switching target)
#   Type B — pure dialect speech (baseline, CS clips filtered later)
#
# Queries are in Bengali script so YouTube returns local content.
# English words added to query text help surface CS videos.
# ══════════════════════════════════════════════════════════════════════════════
YOUTUBE_QUERIES = {

    # ═══════════════════════════════════════════════════════════════════════════
    # QUERY DESIGN PRINCIPLES:
    #
    #  BAD:  "[city] vlog english"   → creator FROM that city, speaks standard
    #        Bengali for wider reach. Region ≠ dialect.
    #
    #  GOOD: "[dialect NAME] ভাষায় + [modern topic]"
    #        "আঞ্চলিক ভাষায়" + topic
    #        Comedy/roast/prank  → natural speech, cannot fake dialect
    #        Tech/unboxing       → English brand names forced in naturally
    #        Reaction            → casual dialect while watching English content
    #        Campus/job          → "internship", "assignment", "interview" CS words
    #
    #  5 angles per dialect (12 queries each):
    #    A — dialect-name + tech/phone review      (English brand/spec terms)
    #    B — dialect-name + comedy/roast/prank     (natural unscripted dialect)
    #    C — dialect-name + reaction/gaming        (English game/content terms)
    #    D — আঞ্চলিক ভাষায় + campus/career       (internship, job, English words)
    #    E — আঞ্চলিক ভাষায় + vlog/day-in-my-life (casual mixed speech)
    #
    #  yt-dlp filters applied in download_youtube():
    #    --dateafter 20210101   → recent content only (post-2021 = more CS)
    #    --match-filter "view_count > 500" → filters out zero-audience uploads
    # ═══════════════════════════════════════════════════════════════════════════

    # ── 1. Old Dhaka (Dhakaiya Kutthi) ───────────────────────────────────────
    # "ঢাকাইয়া" is unique — only appears in videos consciously using the dialect.
    "Old_Dhaka": [
        # A — dialect name + tech (English specs forced in)
        "ঢাকাইয়া ভাষায় phone review english বাংলা",
        "ঢাকাইয়া ভাষায় smartphone unboxing english",
        # B — comedy/roast (unscripted, cannot fake dialect)
        "ঢাকাইয়া ভাষায় comedy roast english mixed",
        "ঢাকাইয়া কুট্টি ভাষায় funny prank english",
        # C — reaction/gaming (English content + dialect commentary)
        "ঢাকাইয়া ভাষায় reaction video english",
        "ঢাকাইয়া ভাষায় gaming english বাংলা",
        # D — campus/career (internship, job, assignment CS words)
        "পুরান ঢাকার আঞ্চলিক ভাষায় university student english",
        "ঢাকাইয়া ভাষায় job interview career english mixed",
        # E — day-in-my-life / casual vlog
        "ঢাকাইয়া ভাষায় day in my life vlog english",
        "পুরান ঢাকার আঞ্চলিক ভাষায় vlog english banglish",
        "dhakaiya bhasha english comedy vlog mixed 2023",
        "ঢাকাইয়া ভাষায় challenge english বাংলা mixed",
    ],

    # ── 2. Comilla ───────────────────────────────────────────────────────────
    "Comilla": [
        # A
        "কুমিল্লার আঞ্চলিক ভাষায় phone review english",
        "কুমিল্লার আঞ্চলিক ভাষায় tech unboxing english",
        # B
        "কুমিল্লার আঞ্চলিক ভাষায় comedy roast english",
        "কুমিল্লার ভাষায় funny prank english বাংলা",
        # C
        "কুমিল্লার আঞ্চলিক ভাষায় reaction video english",
        "কুমিল্লার ভাষায় gaming english বাংলা",
        # D
        "কুমিল্লার আঞ্চলিক ভাষায় university campus english",
        "কুমিল্লার ভাষায় job interview career english mixed",
        # E
        "কুমিল্লার আঞ্চলিক ভাষায় vlog english banglish",
        "কুমিল্লার ভাষায় day in my life english",
        "comilla dialect bangla english comedy vlog 2023",
        "কুমিল্লার আঞ্চলিক ভাষা challenge english mixed",
    ],

    # ── 3. Chittagong (Chatgaiya) ─────────────────────────────────────────────
    # "চাটগাঁইয়া" — strongest unique dialect name, large YouTube community.
    "Chittagong": [
        # A
        "চাটগাঁইয়া ভাষায় phone review english বাংলা",
        "চাটগাঁইয়া ভাষায় tech unboxing gadget english",
        # B
        "চাটগাঁইয়া ভাষায় comedy roast english mixed",
        "চাটগাঁইয়া ভাষায় funny prank english বাংলা",
        # C
        "চাটগাঁইয়া ভাষায় reaction video english",
        "চাটগাঁইয়া ভাষায় gaming english বাংলা",
        # D
        "চাটগাঁইয়া ভাষায় university student career english",
        "চাটগাঁইয়া ভাষায় job interview english mixed",
        # E
        "চাটগাঁইয়া ভাষায় vlog english banglish 2023",
        "চাটগাঁইয়া ভাষায় day in my life english",
        "chatgaiya bhasha english comedy vlog mixed 2023",
        "চাটগাঁইয়া ভাষায় challenge english বাংলা",
    ],

    # ── 4. Sylhet ─────────────────────────────────────────────────────────────
    # Large UK diaspora channel base. "সিলেটি" well-known dialect name.
    "Sylhet": [
        # A
        "সিলেটি ভাষায় phone review english বাংলা",
        "সিলেটি ভাষায় tech unboxing english",
        # B
        "সিলেটি ভাষায় comedy roast english mixed",
        "সিলেটি ভাষায় funny prank english বাংলা",
        # C
        "সিলেটি ভাষায় reaction video english",
        "সিলেটি ভাষায় gaming english বাংলা",
        # D
        "সিলেটি ভাষায় university student career english",
        "সিলেটি ভাষায় job interview english mixed",
        # E
        "সিলেটি আঞ্চলিক ভাষায় vlog english banglish 2023",
        "সিলেটি ভাষায় day in my life english",
        "sylheti dialect english comedy vlog banglish 2023",
        "সিলেটি ভাষায় challenge english বাংলা mixed",
    ],

    # ── 5. Sylhet UK diaspora — highest CS rate in entire corpus ──────────────
    # British-born Sylheti: English is dominant, Bengali dialect is heritage.
    # These speakers CS constantly without thinking about it.
    "Sylhet_UK_CS": [
        # Daily life / vlog
        "sylheti british bangladeshi vlog english 2023",
        "british sylheti day in my life english bangla",
        # Comedy (very strong CS community in UK)
        "sylheti british comedy english bangla mixed",
        "british bangladeshi sylheti roast prank english",
        # Conversation / chat
        "sylheti english conversation british bangladeshi",
        "uk sylheti youth talking english bangla mixed",
        # Reaction / opinion
        "sylheti reaction video english british",
        "british born sylheti english bangla opinion vlog",
        # Diaspora specific
        "tower hamlets bangladeshi sylheti vlog english",
        "second generation sylheti english bangla heritage",
        "british bangladeshi sylheti challenge english",
        "sylheti uk english mixed casual chat 2023",
    ],

    # ── 6. Barishal ───────────────────────────────────────────────────────────
    "Barishal": [
        # A
        "বরিশালের আঞ্চলিক ভাষায় phone review english",
        "বরিশালের আঞ্চলিক ভাষায় tech unboxing english",
        # B
        "বরিশালের আঞ্চলিক ভাষায় comedy roast english",
        "বরিশালের ভাষায় funny prank english বাংলা",
        # C
        "বরিশালের আঞ্চলিক ভাষায় reaction video english",
        "বরিশালের ভাষায় gaming english বাংলা",
        # D
        "বরিশালের আঞ্চলিক ভাষায় university student english",
        "বরিশালের ভাষায় job interview career english",
        # E
        "বরিশালের আঞ্চলিক ভাষায় vlog english banglish",
        "বরিশালের ভাষায় day in my life english",
        "barishal dialect bangla english comedy vlog 2023",
        "বরিশালের আঞ্চলিক ভাষা challenge english mixed",
    ],

    # ── 7. Noakhali ───────────────────────────────────────────────────────────
    "Noakhali": [
        # A
        "নোয়াখালীর আঞ্চলিক ভাষায় phone review english",
        "নোয়াখালীর আঞ্চলিক ভাষায় tech unboxing english",
        # B
        "নোয়াখালীর আঞ্চলিক ভাষায় comedy roast english",
        "নোয়াখালীর ভাষায় funny prank english বাংলা",
        # C
        "নোয়াখালীর আঞ্চলিক ভাষায় reaction video english",
        "নোয়াখালীর ভাষায় gaming english বাংলা",
        # D
        "নোয়াখালীর আঞ্চলিক ভাষায় university student english",
        "নোয়াখালীর ভাষায় job interview career english",
        # E
        "নোয়াখালীর আঞ্চলিক ভাষায় vlog english banglish",
        "নোয়াখালীর ভাষায় day in my life english",
        "noakhali dialect bangla english comedy vlog 2023",
        "নোয়াখালীর আঞ্চলিক ভাষা challenge english mixed",
    ],

    # ── 8. Rangpur ────────────────────────────────────────────────────────────
    # Rangpur: uses "মুই" (I), "হামার" (my) — very distinct morphology.
    "Rangpur": [
        # A
        "রংপুরের আঞ্চলিক ভাষায় phone review english",
        "রংপুরের আঞ্চলিক ভাষায় tech unboxing english",
        # B
        "রংপুরের আঞ্চলিক ভাষায় comedy roast english",
        "রংপুরের ভাষায় funny prank english বাংলা",
        # C
        "রংপুরের আঞ্চলিক ভাষায় reaction video english",
        "রংপুরের ভাষায় gaming english বাংলা",
        # D
        "রংপুরের আঞ্চলিক ভাষায় university student english",
        "রংপুরের ভাষায় job interview career english",
        # E
        "রংপুরের আঞ্চলিক ভাষায় vlog english banglish",
        "রংপুরের ভাষায় day in my life english",
        "rangpur dialect bangla english comedy vlog 2023",
        "রংপুরের আঞ্চলিক ভাষা challenge english mixed",
    ],

    # ── 9. Mymensingh ─────────────────────────────────────────────────────────
    "Mymensingh": [
        # A
        "ময়মনসিংহের আঞ্চলিক ভাষায় phone review english",
        "ময়মনসিংহের আঞ্চলিক ভাষায় tech unboxing english",
        # B
        "ময়মনসিংহের আঞ্চলিক ভাষায় comedy roast english",
        "ময়মনসিংহের ভাষায় funny prank english বাংলা",
        # C
        "ময়মনসিংহের আঞ্চলিক ভাষায় reaction video english",
        "ময়মনসিংহের ভাষায় gaming english বাংলা",
        # D
        "ময়মনসিংহের আঞ্চলিক ভাষায় university student english",
        "ময়মনসিংহের ভাষায় job interview career english",
        # E
        "ময়মনসিংহের আঞ্চলিক ভাষায় vlog english banglish",
        "ময়মনসিংহের ভাষায় day in my life english",
        "mymensingh dialect bangla english comedy vlog 2023",
        "ময়মনসিংহের আঞ্চলিক ভাষা challenge english mixed",
    ],

    # ── 10. Khulna ────────────────────────────────────────────────────────────
    "Khulna": [
        # A
        "খুলনার আঞ্চলিক ভাষায় phone review english",
        "খুলনার আঞ্চলিক ভাষায় tech unboxing english",
        # B
        "খুলনার আঞ্চলিক ভাষায় comedy roast english",
        "খুলনার ভাষায় funny prank english বাংলা",
        # C
        "খুলনার আঞ্চলিক ভাষায় reaction video english",
        "খুলনার ভাষায় gaming english বাংলা",
        # D
        "খুলনার আঞ্চলিক ভাষায় university student english",
        "খুলনার ভাষায় job interview career english",
        # E
        "খুলনার আঞ্চলিক ভাষায় vlog english banglish",
        "খুলনার ভাষায় day in my life english",
        "khulna dialect bangla english comedy vlog 2023",
        "খুলনার আঞ্চলিক ভাষা challenge english mixed",
    ],

    # ── 11. Tangail ───────────────────────────────────────────────────────────
    "Tangail": [
        # A
        "টাঙ্গাইলের আঞ্চলিক ভাষায় phone review english",
        "টাঙ্গাইলের আঞ্চলিক ভাষায় tech unboxing english",
        # B
        "টাঙ্গাইলের আঞ্চলিক ভাষায় comedy roast english",
        "টাঙ্গাইলের ভাষায় funny prank english বাংলা",
        # C
        "টাঙ্গাইলের আঞ্চলিক ভাষায় reaction video english",
        "টাঙ্গাইলের ভাষায় gaming english বাংলা",
        # D
        "টাঙ্গাইলের আঞ্চলিক ভাষায় university student english",
        "টাঙ্গাইলের ভাষায় job interview career english",
        # E
        "টাঙ্গাইলের আঞ্চলিক ভাষায় vlog english banglish",
        "টাঙ্গাইলের ভাষায় day in my life english",
        "tangail dialect bangla english comedy vlog 2023",
        "টাঙ্গাইলের আঞ্চলিক ভাষা challenge english mixed",
    ],

    # ── 12. Kishoreganj ───────────────────────────────────────────────────────
    "Kishoreganj": [
        # A
        "কিশোরগঞ্জের আঞ্চলিক ভাষায় phone review english",
        "কিশোরগঞ্জের আঞ্চলিক ভাষায় tech unboxing english",
        # B
        "কিশোরগঞ্জের আঞ্চলিক ভাষায় comedy roast english",
        "কিশোরগঞ্জের ভাষায় funny prank english বাংলা",
        # C
        "কিশোরগঞ্জের আঞ্চলিক ভাষায় reaction video english",
        "কিশোরগঞ্জের ভাষায় gaming english বাংলা",
        # D
        "কিশোরগঞ্জের আঞ্চলিক ভাষায় university student english",
        "কিশোরগঞ্জের ভাষায় job interview career english",
        # E
        "কিশোরগঞ্জের আঞ্চলিক ভাষায় vlog english banglish",
        "কিশোরগঞ্জের ভাষায় day in my life english",
        "kishoreganj dialect bangla english comedy vlog 2023",
        "কিশোরগঞ্জের আঞ্চলিক ভাষা challenge english mixed",
    ],

    # ── 13. Habiganj ──────────────────────────────────────────────────────────
    "Habiganj": [
        # A
        "হবিগঞ্জের আঞ্চলিক ভাষায় phone review english",
        "হবিগঞ্জের আঞ্চলিক ভাষায় tech unboxing english",
        # B
        "হবিগঞ্জের আঞ্চলিক ভাষায় comedy roast english",
        "হবিগঞ্জের ভাষায় funny prank english বাংলা",
        # C
        "হবিগঞ্জের আঞ্চলিক ভাষায় reaction video english",
        "হবিগঞ্জের ভাষায় gaming english বাংলা",
        # D
        "হবিগঞ্জের আঞ্চলিক ভাষায় university student english",
        "হবিগঞ্জের ভাষায় job interview career english",
        # E
        "হবিগঞ্জের আঞ্চলিক ভাষায় vlog english banglish",
        "হবিগঞ্জের ভাষায় day in my life english",
        "habiganj dialect bangla english comedy vlog 2023",
        "হবিগঞ্জের আঞ্চলিক ভাষা challenge english mixed",
    ],

    # ── 14. Narail ────────────────────────────────────────────────────────────
    "Narail": [
        # A
        "নড়াইলের আঞ্চলিক ভাষায় phone review english",
        "নড়াইলের আঞ্চলিক ভাষায় tech unboxing english",
        # B
        "নড়াইলের আঞ্চলিক ভাষায় comedy roast english",
        "নড়াইলের ভাষায় funny prank english বাংলা",
        # C
        "নড়াইলের আঞ্চলিক ভাষায় reaction video english",
        "নড়াইলের ভাষায় gaming english বাংলা",
        # D
        "নড়াইলের আঞ্চলিক ভাষায় university student english",
        "নড়াইলের ভাষায় job interview career english",
        # E
        "নড়াইলের আঞ্চলিক ভাষায় vlog english banglish",
        "নড়াইলের ভাষায় day in my life english",
        "narail dialect bangla english comedy vlog 2023",
        "নড়াইলের আঞ্চলিক ভাষা challenge english mixed",
    ],

    # ── 15. Narsingdi ─────────────────────────────────────────────────────────
    "Narsingdi": [
        # A
        "নরসিংদীর আঞ্চলিক ভাষায় phone review english",
        "নরসিংদীর আঞ্চলিক ভাষায় tech unboxing english",
        # B
        "নরসিংদীর আঞ্চলিক ভাষায় comedy roast english",
        "নরসিংদীর ভাষায় funny prank english বাংলা",
        # C
        "নরসিংদীর আঞ্চলিক ভাষায় reaction video english",
        "নরসিংদীর ভাষায় gaming english বাংলা",
        # D
        "নরসিংদীর আঞ্চলিক ভাষায় university student english",
        "নরসিংদীর ভাষায় job interview career english",
        # E
        "নরসিংদীর আঞ্চলিক ভাষায় vlog english banglish",
        "নরসিংদীর ভাষায় day in my life english",
        "narsingdi dialect bangla english comedy vlog 2023",
        "নরসিংদীর আঞ্চলিক ভাষা challenge english mixed",
    ],

    # ── 16. Sandwip ───────────────────────────────────────────────────────────
    # Smaller community — use broader patterns + explicit dialect label.
    "Sandwip": [
        "সন্দ্বীপের আঞ্চলিক ভাষায় vlog english",
        "সন্দ্বীপের আঞ্চলিক ভাষায় comedy english",
        "সন্দ্বীপের আঞ্চলিক ভাষায় phone review english",
        "সন্দ্বীপের ভাষায় reaction english বাংলা",
        "সন্দ্বীপের আঞ্চলিক ভাষায় interview english",
        "sandwip dialect bangla english vlog comedy",
        "সন্দ্বীপের আঞ্চলিক ভাষা english mixed 2023",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# FOLDER METADATA
# Maps each download folder to dialect + domain labels.
# Used by 02_segment_audio.py to tag clips correctly.
# ══════════════════════════════════════════════════════════════════════════════
FOLDER_METADATA = {
    **YOUTUBE_FOLDER_METADATA,
    # TikTok buckets
    "Sylhet_TikTok":     {"dialect": "Sylhet",      "domain": "TikTok"},
    "Chittagong_TikTok": {"dialect": "Chittagong",  "domain": "TikTok"},
    "Rangpur_TikTok":    {"dialect": "Rangpur",     "domain": "TikTok"},
    "Bangladesh_TikTok": {"dialect": "unknown",     "domain": "TikTok"},
    # Facebook buckets (dialect set per key in FACEBOOK_URLS)
    "Sylhet_FB":     {"dialect": "Sylhet",     "domain": "Facebook"},
    "Barishal_FB":   {"dialect": "Barishal",   "domain": "Facebook"},
    "Chittagong_FB": {"dialect": "Chittagong", "domain": "Facebook"},
    "Noakhali_FB":   {"dialect": "Noakhali",   "domain": "Facebook"},
    "Rangpur_FB":    {"dialect": "Rangpur",    "domain": "Facebook"},
    "Mymensingh_FB": {"dialect": "Mymensingh", "domain": "Facebook"},
    "Khulna_FB":     {"dialect": "Khulna",     "domain": "Facebook"},
    "Tangail_FB":    {"dialect": "Tangail",    "domain": "Facebook"},
}


# ══════════════════════════════════════════════════════════════════════════════
# TIKTOK — hashtag search
# ══════════════════════════════════════════════════════════════════════════════
TIKTOK_BUCKETS = {
    "Sylhet_TikTok": {
        "hashtags": ["sylheti", "britishbangladeshi", "sylhetilife"],
    },
    "Chittagong_TikTok": {
        "hashtags": ["chatgaiya", "chottogramibhasha", "chittagongdialect"],
    },
    "Rangpur_TikTok": {
        "hashtags": ["rangpurdialect", "rangpurbangladesh"],
    },
    "Bangladesh_TikTok": {
        "hashtags": ["bangladeshivlog", "banglish", "dhakalife"],
    },
}




# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def _yt_dlp_cmd(output_dir: str) -> list:
    """Base yt-dlp command with explicit ffmpeg location."""
    from pipeline_utils import find_command
    ffmpeg_path = find_command("ffmpeg")
    ffmpeg_dir  = os.path.dirname(ffmpeg_path) if ffmpeg_path else None

    cmd = [
        find_command("yt-dlp") or "yt-dlp",
        "--extract-audio",
        "--audio-format",  AUDIO_FORMAT,
        "--audio-quality", "0",
        "--output",        os.path.join(output_dir, "%(id)s.%(ext)s"),
        "--no-playlist",
        "--quiet",
        "--ignore-errors",
    ]
    if ffmpeg_dir:
        cmd += ["--ffmpeg-location", ffmpeg_dir]
    return cmd


def preflight_download_dependencies():
    resolved = require_commands(
        [
            ("yt-dlp",  ("yt-dlp",),  "pip install yt-dlp"),
            ("ffmpeg",  ("ffmpeg",),  "conda install ffmpeg -c conda-forge"),
            ("ffprobe", ("ffprobe",), "conda install ffmpeg -c conda-forge"),
        ],
        context="audio download",
    )
    require_command_runs("ffmpeg",  [resolved["ffmpeg"],  "-version"], context="audio download")
    require_command_runs("ffprobe", [resolved["ffprobe"], "-version"], context="audio download")
    warn_if_missing_js_runtime()


def download_youtube(buckets: dict, max_per_query: int = 10,
                     base_dir: str = OUTPUT_DIR):
    """
    Download audio from YouTube for all dialect buckets.

    Filters applied to every search:
      --dateafter 20210101       → post-2021 only; recent content has more CS
      --match-filter view_count>500 → skip zero-audience uploads (usually noise)
      --max-downloads per query  → capped by max_per_query after dedup
      --download-archive         → global dedup across all dialects
    """
    print("\n" + "=" * 60)
    print("YOUTUBE - CS-focused dialect search")
    print("=" * 60)

    archive_path = os.path.join(base_dir, "downloaded_ids.txt")
    print(f"  Dedup archive : {archive_path}")
    print(f"  Max per query : {max_per_query}")
    print(f"  Date filter   : 2021-01-01 onwards")
    print(f"  View filter   : > 500 views")

    for bucket_name, spec in buckets.items():
        query_list = spec["queries"] if isinstance(spec, dict) else spec

        out_dir = os.path.join(base_dir, "youtube", bucket_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n  [{bucket_name}] {len(query_list)} queries x {max_per_query}")

        for query in query_list:
            search_url = f"ytsearch{max_per_query}:{query}"
            subprocess.run(
                _yt_dlp_cmd(out_dir)
                + ["--download-archive", archive_path]
                + ["--dateafter",    "20210101"]
                + ["--match-filter", "view_count > 500"]
                + [search_url]
            )

        downloaded = sum(1 for f in os.scandir(out_dir)
                         if f.name.endswith((".wav", ".webm", ".m4a", ".opus")))
        print(f"  -> {downloaded} files in {out_dir}")


def download_tiktok(tiktok_buckets: dict, max_per_tag: int = 20,
                    base_dir: str = OUTPUT_DIR):
    print("\n" + "=" * 60)
    print("TIKTOK - hashtag buckets")
    print("=" * 60)

    for bucket_name, spec in tiktok_buckets.items():
        out_dir = os.path.join(base_dir, "tiktok", bucket_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n  [{bucket_name}]")
        for tag in spec["hashtags"]:
            print(f"    #{tag}")
            subprocess.run(_yt_dlp_cmd(out_dir) + [
                "--playlist-end", str(max_per_tag),
                f"https://www.tiktok.com/tag/{tag}",
            ])


def download_facebook(url_dict: dict, base_dir: str = OUTPUT_DIR):
    print("\n" + "=" * 60)
    print("FACEBOOK - manual URLs")
    print("=" * 60)

    has_urls = any(len(urls) > 0 for urls in url_dict.values())
    if not has_urls:
        print("  No Facebook URLs provided. Skipping.")
        print("  To add: browse Facebook, find dialect videos, paste URLs above.")
        return

    for dialect, urls in url_dict.items():
        if not urls:
            continue
        out_dir = os.path.join(base_dir, "facebook", f"{dialect}_FB")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n  [{dialect}] {len(urls)} URLs")
        for url in urls:
            cmd = _yt_dlp_cmd(out_dir) + ["--cookies-from-browser", "chrome", url]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                subprocess.run(_yt_dlp_cmd(out_dir) + [url])


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    preflight_download_dependencies()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    query_path = os.path.join(OUTPUT_DIR, "query_catalog.json")
    write_query_catalog(query_path, YOUTUBE_BUCKETS)

    # YouTube — CS-first queries across all 16 dialects.
    # 8 queries x 10 videos = up to 80 videos per dialect before dedup.
    # Increase max_per_query to 15 if you want more data.
    download_youtube(YOUTUBE_BUCKETS, max_per_query=8)

    # TikTok — unreliable (hashtag pages are bot-blocked by TikTok).
    # Uncomment only if you want to try; expect 0 results most of the time.
    # download_tiktok(TIKTOK_BUCKETS, max_per_tag=20)

    # Facebook — add URLs manually in FACEBOOK_URLS above, then uncomment.
    # Requires being logged into Facebook in Chrome.
    # download_facebook(FACEBOOK_URLS)

    # Save folder metadata for 02_segment_audio.py
    meta_path = os.path.join(OUTPUT_DIR, "folder_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(FOLDER_METADATA, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("All downloads complete.")
    print(f"Saved to : {OUTPUT_DIR}/")
    print(f"Queries  : {query_path}")
    print(f"Metadata : {meta_path}")
    print("Next     : python 02_segment_audio.py")
