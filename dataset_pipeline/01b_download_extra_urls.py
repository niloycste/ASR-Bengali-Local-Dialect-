"""
Step 1b: Download audio from any website using direct URLs.
Supports Facebook, Instagram, Twitter/X, Dailymotion, Vimeo, Reddit, and
1000+ other sites that yt-dlp supports.

Usage:
  1. Find a video on Facebook / any site
  2. Copy the URL and paste it below under the correct dialect
  3. Run:  python 01b_download_extra_urls.py

For Facebook: make sure you are logged in to Facebook in Chrome first.
The script will use your Chrome cookies automatically.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from pipeline_utils import find_command

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = str(SCRIPT_DIR / "raw_audio")

# ══════════════════════════════════════════════════════════════════════════════
# PASTE YOUR URLs HERE
# Add as many URLs as you like under each dialect.
# Any website works — Facebook, Instagram, Twitter/X, Dailymotion, etc.
# Example:
#   "Chittagong": [
#       "https://www.facebook.com/watch/?v=123456789",
#       "https://www.instagram.com/reel/AbCdEfGh/",
#   ],
# ══════════════════════════════════════════════════════════════════════════════
EXTRA_URLS: dict[str, list[str]] = {
    "Old_Dhaka": [
        "https://www.facebook.com/share/v/186TeHuH3K/",
        "https://www.facebook.com/share/v/1HTmDaSvtj/",
        "https://www.facebook.com/share/v/1DhQm45WQz/",
        "https://www.facebook.com/share/v/1E9GvHWYmy/",
        "https://www.facebook.com/share/v/1DW1qy58hq/",
        "https://www.facebook.com/share/v/1hFs4S6CpB/",
        "https://www.facebook.com/share/v/17qRHUEXL1/",
        "https://www.facebook.com/share/v/1bopTV31Jv/",
    ],
    "Comilla": [
        "https://www.facebook.com/share/v/1GiWzHNfS7/",
        "https://www.facebook.com/share/v/1CYmYtnihZ/",
        "https://www.facebook.com/share/v/1ArkwkBe7k/",
        "https://www.facebook.com/share/v/1H5TVVGLLz/",
        "https://www.facebook.com/share/v/1AovSRRurG/",
        "https://www.facebook.com/share/v/1CY8y26Vrn/",
        "https://www.facebook.com/share/v/1AAUyUCRPB/",
        "https://www.facebook.com/share/r/1RYccPrzf3/",
        "https://www.facebook.com/share/v/17xaUBYT3F/",
        "https://www.facebook.com/share/v/1CLFexoUau/",
        "https://www.facebook.com/share/v/1AttmWW9Yq/",
        "https://www.facebook.com/share/v/185r65LSCq/",
    ],
    "Chittagong": [
         "https://www.facebook.com/share/v/17HZh4XWGN/",
         "https://www.facebook.com/share/v/18hAq4hy95/",
         "https://www.facebook.com/share/v/1DhSq75Foe/",
         "https://www.facebook.com/share/r/1CNFVwoeLx/",
         "https://www.facebook.com/reel/834203016111512/",
         "https://www.facebook.com/share/v/1DKCdU6WNy/",
         "https://www.facebook.com/share/v/1KfTiGDprr/",
         "https://www.facebook.com/share/v/17oCUz9ohd/",
         "https://www.facebook.com/share/v/1GnngufUFH/",
         "https://www.facebook.com/share/r/1LBbGJ3i4L/",
         "https://www.facebook.com/share/v/1CZJdLsm79/",
         "https://www.facebook.com/share/v/1Ft6McbmJY/",
         "https://www.facebook.com/share/v/1EeQtn1NmS/",
         "https://www.facebook.com/share/v/1CDzfvoYSy/",
         "https://www.facebook.com/share/v/175MKVrirw/",
         "https://www.facebook.com/share/v/18Lk4fhMxF/",
         "https://www.facebook.com/share/v/1DoqrMs1qX/",
    ],
    "Sylhet": [
        "https://www.facebook.com/share/v/18N2rerbWJ/",
        "https://www.facebook.com/share/v/1Gf2SkBUvw/",
        "https://www.facebook.com/share/v/1G63w86dN9/",
        "https://www.facebook.com/share/v/18NvPwJNxc/",
        "https://www.facebook.com/share/v/1CL3cY6a9N/",
        "https://www.facebook.com/share/v/1DXgXbwXMV/",
        "https://www.facebook.com/share/v/1DGzs9cLWC/",
        "https://www.facebook.com/share/v/17o3gVTtwE/",
        "https://www.facebook.com/share/v/18KXQt7iiP/",
        "https://www.facebook.com/share/v/1KHZduamXz/",
        "https://www.facebook.com/share/v/1GVzi5j9xj/",
        "https://www.facebook.com/share/v/1HGvpBqje9/",
        "https://www.facebook.com/share/v/1aappY6ibX/",
        "https://www.facebook.com/share/v/1DsjQzPwCL/",
        "https://www.facebook.com/share/v/1AYxcFrj3w/",
        "https://www.facebook.com/share/v/1CdpxSp95x/",
        "https://www.facebook.com/share/v/1NSEBZuWz3/",
    ],
    "Barishal": [
       "https://www.facebook.com/share/v/188oSXmYu2/",
       "https://www.facebook.com/share/v/14YJQ8Sbed5/",
       "https://www.facebook.com/share/v/1AyfGRPNkY/",
       "https://www.facebook.com/share/r/1ASe3B3vYT/",
       "https://www.facebook.com/share/v/1Cc3mJUSyz/",
       "https://www.facebook.com/share/v/1C7Aw5nwNv/",
       "https://www.facebook.com/share/v/1GCkCrYD2G/",
       "https://www.facebook.com/share/v/1CVnGTjLWM/",
       "https://www.facebook.com/share/v/1GVSLxEKMS/",
       "https://www.facebook.com/share/v/18YD3xopYB/",
       "https://www.facebook.com/share/v/1G3WuC2nGi/",
       "https://www.facebook.com/share/v/1TB9vdic8V/",
       "https://www.facebook.com/share/v/1GVP8SQ5vt/",
       "https://www.facebook.com/share/v/1EBKEkZHP2/",
       "https://www.facebook.com/share/v/1HfNaghXdA/",
       "https://www.facebook.com/share/v/1NSEBZuWz3/",


    ],
    "Noakhali": [
        "https://www.facebook.com/share/v/1HfNaghXdA/",
        "https://www.facebook.com/share/v/1NSEBZuWz3/",
        "https://www.facebook.com/share/v/17PiEmoBCq/",
        "https://www.facebook.com/share/v/17FA2CC13t/",
        "https://www.facebook.com/share/v/1CZ6pZ2mnK/",
        "https://www.facebook.com/share/v/17i7pGBFF4/",
        "https://www.facebook.com/share/v/1GsmgnLf44/",
        "https://www.facebook.com/share/v/1J63vvzAEU/",
        "https://www.facebook.com/share/v/18eqnE7dYC/",
        "https://www.facebook.com/share/v/1KuyZP8XsX/",
        "https://www.facebook.com/share/v/17NF3y9xe6/",
        "https://www.facebook.com/share/v/17UDLzMPoj/",
    ],
    # "Rangpur": [
    #     # paste Rangpur dialect video URLs here
    # ],
    # "Mymensingh": [
    #     # paste Mymensingh dialect video URLs here
    # ],
    # "Khulna": [
    #     # paste Khulna dialect video URLs here
    # ],
    # "Kushtia": [
    #     # paste Kushtia dialect video URLs here
    # ],
    # "Tangail": [
    #     # paste Tangail dialect video URLs here
    # ],
    # "Kishoreganj": [
    #     # paste Kishoreganj dialect video URLs here
    # ],
    # "Habiganj": [
    #     # paste Habiganj dialect video URLs here
    # ],
    # "Narail": [
    #     # paste Narail dialect video URLs here
    # ],
    # "Narsingdi": [
    #     # paste Narsingdi dialect video URLs here
    # ],
    # "Sandwip": [
    #     # paste Sandwip dialect video URLs here
    # ],
}


# ══════════════════════════════════════════════════════════════════════════════
# FOLDER METADATA — dialect label for each key above
# Add a new entry here if you add a new dialect key above.
# ══════════════════════════════════════════════════════════════════════════════
FOLDER_METADATA = {
    "Old_Dhaka":   {"dialect": "Old_Dhaka",   "domain": "General"},
    "Comilla":     {"dialect": "Comilla",     "domain": "General"},
    "Chittagong":  {"dialect": "Chittagong",  "domain": "General"},
    "Sylhet":      {"dialect": "Sylhet",      "domain": "General"},
    "Barishal":    {"dialect": "Barishal",    "domain": "General"},
    "Noakhali":    {"dialect": "Noakhali",    "domain": "General"},
    "Rangpur":     {"dialect": "Rangpur",     "domain": "General"},
    "Mymensingh":  {"dialect": "Mymensingh",  "domain": "General"},
    "Khulna":      {"dialect": "Khulna",      "domain": "General"},
    "Kushtia":     {"dialect": "Kushtia",     "domain": "General"},
    "Tangail":     {"dialect": "Tangail",     "domain": "General"},
    "Kishoreganj": {"dialect": "Kishoreganj", "domain": "General"},
    "Habiganj":    {"dialect": "Habiganj",    "domain": "General"},
    "Narail":      {"dialect": "Narail",      "domain": "General"},
    "Narsingdi":   {"dialect": "Narsingdi",   "domain": "General"},
    "Sandwip":     {"dialect": "Sandwip",     "domain": "General"},
}


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD
# ══════════════════════════════════════════════════════════════════════════════

def _base_cmd(output_dir: str) -> list:
    ffmpeg_path = find_command("ffmpeg")
    ffmpeg_dir  = os.path.dirname(ffmpeg_path) if ffmpeg_path else None
    cmd = [
        find_command("yt-dlp") or "yt-dlp",
        "--extract-audio",
        "--audio-format",  "wav",
        "--audio-quality", "0",
        "--output",        os.path.join(output_dir, "%(id)s.%(ext)s"),
        "--no-playlist",
        "--ignore-errors",
    ]
    if ffmpeg_dir:
        cmd += ["--ffmpeg-location", ffmpeg_dir]
    return cmd


def _is_facebook(url: str) -> bool:
    return "facebook.com" in url or "fb.watch" in url or "fb.com" in url


def download_url(url: str, output_dir: str, archive_path: str):
    """Download a single URL. Uses Chrome cookies for Facebook."""
    cmd = _base_cmd(output_dir) + ["--download-archive", archive_path]

    if _is_facebook(url):
        # Try with Chrome cookies first (you must be logged in to Facebook in Chrome)
        cmd_fb = cmd + ["--cookies-from-browser", "chrome", url]
        result = subprocess.run(cmd_fb)
        if result.returncode != 0:
            print(f"    [retry without cookies] {url}")
            subprocess.run(cmd + [url])
    else:
        subprocess.run(cmd + [url])


def main():
    has_urls = any(len(urls) > 0 for urls in EXTRA_URLS.values())
    if not has_urls:
        print("No URLs found. Open this script and paste video URLs into EXTRA_URLS.")
        return

    # Shared dedup archive with 01_download_audio.py — avoids re-downloading
    # anything already pulled by the YouTube step.
    archive_path = os.path.join(OUTPUT_DIR, "downloaded_ids.txt")

    print("=" * 60)
    print("Downloading extra URLs (Facebook / any site)")
    print(f"Dedup archive: {archive_path}")
    print("=" * 60)

    total_downloaded = 0

    for dialect, urls in EXTRA_URLS.items():
        if not urls:
            continue

        out_dir = os.path.join(OUTPUT_DIR, "extra", dialect)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[{dialect}] {len(urls)} URLs -> {out_dir}")

        for url in urls:
            print(f"  {url}")
            download_url(url, out_dir, archive_path)

        count = sum(1 for f in os.scandir(out_dir)
                    if f.name.endswith((".wav", ".webm", ".m4a", ".opus")))
        print(f"  -> {count} files")
        total_downloaded += count

    # Merge this script's folder metadata into the existing folder_metadata.json
    # so 02_segment_audio.py picks up the correct dialect labels.
    meta_path = os.path.join(OUTPUT_DIR, "folder_metadata.json")
    existing = {}
    if os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8") as f:
            existing = json.load(f)

    # Add entries for the "extra" sub-folders used by this script
    for dialect, meta in FOLDER_METADATA.items():
        key = f"extra_{dialect}"
        existing[key] = meta

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"Done. {total_downloaded} files downloaded.")
    print(f"Metadata updated: {meta_path}")
    print("Next: python 02_segment_audio.py")


if __name__ == "__main__":
    main()
