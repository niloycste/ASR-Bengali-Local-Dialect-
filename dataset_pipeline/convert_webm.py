"""
Convert downloaded .webm / .m4a / .mp4 files in raw_audio/ to 16 kHz mono .wav.

This is optional now because Step 2 can read multiple source formats directly,
but it is still useful if you want a normalized raw-audio tree.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from pipeline_utils import find_command

SCRIPT_DIR = Path(__file__).resolve().parent
SEARCH_DIR = SCRIPT_DIR / "raw_audio"
SAMPLE_RATE = 16000


def convert_file(src_path: Path, ffmpeg_exe: str) -> bool:
    """Convert one file to wav and remove the source on success."""
    wav_path = src_path.with_suffix(".wav")
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(src_path),
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0:
        src_path.unlink()
        return True

    print(f"  [FAILED] {src_path.name}")
    print(result.stderr.decode(errors="ignore")[:200])
    return False


def convert_all(search_dir: str | Path = SEARCH_DIR):
    """Convert every supported compressed media file under raw_audio/."""
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg_exe = find_command("ffmpeg")
        if not ffmpeg_exe:
            print("[ERROR] ffmpeg not found and imageio_ffmpeg is not installed.")
            print("        Install ffmpeg or run: pip install imageio-ffmpeg")
            return

    search_path = Path(search_dir)
    extensions = (".webm", ".m4a", ".mp4")
    files_by_ext = {
        ext: list(search_path.rglob(f"*{ext}"))
        for ext in extensions
    }
    all_files = [
        src_file
        for ext in extensions
        for src_file in files_by_ext[ext]
    ]
    counts = ", ".join(f"{len(files_by_ext[ext])} {ext.lstrip('.')}" for ext in extensions)
    print(f"Found {len(all_files)} files to convert ({counts}).\n")

    ok = fail = 0
    for src_file in all_files:
        print(f"  Converting: {src_file.name} ...", end=" ")
        if convert_file(src_file, ffmpeg_exe):
            print("OK")
            ok += 1
        else:
            fail += 1

    print(f"\nDone. Converted: {ok}  Failed: {fail}")
    if ok > 0:
        print("Next: python 02_segment_audio.py")


if __name__ == "__main__":
    convert_all()
