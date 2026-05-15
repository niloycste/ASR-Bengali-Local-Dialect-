"""
Shared helpers for pipeline preflight checks and small utilities.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
from typing import Iterable


# Known conda environment locations to search when PATH is not activated.
# These are checked automatically if the tool is not found on PATH.
_CONDA_FALLBACK_DIRS = [
    r"C:\Users\USER\.conda\envs\ASR\Library\bin",
    r"C:\Users\USER\.conda\envs\ASR\Scripts",
    r"C:\ProgramData\anaconda3\Library\bin",
    r"C:\ProgramData\anaconda3\Scripts",
]


def find_command(*names: str) -> str | None:
    """
    Return the first executable found via (in priority order):
      1. Standard PATH lookup
      2. Known conda environment directories
      3. imageio_ffmpeg bundled binary (for ffmpeg/ffprobe only)
    """
    for name in names:
        # 1. Standard PATH
        path = shutil.which(name)
        if path:
            return path
        # 2. Known conda dirs
        for d in _CONDA_FALLBACK_DIRS:
            for ext in ("", ".exe", ".EXE"):
                candidate = os.path.join(d, name + ext)
                if os.path.isfile(candidate):
                    return candidate

    # 3. imageio_ffmpeg bundled binary — works even with broken conda ffmpeg
    if any(n in ("ffmpeg", "ffprobe") for n in names):
        try:
            import imageio_ffmpeg
            bundled = imageio_ffmpeg.get_ffmpeg_exe()
            if bundled and os.path.isfile(bundled):
                return bundled
        except ImportError:
            pass

    return None


def print_dependency_summary() -> None:
    """Print the external tools the pipeline can see."""
    print("Detected tools:")
    print(f"  yt-dlp : {find_command('yt-dlp') or 'NOT FOUND'}")
    print(f"  ffmpeg : {find_command('ffmpeg') or 'NOT FOUND'}")
    print(f"  ffprobe: {find_command('ffprobe') or 'NOT FOUND'}")
    print(
        "  JS runtime: "
        f"{find_command('node') or find_command('deno') or find_command('bun') or 'NOT FOUND'}"
    )


def require_commands(
    requirements: Iterable[tuple[str, tuple[str, ...], str]],
    *,
    context: str,
) -> dict[str, str]:
    """
    Ensure required external tools exist on PATH.

    requirements: iterable of (label, candidate_names, install_hint)
    Returns resolved command paths keyed by label.
    """
    resolved: dict[str, str] = {}
    missing: list[tuple[str, str]] = []

    for label, names, install_hint in requirements:
        path = find_command(*names)
        if path:
            resolved[label] = path
        else:
            missing.append((label, install_hint))

    if missing:
        print(f"[ERROR] Missing external tools for {context}.")
        for label, install_hint in missing:
            print(f"  - {label}: {install_hint}")
        print_dependency_summary()
        raise SystemExit(1)

    return resolved


def require_command_runs(label: str, command: list[str], *, context: str) -> None:
    """
    Ensure a discovered command is actually runnable.

    Only fails on OSError (file not found / permission denied).
    Non-zero exit codes are accepted — conda's ffmpeg returns exit code 1
    for '-version' on Windows even though it works correctly.
    """
    try:
        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        print(f"[ERROR] {label} was found on PATH but could not be launched for {context}.")
        print(f"        Tried: {' '.join(command)}")
        print(f"        This usually means a broken conda install or missing DLL.")
        print(f"        Fix: conda install ffmpeg  OR  winget install ffmpeg")
        raise SystemExit(1)


def warn_if_missing_js_runtime() -> None:
    """Warn when yt-dlp has no JavaScript runtime available for YouTube."""
    if not find_command("node", "deno", "bun"):
        print("[WARNING] No JavaScript runtime found on PATH (node, deno, or bun).")
        print("          yt-dlp may still work, but YouTube extraction can be less reliable.")
        print("          Install Node.js or Deno if downloads look incomplete.")


def shuffled(items: list, seed: int = 42) -> list:
    """Return a deterministically shuffled copy."""
    clone = list(items)
    random.Random(seed).shuffle(clone)
    return clone


def exit_with_import_error(package_name: str, install_name: str | None = None) -> None:
    """Print a consistent import failure message and exit."""
    install_target = install_name or package_name
    print(f"[ERROR] Missing Python package: {package_name}")
    print(f"        Install it with: pip install {install_target}")
    sys.exit(1)
