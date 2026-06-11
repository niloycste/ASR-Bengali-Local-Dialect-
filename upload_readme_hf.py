from huggingface_hub import HfApi, login
import os
import sys
from pathlib import Path

# --- Project-level imports ---
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
try:
    from fine_tuning.config import TRANSCRIPTS_MERGED_DIR
except ImportError:
    print("[ERROR] Could not import config. Ensure script is run from project root.")
    sys.exit(1)

# 1. Log in with your token
# The recommended way to log in is via the CLI: `huggingface-cli login`
# This script will then use the cached token automatically.
try:
    # login() will automatically use the cached token from the CLI or an env var.
    login()
    print("Hugging Face login successful.")
except Exception as e:
    print(f"Hugging Face login failed: {e}")
    print("\nPlease try logging in via the command line: `huggingface-cli login`")
    sys.exit(1)

api = HfApi()
repo_id = "niloycste68/Bangali_local_dialect_ASR_HF_Dataset"

# 2. Read the DATASET-ONLY card for Hugging Face.
# NOTE: the Hugging Face dataset page should show dataset-only info (description,
# fields, how to use), NOT the full project README (which covers the whole
# training/evaluation pipeline). So we upload hf_dataset_README.md, not README.md.
readme_path = SCRIPT_DIR / "hf_dataset_README.md"
if not readme_path.exists():
    print(f"[ERROR] hf_dataset_README.md not found at: {readme_path}")
    print("        This is the dataset-only card uploaded to the Hugging Face page.")
    sys.exit(1)

# 3. Upload README directly to Hugging Face
print(f"Uploading README.md from '{readme_path}' to '{repo_id}'...")
api.upload_file(
    path_or_fileobj=str(readme_path),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="dataset"
)
print("README successfully uploaded! Check your repository page on Hugging Face.")

# 4. Upload the raw metadata files (JSON and TSV)
if TRANSCRIPTS_MERGED_DIR.exists():
    print(f"\nUploading metadata files from '{TRANSCRIPTS_MERGED_DIR}' to Hugging Face...")
    api.upload_folder(
        folder_path=str(TRANSCRIPTS_MERGED_DIR),
        path_in_repo="metadata",  # Saves them in a folder called 'metadata' on HF
        repo_id=repo_id,
        repo_type="dataset"
    )
    print("Metadata files successfully uploaded! They are now visible on the website.")
else:
    print(f"\n[WARN] Could not find metadata folder: {TRANSCRIPTS_MERGED_DIR}")
    print(f"        Skipping metadata upload.")