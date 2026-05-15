import json
from pathlib import Path

def convert_json_to_jsonl():
    # Define the input and output paths
    input_path = Path(r"c:\Niloy Islam PhD\ASR work bengali\dataset_pipeline\transcripts\transcripts_reviewed.json")
    output_path = Path(r"c:\Niloy Islam PhD\ASR work bengali\dataset_pipeline\transcripts\transcripts_reviewed_niloy_for_check.jsonl")

    if not input_path.exists():
        print(f"[ERROR] Could not find {input_path}")
        return

    print(f"Loading data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        clips = json.load(f)

    print(f"Writing to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for clip in clips:
            f.write(json.dumps(clip, ensure_ascii=False) + "\n")

    print(f"Successfully converted {len(clips)} clips to JSONL!")

if __name__ == "__main__":
    convert_json_to_jsonl()