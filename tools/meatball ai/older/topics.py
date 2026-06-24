from pathlib import Path

FRAGMENTS_DIR = Path("assets/data/fragments")
OUTPUT_FILE = Path("assets/data/topics.txt")

topics = []

for path in sorted(FRAGMENTS_DIR.glob("*.jsonl")):
    topics.append(path.stem)

with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    for topic in topics:
        f.write(topic + "\n")

print(f"Saved {len(topics)} topics to {OUTPUT_FILE}")
