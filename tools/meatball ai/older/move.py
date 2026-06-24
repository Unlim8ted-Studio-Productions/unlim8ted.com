#!/usr/bin/env python3

import json
from pathlib import Path
from collections import defaultdict

INPUT_FILE = r"assets\data\fragments.jsonl"
OUTPUT_DIR = r"assets\data\fragments"


def sanitize_filename(name: str) -> str:
    return name.strip().replace(" ", "_").replace("/", "_").replace("\\", "_").lower()


def main():
    input_path = Path(INPUT_FILE)

    if not input_path.exists():
        raise FileNotFoundError(f"Could not find {INPUT_FILE}")

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)

    with input_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"Skipping invalid JSON on line {line_number}: {e}")
                continue

            topic = obj.get("topic")

            if not topic:
                topic = "uncategorized"

            grouped[topic].append(obj)

    for topic, rows in grouped.items():
        filename = sanitize_filename(topic) + ".jsonl"
        output_file = output_dir / filename

        with output_file.open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"{topic}: {len(rows)} fragments -> {output_file}")

    print()
    print(f"Created {len(grouped)} topic files in '{OUTPUT_DIR}/'")


if __name__ == "__main__":
    main()
