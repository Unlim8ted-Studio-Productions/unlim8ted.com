#!/usr/bin/env python3

import json
from pathlib import Path

CATEGORIES_PATH = Path(r"assets\data\categories.jsonl")
FRAGMENTS_PATH = Path(r"assets\data\fragments-training.jsonl")
OUTPUT_PATH = Path(r"assets\data\fragments-training.updated.jsonl")


FIELDS_TO_COPY = ["topics", "question", "history"]


def read_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {e}")

    return rows


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    categories = read_jsonl(CATEGORIES_PATH)
    fragments = read_jsonl(FRAGMENTS_PATH)

    if len(categories) != len(fragments):
        raise ValueError(
            f"Line count mismatch: "
            f"{CATEGORIES_PATH} has {len(categories)} lines, "
            f"{FRAGMENTS_PATH} has {len(fragments)} lines."
        )

    updated_rows = []

    for index, (category_row, fragment_row) in enumerate(
        zip(categories, fragments), start=1
    ):
        for field in FIELDS_TO_COPY:
            if field not in category_row:
                raise KeyError(
                    f"Missing field '{field}' on line {index} of {CATEGORIES_PATH}"
                )

            fragment_row[field] = category_row[field]

        updated_rows.append(fragment_row)

    write_jsonl(OUTPUT_PATH, updated_rows)

    print(f"Done. Wrote updated file to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
