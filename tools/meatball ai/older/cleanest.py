import json
from pathlib import Path
from collections import Counter

# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = Path("assets/data/categories.jsonl")
FRAGMENTS_DIR = Path("assets/data/fragments")

CLEAN_OUTPUT = Path("assets/data/categories.clean.jsonl")
REMOVED_OUTPUT = Path("assets/data/categories.removed_for_analysis.jsonl")
REPORT_OUTPUT = Path("assets/data/categories.clean_report.json")

COMMON_LABEL = "common"


# ============================================================
# HELPERS
# ============================================================


def normalize_label(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


def get_valid_labels():
    labels = set()

    for path in FRAGMENTS_DIR.glob("*.jsonl"):
        label = normalize_label(path.stem)

        if label == COMMON_LABEL:
            continue

        labels.add(label)

    return labels


def get_row_labels(row: dict):
    labels = row.get("categories", row.get("topics", []))

    if not isinstance(labels, list):
        return []

    output = []

    for label in labels:
        label = normalize_label(label)

        if not label:
            continue

        if label == COMMON_LABEL:
            continue

        if label not in output:
            output.append(label)

    return output


def set_row_labels(row: dict, labels):
    new_row = dict(row)

    if "categories" in new_row:
        new_row["categories"] = labels
    elif "topics" in new_row:
        new_row["topics"] = labels
    else:
        new_row["categories"] = labels

    return new_row


# ============================================================
# MAIN
# ============================================================


def main():
    valid_labels = get_valid_labels()

    kept = 0
    removed = 0
    bad_json = 0

    invalid_counter = Counter()

    CLEAN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with DATASET_PATH.open("r", encoding="utf-8-sig") as src, CLEAN_OUTPUT.open(
        "w", encoding="utf-8"
    ) as clean_f, REMOVED_OUTPUT.open("w", encoding="utf-8") as removed_f:

        for line_num, line in enumerate(src, start=1):
            raw = line.strip()

            if not raw:
                continue

            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                bad_json += 1
                removed += 1

                removed_f.write(
                    json.dumps(
                        {
                            "line": line_num,
                            "reason": "bad_json",
                            "raw": raw,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                continue

            labels = get_row_labels(row)

            invalid = [label for label in labels if label not in valid_labels]

            valid = [label for label in labels if label in valid_labels]

            for label in invalid:
                invalid_counter[label] += 1

            if invalid or not valid:
                removed += 1

                removed_f.write(
                    json.dumps(
                        {
                            "line": line_num,
                            "reason": "invalid_or_empty_labels",
                            "invalid_labels": invalid,
                            "valid_labels": valid,
                            "row": row,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                continue

            clean_row = set_row_labels(row, valid)
            clean_f.write(json.dumps(clean_row, ensure_ascii=False) + "\n")
            kept += 1

    report = {
        "dataset": str(DATASET_PATH),
        "fragments_dir": str(FRAGMENTS_DIR),
        "clean_output": str(CLEAN_OUTPUT),
        "removed_output": str(REMOVED_OUTPUT),
        "valid_fragment_labels": len(valid_labels),
        "kept_rows": kept,
        "removed_rows": removed,
        "bad_json_lines": bad_json,
        "invalid_label_counts": invalid_counter.most_common(),
    }

    with REPORT_OUTPUT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Kept rows: {kept}")
    print(f"Removed rows: {removed}")
    print(f"Bad JSON: {bad_json}")
    print(f"Clean file: {CLEAN_OUTPUT}")
    print(f"Removed analysis file: {REMOVED_OUTPUT}")
    print(f"Report: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
