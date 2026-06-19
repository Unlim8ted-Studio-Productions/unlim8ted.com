import argparse
import json
import re
from pathlib import Path


def clean_text(x):
    x = str(x or "").strip()
    x = re.sub(r"\s+", " ", x)
    return x


def load_jsonl(path):
    rows = []

    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"Bad JSON skipped at {path}:{line_num}: {e}")
                continue

            if isinstance(obj, dict):
                rows.append(obj)

    return rows


def get_fragment_text(row):
    for key in ["text", "answer", "content", "fragment", "value"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return clean_text(value)

    return ""


def build_id_to_text(fragment_paths):
    id_to_text = {}
    duplicate_ids = []

    for path in fragment_paths:
        rows = load_jsonl(path)

        for row in rows:
            fid = row.get("id")

            if not isinstance(fid, str) or not fid.strip():
                continue

            fid = fid.strip()
            text = get_fragment_text(row)

            if not text:
                continue

            if fid in id_to_text:
                duplicate_ids.append(fid)

            id_to_text[fid] = text

    if duplicate_ids:
        print("Duplicate IDs found:", len(duplicate_ids))
        print("First duplicates:", duplicate_ids[:20])

    print("Loaded fragment texts:", len(id_to_text))

    return id_to_text


def assemble(ids, id_to_text):
    parts = []

    missing = []

    for fid in ids:
        text = id_to_text.get(fid)

        if text:
            parts.append(text)
        else:
            missing.append(fid)

    output = ""

    for part in parts:
        if not output:
            output = part
        elif output.endswith((" ", "\n")):
            output += part
        elif part.startswith((".", ",", "!", "?", ";", ":")):
            output += part
        else:
            output += " " + part

    return output, missing


def get_target_ids(row):
    target = row.get("target", {})

    if not isinstance(target, dict):
        return []

    ids = target.get("fragments", [])

    if not isinstance(ids, list):
        return []

    return [x for x in ids if isinstance(x, str)]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training",
        default="assets/data/fragment_training_from_qa_local_test.jsonl",
    )

    parser.add_argument(
        "--fragments",
        default="fragments.jsonl",
    )

    parser.add_argument(
        "--bridges",
        default="fragment_bridges.jsonl",
    )

    parser.add_argument(
        "--out",
        default="assets/data/fragment_training_from_qa_local_test_decoded.jsonl",
    )

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=25)

    args = parser.parse_args()

    id_to_text = build_id_to_text(
        [
            args.fragments,
            args.bridges,
        ]
    )

    rows = load_jsonl(args.training)

    if args.limit > 0:
        rows = rows[: args.limit]

    print("Training rows:", len(rows))
    print("Output:", args.out)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).unlink(missing_ok=True)

    missing_total = 0
    decoded_total = 0

    with Path(args.out).open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows, 1):
            target_ids = get_target_ids(row)
            decoded_text, missing_ids = assemble(target_ids, id_to_text)

            missing_total += len(missing_ids)

            decoded = dict(row)
            decoded["decoded_target"] = {
                "shape": row.get("target", {}).get("shape", ""),
                "fragments": target_ids,
                "text": decoded_text,
                "missing_ids": missing_ids,
            }

            # Easier quick-view fields.
            decoded["decoded_output"] = decoded_text
            decoded["missing_fragment_ids"] = missing_ids

            f.write(json.dumps(decoded, ensure_ascii=False) + "\n")
            decoded_total += 1

            if i <= 10 or (args.print_every > 0 and i % args.print_every == 0):
                print()
                print(f"ROW {i}/{len(rows)}")
                print("input:", clean_text(row.get("input"))[:220])
                print("official:", clean_text(row.get("official_answer"))[:220])
                print("target ids:", target_ids)
                print("decoded:", decoded_text[:500])

                if missing_ids:
                    print("missing ids:", missing_ids)

    print()
    print("DONE")
    print("decoded rows:", decoded_total)
    print("missing id refs:", missing_total)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
