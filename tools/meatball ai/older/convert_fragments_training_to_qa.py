import json
import random
import re
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

FRAGMENTS_TRAINING_PATH = Path("assets/data/fragments-training.jsonl")

FRAGMENTS_DIR = Path("assets/data/fragments")

I_CANT_FILE = FRAGMENTS_DIR / "i_cant_i_dont_know.jsonl"

OUTPUT_PATH = Path("assets/data/fragments-training-qa.jsonl")

FALLBACK_UNKNOWN_TEXT = "I don't know."

SKIP_BAD_ROWS = False
# False = keep rows even if some fragments are missing.
# True = skip rows where any fragment ID cannot be found.

INCLUDE_TOPICS = True
INCLUDE_SOURCE_FRAGMENT_IDS = True
INCLUDE_SOURCE = True

SEED = 42


# ============================================================
# RANDOM
# ============================================================

random.seed(SEED)


# ============================================================
# JSONL UTILS
# ============================================================


def read_jsonl(path: Path):
    rows = []

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[skip] bad JSON in {path} line {line_num}: {e}")
                continue

            rows.append(row)

    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ============================================================
# FRAGMENT LOADING
# ============================================================


def extract_fragment_text(row):
    """
    Handles likely fragment formats.

    Examples:
    {"id": "frag_1", "text": "..."}
    {"id": "frag_1", "content": "..."}
    {"id": "frag_1", "answer": "..."}
    {"id": "frag_1", "value": "..."}
    {"id": "frag_1", "body": "..."}
    """

    for key in ["text", "content", "answer", "value", "body"]:
        value = row.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def load_fragments_from_dir(fragments_dir: Path):
    fragment_text_by_id = {}
    source_file_by_id = {}

    if not fragments_dir.exists():
        raise FileNotFoundError(f"Missing fragments directory: {fragments_dir}")

    files = sorted(fragments_dir.glob("*.jsonl"))

    if not files:
        raise RuntimeError(f"No .jsonl files found in {fragments_dir}")

    for file_path in files:
        rows = read_jsonl(file_path)

        for row in rows:
            frag_id = row.get("id")

            if not frag_id:
                continue

            frag_id = str(frag_id).strip()
            text = extract_fragment_text(row)

            if not text:
                print(f"[warn] fragment has no text: {frag_id} in {file_path}")
                continue

            if frag_id in fragment_text_by_id:
                print(
                    f"[warn] duplicate fragment id {frag_id}: "
                    f"{source_file_by_id[frag_id]} and {file_path}. Keeping first."
                )
                continue

            fragment_text_by_id[frag_id] = text
            source_file_by_id[frag_id] = str(file_path).replace("\\", "/")

    return fragment_text_by_id, source_file_by_id


def load_i_cant_choices(path: Path):
    """
    Loads fallback answer choices from i_cant_i_dont_know.jsonl.

    Supports rows like:
    {"id": "...", "text": "..."}
    {"id": "...", "answer": "..."}
    {"text": "..."}
    {"content": "..."}
    """

    choices = []

    if not path.exists():
        print(f"[warn] missing {path}; using fallback unknown text")
        return choices

    rows = read_jsonl(path)

    for row in rows:
        text = extract_fragment_text(row)

        if text:
            choices.append(
                {
                    "id": str(row.get("id", "i_cant_i_dont_know_choice")).strip(),
                    "text": text,
                    "source_file": str(path).replace("\\", "/"),
                }
            )

    if not choices:
        print(f"[warn] no usable i_cant_i_dont_know choices found in {path}")

    return choices


# ============================================================
# TEXT JOINING / CLEANUP
# ============================================================


def normalize_history(history):
    if history is None:
        return []

    if isinstance(history, list):
        return [str(x) for x in history]

    return [str(history)]


def normalize_answer_ids(answer):
    if answer is None:
        return []

    if isinstance(answer, str):
        return [answer]

    if isinstance(answer, list):
        return [str(x).strip() for x in answer if str(x).strip()]

    return [str(answer)]


def smart_join_fragments(texts):
    clean_texts = []

    for text in texts:
        text = str(text).strip()

        if not text:
            continue

        clean_texts.append(text)

    if not clean_texts:
        return FALLBACK_UNKNOWN_TEXT

    answer = " ".join(clean_texts)

    answer = re.sub(r"\s+", " ", answer).strip()
    answer = re.sub(r"\s+([,.;:!?])", r"\1", answer)
    answer = re.sub(r"([,.;:!?])([A-Za-z0-9])", r"\1 \2", answer)

    answer = re.sub(r"\.{2,}", ".", answer)
    answer = re.sub(r",{2,}", ",", answer)
    answer = re.sub(r"!{2,}", "!", answer)
    answer = re.sub(r"\?{2,}", "?", answer)

    answer = re.sub(r"^[,.;:\s]+", "", answer).strip()

    if answer and answer[-1] not in ".!?":
        answer += "."

    return answer


# ============================================================
# CONVERSION
# ============================================================


def convert_rows(
    training_rows,
    fragment_text_by_id,
    source_file_by_id,
    i_cant_choices,
):
    output_rows = []

    missing_counts = {}
    skipped = 0
    converted = 0
    i_cant_replacements = 0

    for idx, row in enumerate(training_rows, start=1):
        question = str(row.get("question", "")).strip()
        history = normalize_history(row.get("history", []))
        topics = row.get("topics", [])

        answer_ids = normalize_answer_ids(row.get("answer", []))

        if not question:
            print(f"[skip] row {idx} missing question")
            skipped += 1
            continue

        if not answer_ids:
            print(f"[skip] row {idx} missing answer ids")
            skipped += 1
            continue

        missing = []
        fragment_texts = []
        fragment_sources = []
        resolved_source_fragment_ids = []
        replacement_notes = []

        for frag_id in answer_ids:
            frag_id = str(frag_id).strip()

            if frag_id == "i_cant_i_dont_know":
                if i_cant_choices:
                    choice = random.choice(i_cant_choices)

                    fragment_texts.append(choice["text"])
                    fragment_sources.append(choice["source_file"])
                    resolved_source_fragment_ids.append(choice["id"])

                    replacement_notes.append(
                        {
                            "original": "i_cant_i_dont_know",
                            "replacement_id": choice["id"],
                            "replacement_text": choice["text"],
                        }
                    )

                    i_cant_replacements += 1
                else:
                    fragment_texts.append(FALLBACK_UNKNOWN_TEXT)
                    fragment_sources.append(str(I_CANT_FILE).replace("\\", "/"))
                    resolved_source_fragment_ids.append("i_cant_i_dont_know_fallback")

                    replacement_notes.append(
                        {
                            "original": "i_cant_i_dont_know",
                            "replacement_id": "i_cant_i_dont_know_fallback",
                            "replacement_text": FALLBACK_UNKNOWN_TEXT,
                        }
                    )

                    i_cant_replacements += 1

                continue

            if frag_id in fragment_text_by_id:
                fragment_texts.append(fragment_text_by_id[frag_id])
                fragment_sources.append(source_file_by_id.get(frag_id, ""))
                resolved_source_fragment_ids.append(frag_id)
            else:
                missing.append(frag_id)
                missing_counts[frag_id] = missing_counts.get(frag_id, 0) + 1

        if missing and SKIP_BAD_ROWS:
            print(f"[skip] row {idx} missing fragments: {missing}")
            skipped += 1
            continue

        if not fragment_texts:
            answer_text = FALLBACK_UNKNOWN_TEXT
        else:
            answer_text = smart_join_fragments(fragment_texts)

        out = {
            "id": row.get("id", f"fragqa_{idx:06d}"),
            "question": question,
            "history": history,
            "answer": answer_text,
        }

        if INCLUDE_TOPICS:
            out["topics"] = topics

        if INCLUDE_SOURCE_FRAGMENT_IDS:
            out["source_fragment_ids"] = answer_ids
            out["resolved_source_fragment_ids"] = resolved_source_fragment_ids

        if replacement_notes:
            out["i_cant_replacements"] = replacement_notes

        if missing:
            out["missing_fragment_ids"] = missing

        if INCLUDE_SOURCE:
            out["source"] = "fragments-training.jsonl"
            out["fragment_source_files"] = fragment_sources

        for key in ["intent", "project", "project_key", "category", "tags"]:
            if key in row:
                out[key] = row[key]

        output_rows.append(out)
        converted += 1

    return output_rows, {
        "converted": converted,
        "skipped": skipped,
        "missing_counts": missing_counts,
        "i_cant_replacements": i_cant_replacements,
    }


# ============================================================
# MAIN
# ============================================================


def main():
    print("[1] loading fragments...")
    fragment_text_by_id, source_file_by_id = load_fragments_from_dir(FRAGMENTS_DIR)
    print(f"loaded fragments: {len(fragment_text_by_id)}")

    print("[2] loading i_cant_i_dont_know choices...")
    i_cant_choices = load_i_cant_choices(I_CANT_FILE)
    print(f"loaded i_cant choices: {len(i_cant_choices)}")

    print("[3] loading fragments-training rows...")
    training_rows = read_jsonl(FRAGMENTS_TRAINING_PATH)
    print(f"loaded training rows: {len(training_rows)}")

    print("[4] converting...")
    output_rows, stats = convert_rows(
        training_rows=training_rows,
        fragment_text_by_id=fragment_text_by_id,
        source_file_by_id=source_file_by_id,
        i_cant_choices=i_cant_choices,
    )

    print("[5] writing output...")
    write_jsonl(OUTPUT_PATH, output_rows)

    print()
    print("================ CONVERSION DONE ================")
    print(f"output:              {OUTPUT_PATH}")
    print(f"converted:           {stats['converted']}")
    print(f"skipped:             {stats['skipped']}")
    print(f"i_cant replacements: {stats['i_cant_replacements']}")

    missing_counts = stats["missing_counts"]

    print(f"unique missing fragment ids: {len(missing_counts)}")

    if missing_counts:
        print()
        print("top missing fragment ids:")
        for frag_id, count in sorted(
            missing_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:50]:
            print(f"  {count:5d}  {frag_id}")

    print("=================================================")
    print()


if __name__ == "__main__":
    main()
