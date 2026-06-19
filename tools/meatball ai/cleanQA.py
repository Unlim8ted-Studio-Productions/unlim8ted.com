import json
import multiprocessing as mp
from pathlib import Path

INPUT_PATH = Path("assets/data/SmartMeatballQA.jsonl")
OUTPUT_PATH = Path("assets/data/SmartMeatballQA.nohistory.jsonl")

GROUP_SIZE = 5000

from testDynamic import (
    load_subject_finder,
    load_subject_inserter,
    predict_subject,
    predict_inserter_op,
    apply_inserter_op,
    normalize_text,
)


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_question(row):
    return row.get("question") or row.get("input") or row.get("message") or ""


def get_answer(row):
    return row.get("answer", "")


def get_topic_subject(row):
    topics = row.get("topics", [])

    if isinstance(topics, list) and topics:
        return str(topics[0]).replace("_", " ").strip()

    if isinstance(topics, str) and topics.strip():
        return topics.replace("_", " ").strip()

    return ""


def format_history_for_subject_model(row):
    raw_history = row.get("history", [])

    if raw_history is None:
        raw_history = []

    if isinstance(raw_history, str):
        raw_history = [raw_history]

    if not isinstance(raw_history, list):
        raw_history = [str(raw_history)]

    formatted = []

    for item in raw_history:
        text = str(item).strip()

        if not text:
            continue

        # Already trained format.
        if text.lower().startswith("user:") or text.lower().startswith("bot:"):
            formatted.append(text)
        else:
            formatted.append(f"User: {text}")

    # Old SmartMeatballQA history only has the previous user question.
    # SubjectFinder was trained with user + bot context, so add a fake bot anchor
    # using the row topic as the known subject.
    subject = get_topic_subject(row)

    if subject:
        formatted.append(f"Bot: {subject} is the thing we were talking about.")

    return formatted


def rewrite_if_needed(subject_pack, inserter_pack, question, history):
    question = str(question).strip()

    if not question:
        return ""

    subject_result = predict_subject(
        subject_pack=subject_pack,
        message=question,
        history=history,
    )

    subject = subject_result.get("subject", "").strip()

    if not subject:
        return question

    op_result = predict_inserter_op(
        inserter_pack=inserter_pack,
        message=question,
        subject=subject,
    )

    op = op_result.get("op", "already_standalone")

    rewritten = apply_inserter_op(
        message=question,
        subject=subject,
        op=op,
    )

    return rewritten or question


def process_group(args):
    group_index, rows = args

    print(f"[worker {group_index}] loading models...")
    subject_pack = load_subject_finder()
    inserter_pack = load_subject_inserter()

    output_rows = []
    rewritten_count = 0
    kept_count = 0
    skipped_count = 0

    for i, row in enumerate(rows, start=1):
        question = get_question(row)
        answer = get_answer(row)

        if not question or answer == "":
            skipped_count += 1
            continue

        history = format_history_for_subject_model(row)

        new_question = rewrite_if_needed(
            subject_pack=subject_pack,
            inserter_pack=inserter_pack,
            question=question,
            history=history,
        )

        if normalize_text(new_question) != normalize_text(question):
            rewritten_count += 1
        else:
            kept_count += 1

        # FINAL CLEAN FORMAT: ONLY QUESTION + ANSWER
        output_rows.append(
            {
                "question": new_question,
                "answer": answer,
            }
        )

        if i % 1000 == 0:
            print(f"[worker {group_index}] {i}/{len(rows)}")

    print(
        f"[worker {group_index}] done | "
        f"out={len(output_rows)} rewritten={rewritten_count} "
        f"kept={kept_count} skipped={skipped_count}"
    )

    return {
        "group_index": group_index,
        "rows": output_rows,
        "rewritten": rewritten_count,
        "kept": kept_count,
        "skipped": skipped_count,
    }


def split_groups(rows, size):
    return [rows[i : i + size] for i in range(0, len(rows), size)]


def main():
    print("Reading SmartMeatballQA...")
    rows = read_jsonl(INPUT_PATH)

    groups = split_groups(rows, GROUP_SIZE)
    jobs = [(i, group) for i, group in enumerate(groups)]

    print(f"Input rows:     {len(rows)}")
    print(f"Group size:     {GROUP_SIZE}")
    print(f"Worker groups:  {len(groups)}")

    ctx = mp.get_context("spawn")

    with ctx.Pool(processes=len(jobs)) as pool:
        results = pool.map(process_group, jobs)

    results.sort(key=lambda x: x["group_index"])

    final_rows = []
    total_rewritten = 0
    total_kept = 0
    total_skipped = 0

    for result in results:
        final_rows.extend(result["rows"])
        total_rewritten += result["rewritten"]
        total_kept += result["kept"]
        total_skipped += result["skipped"]

    write_jsonl(OUTPUT_PATH, final_rows)

    print()
    print("Done.")
    print(f"Input rows:     {len(rows)}")
    print(f"Output rows:    {len(final_rows)}")
    print(f"Rewritten:      {total_rewritten}")
    print(f"Kept original:  {total_kept}")
    print(f"Skipped:        {total_skipped}")
    print(f"Saved to:       {OUTPUT_PATH}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
