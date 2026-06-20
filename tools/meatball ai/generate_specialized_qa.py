import argparse
import json
import math
from pathlib import Path
from typing import Optional


UNLIM_KEYWORDS = [
    "unlim8ted",
    "timecat",
    "square pixel",
    "square pixels",
    "unicornia",
    "life of a meatball",
    "the glitch",
    "puzzle square",
    "air hockey online",
    "unlim8ted chess",
    "ftl chooseyourside",
    "chessvr",
    "kindle e-ink games",
    "confuzzled",
    "download any youtube video",
    "face stuff",
    "multiplayer physics simulator",
    "easy pygame ui maker",
    "pygame ui designer",
    "copy keyframes to selected blender addon",
    "chatapp",
    "organisms sim",
    "ftl node based modding",
    "wrighting",
    "music ai gen",
    "cineme",
    "unlim8ted phone",
    "services.unlim8ted.com",
    "assets.unlim8ted.com",
    "paint-app-v5.9",
    "star tracker",
    "wisesize",
    "wearwise",
    "podcasts",
    "physical items",
    "images",
]

QUESTION_TEMPLATES = [
    "What is {subject}?",
    "Tell me about {subject}.",
    "Give me an overview of {subject}.",
    "How would you describe {subject}?",
    "What does {subject} mean?",
    "What is the main idea of {subject}?",
    "What should someone know about {subject}?",
    "Why is {subject} important?",
    "Why does {subject} matter?",
    "What makes {subject} useful?",
    "What is notable about {subject}?",
    "What role does {subject} play in {topic}?",
    "How does {subject} fit into {topic}?",
    "How does {subject} connect to {topic}?",
    "Why do people study {subject}?",
    "Why do people care about {subject}?",
    "What is a simple explanation of {subject}?",
    "What is the purpose of {subject}?",
    "What is the value of {subject}?",
    "How can {subject} be understood clearly?",
    "What is a practical way to think about {subject}?",
    "What does {subject} help explain?",
    "What is one clear way to define {subject}?",
    "How would a beginner understand {subject}?",
    "What is the core meaning of {subject}?",
    "What is the basic idea behind {subject}?",
    "What does {subject} contribute to {topic}?",
    "How does {subject} support better understanding of {topic}?",
    "What is the short explanation of {subject}?",
    "How can {subject} be explained simply?",
    "What is the big-picture role of {subject}?",
    "What is a direct explanation of {subject}?",
    "What does {subject} tell us about {topic}?",
    "How does {subject} shape the way people think about {topic}?",
]

ANSWER_WRAPPERS = [
    "{answer}",
    "In simple terms, {answer}",
    "Put simply, {answer}",
    "At a basic level, {answer}",
    "The short version is this: {answer}",
    "A clear way to explain it is this: {answer}",
    "Broadly, {answer}",
    "In practice, {answer}",
    "The main point is that {answer}",
    "A useful summary is: {answer}",
    "The core idea is that {answer}",
    "One clear explanation is: {answer}",
    "A practical explanation is: {answer}",
    "For most purposes, {answer}",
    "The simple answer is that {answer}",
    "The direct answer is that {answer}",
    "A concise explanation is: {answer}",
    "At a glance, {answer}",
    "The key thing to understand is that {answer}",
    "A straightforward way to put it is: {answer}",
    "The basic picture is that {answer}",
    "In the context of {topic}, {answer}",
    "Within {topic}, {answer}",
    "For {topic}, {answer}",
    "A beginner-friendly explanation is: {answer}",
    "The main takeaway is that {answer}",
    "One good way to frame it is: {answer}",
    "A useful starting point is this: {answer}",
    "The central idea is that {answer}",
    "The broad explanation is: {answer}",
]

FOLLOWUPS = [
    "",
    " This helps make the idea more practical.",
    " This matters because it shapes how related systems are understood.",
    " This is useful because it connects the concept to real examples.",
    " This matters because it influences how people reason about the topic.",
    " This gives the concept clearer structure and context.",
    " This helps explain why the concept appears so often in the subject.",
    " This makes the idea easier to apply or recognize.",
]


def is_unlim_row(question: str, answer: str) -> bool:
    haystack = f"{question} {answer}".lower()
    return any(keyword in haystack for keyword in UNLIM_KEYWORDS)


def extract_subject(question: str) -> Optional[str]:
    text = question.strip().rstrip(".?!")
    lowered = text.lower()
    prefixes = [
        "what is ",
        "what are ",
        "tell me about ",
        "give me an overview of ",
        "how would you describe ",
        "what is the main idea of ",
        "what should someone know about ",
        "what is a simple explanation of ",
        "what is the purpose of ",
        "what is the value of ",
        "what is one clear way to define ",
        "what is the core meaning of ",
        "what is the basic idea behind ",
        "what is the short explanation of ",
        "what is a practical way to think about ",
        "what is the big-picture role of ",
        "what is a direct explanation of ",
    ]
    for prefix in prefixes:
        if lowered.startswith(prefix):
            subject = text[len(prefix):].strip()
            for suffix in [
                " play in games",
                " play in game",
                " play in technology",
                " fit into games",
                " fit into technology",
                " connect to games",
                " connect to technology",
            ]:
                if subject.lower().endswith(suffix):
                    subject = subject[: -len(suffix)].strip()
            if subject:
                return subject
    return None


def normalize_question(question: str) -> str:
    q = question.strip()
    if q and q[0].islower():
        q = q[0].upper() + q[1:]
    if q and q[-1] not in ".?!":
        q += "?"
    return q


def normalize_answer(answer: str) -> str:
    a = answer.strip()
    if a and a[0].islower():
        a = a[0].upper() + a[1:]
    if a and a[-1] not in ".?!":
        a += "."
    return a


def topic_label_from_stem(stem: str) -> str:
    label = stem.replace("_", " ")
    special = {
        "ai": "AI",
        "health basics": "health basics",
        "earth science": "earth science",
        "unlim8ted": "Unlim8ted",
    }
    return special.get(label, label)


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "question" in obj and "answer" in obj:
                rows.append(obj)
    return rows


def build_seed_groups(rows):
    unlim = []
    general = []
    for row in rows:
        subject = extract_subject(row["question"])
        if not subject:
            continue
        if is_unlim_row(row["question"], row["answer"]):
            unlim.append(row)
        else:
            general.append(row)
    return unlim, general


def generate_variants(seed_rows, target_count, topic_label, seen):
    if not seed_rows or target_count <= 0:
        return []

    generated = []
    q_len = len(QUESTION_TEMPLATES)
    a_len = len(ANSWER_WRAPPERS)
    f_len = len(FOLLOWUPS)
    attempts = 0
    max_attempts = target_count * 20

    while len(generated) < target_count and attempts < max_attempts:
        idx = attempts
        seed = seed_rows[idx % len(seed_rows)]
        subject = extract_subject(seed["question"])
        if not subject:
            attempts += 1
            continue
        q_template = QUESTION_TEMPLATES[(idx // len(seed_rows)) % q_len]
        a_template = ANSWER_WRAPPERS[(idx // (len(seed_rows) * q_len)) % a_len]
        followup = FOLLOWUPS[(idx // (len(seed_rows) * q_len * a_len)) % f_len]

        question = normalize_question(
            q_template.format(subject=subject, topic=topic_label)
        )
        answer = normalize_answer(
            a_template.format(
                answer=seed["answer"].strip().rstrip(".?!") + ".",
                subject=subject,
                topic=topic_label,
            )
            + followup
        )

        key = f"{question}\t{answer}"
        if key not in seen:
            seen.add(key)
            generated.append({"question": question, "answer": answer})
        attempts += 1

    return generated


def append_rows(path: Path, rows):
    if not rows:
        return
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def process_file(path: Path, append_count: int):
    rows = load_rows(path)
    topic_label = topic_label_from_stem(path.stem)
    seen = {
        f'{row["question"].strip()}\t{row["answer"].strip()}'
        for row in rows
    }

    unlim_rows, general_rows = build_seed_groups(rows)
    total_existing = len(rows)
    unlim_existing = len(unlim_rows)

    target_unlim_total = math.ceil((total_existing + append_count) / 3)
    need_unlim = 0
    if unlim_existing > 0:
        need_unlim = max(0, target_unlim_total - unlim_existing)
        need_unlim = min(need_unlim, append_count)

    need_general = append_count - need_unlim

    # If there are no general seeds, reuse unlim seeds rather than failing.
    if not general_rows:
        general_rows = unlim_rows

    new_rows = []
    new_rows.extend(generate_variants(unlim_rows, need_unlim, topic_label, seen))
    new_rows.extend(generate_variants(general_rows, need_general, topic_label, seen))

    # If uniqueness limits prevent hitting the target, fall back to all rows.
    if len(new_rows) < append_count:
        fallback = generate_variants(rows, append_count - len(new_rows), topic_label, seen)
        new_rows.extend(fallback)

    append_rows(path, new_rows)
    return {
        "file": path.name,
        "existing": total_existing,
        "appended": len(new_rows),
        "unlim_existing": unlim_existing,
        "unlim_target_new": need_unlim,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        default="assets/data/specialized_QA",
        help="Directory containing specialized QA jsonl files.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10000,
        help="Rows to append per file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be appended without writing changes.",
    )
    args = parser.parse_args()

    base = Path(args.dir)
    ignore = {"combined_test.jsonl"}
    files = sorted(
        path
        for path in base.glob("*.jsonl")
        if path.name not in ignore and not path.name.startswith("combined_")
    )

    results = []
    for path in files:
        if args.dry_run:
            rows = load_rows(path)
            unlim_rows, general_rows = build_seed_groups(rows)
            total_existing = len(rows)
            unlim_existing = len(unlim_rows)
            target_unlim_total = math.ceil((total_existing + args.count) / 3)
            need_unlim = 0
            if unlim_existing > 0:
                need_unlim = max(0, target_unlim_total - unlim_existing)
                need_unlim = min(need_unlim, args.count)
            results.append(
                {
                    "file": path.name,
                    "existing": total_existing,
                    "appended": args.count,
                    "unlim_existing": unlim_existing,
                    "unlim_target_new": need_unlim,
                }
            )
        else:
            results.append(process_file(path, args.count))

    for result in results:
        print(
            f'{result["file"]}\t'
            f'existing={result["existing"]}\t'
            f'appended={result["appended"]}\t'
            f'unlim_existing={result["unlim_existing"]}\t'
            f'unlim_target_new={result["unlim_target_new"]}'
        )


if __name__ == "__main__":
    main()
