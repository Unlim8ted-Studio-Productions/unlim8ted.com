import json
import random
import re
import time
import sys
import os
import logging
import traceback
import faulthandler
import multiprocessing as mp
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

faulthandler.enable()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# ============================================================
# CONFIG
# ============================================================

GENERATOR_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

OUTPUT_DIR = Path("assets/data/specialized_qa_raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROWS_PER_TOPIC = 5000

NUM_WORKERS = 4

EXAMPLES_PER_TOPIC = 30
ROWS_PER_BUCKET_BATCH = 8
GENERAL_BATCH_SIZE = 10

MAX_ATTEMPTS_PER_TOPIC = 3000

GEN_MAX_NEW_TOKENS = 768
GEN_TEMPERATURE = 0.65
GEN_TOP_P = 0.9

SEPARATOR = "|||ANSWER|||"

SEED = 1337
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# TOPICS — NO UNLIM8TED
# ============================================================

TOPICS = {
    "insects": "insects, bugs, colonies, pollination, metamorphosis, body parts",
    "food": "food, cooking, baking, ingredients, meals, taste, nutrition basics",
    "vehicles": "cars, trains, airplanes, boats, bicycles, engines, transportation",
    "space": "space, planets, moons, stars, galaxies, spacecraft, astronomy",
    "earth_science": "weather, climate, oceans, rocks, volcanoes, earthquakes, geology",
    "physics": "motion, forces, energy, electricity, magnets, light, sound, waves",
    "chemistry": "atoms, molecules, elements, reactions, acids, bases, materials",
    "biology": "cells, organs, animal types, plant types, ecosystems, genetics, evolution",
    "technology": "computers, internet, software, hardware, apps, websites, data",
    "ai": "artificial intelligence, machine learning, neural networks, datasets, models",
    "math": "numbers, shapes, patterns, operations, equations, geometry, probability",
    "programming": "programming, Python, JavaScript, HTML, CSS, JSON, APIs, debugging",
    "games": "video games, board games, rules, mechanics, players, strategy",
    "movies": "movies, film, animation, characters, stories, scenes, genres",
    "books": "books, stories, authors, chapters, characters, reading",
    "music": "music, songs, instruments, rhythm, melody, albums, recording",
    "sports": "sports, exercise, teams, rules, training, equipment",
    "history": "history, ancient civilizations, wars, inventions, leaders, historical events",
    "countries": "countries, cities, geography, culture, landmarks, maps",
    "art": "art, drawing, painting, photography, sculpture, design",
    "health_basics": "basic health, sleep, exercise, hygiene, nutrition, body systems",
}

SUBJECT_BUCKETS = [
    "what_is",
    "what_does",
    "where_is",
    "why_does",
    "how_does",
    "parts",
    "uses",
    "features",
    "classification",
    "examples",
    "comparison",
    "misconception",
    "edge_case",
    "followup_rewritten",
    "simple_explanation",
    "more_detail",
]

GENERAL_BUCKETS = [
    "general_topic_question",
    "broad_explanation",
    "common_confusion",
    "beginner_question",
    "comparison_across_subjects",
    "edge_case",
    "clarification",
    "fallback",
]

generator_tokenizer = None
generator_model = None


def print_system_info():
    logging.info(f"Python: {sys.version}")
    logging.info(f"Torch: {torch.__version__}")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logging.info(f"CUDA version: {torch.version.cuda}")
        logging.info(f"GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            logging.info(f"GPU {i}: {props.name}")
            logging.info(f"GPU {i} VRAM total: {props.total_memory / 1024**3:.2f} GB")
            logging.info(
                f"GPU {i} memory free: {free / 1024**3:.2f} GB / {total / 1024**3:.2f} GB"
            )


def load_model(model_name):
    logging.info("=" * 80)
    logging.info(f"PID {os.getpid()} START load_model: {model_name}")
    logging.info("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    logging.info(f"PID {os.getpid()} Loading model: {model_name}")
    logging.info(f"PID {os.getpid()} dtype={dtype}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    model.to(DEVICE)
    model.eval()

    logging.info(f"PID {os.getpid()} SUCCESS load_model: {model_name}")

    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        logging.info(
            f"PID {os.getpid()} GPU memory after load: "
            f"free={free / 1024**3:.2f} GB / {total / 1024**3:.2f} GB"
        )

    return tokenizer, model


def init_worker():
    global generator_tokenizer
    global generator_model

    pid = os.getpid()
    logging.info(f"PID {pid} INIT START")

    random.seed(SEED + pid)
    torch.manual_seed(SEED + pid)

    generator_tokenizer, generator_model = load_model(GENERATOR_MODEL_NAME)

    logging.info(f"PID {pid} INIT DONE / MODEL LOADED")


def clean_text(text):
    text = str(text).replace("\n", " ").replace("\t", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_answer_text(text):
    text = str(text)
    text = text.replace(SEPARATOR, " ")
    text = text.replace("|||ANSWER||", " ")
    text = text.replace("|||", " ")
    text = text.replace("<TAB>", " ")
    text = text.replace("|...", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize(text):
    text = str(text).lower().strip()
    text = text.replace(SEPARATOR.lower(), " ")
    text = text.replace("|||", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def row_key(row):
    return normalize(row.get("question", ""))


def is_bad_answer(answer):
    a = normalize(answer)

    if not a:
        return True

    if a in {"answer", "the answer", "none", "n a", "na", "unknown"}:
        return True

    if len(a) < 3:
        return True

    return False


def is_bad_question(question):
    q = normalize(question)

    if not q:
        return True

    if q in {"question", "the question"}:
        return True

    if len(q) < 3:
        return True

    return False


def load_existing(path):
    if not path.exists():
        return []

    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            if isinstance(obj, dict) and "question" in obj and "answer" in obj:
                q = clean_text(obj["question"])
                a = clean_answer_text(obj["answer"])

                if not is_bad_question(q) and not is_bad_answer(a):
                    rows.append({"question": q, "answer": a})

    return rows


def append_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.no_grad()
def chat_generate(messages, max_new_tokens, temperature, top_p):
    global generator_tokenizer
    global generator_model

    input_text = generator_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = generator_tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    ).to(generator_model.device)

    output_ids = generator_model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.08,
        pad_token_id=generator_tokenizer.eos_token_id,
        eos_token_id=generator_tokenizer.eos_token_id,
    )

    generated = generator_tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )

    return generated.strip()


def voice_style_text():
    return """
Voice style:
- The speaker is Meatball.
- Meatball is casual, clear, slightly weird, playful, and useful.
- Facts first. Meatball flavor second.
- Do not sound like ChatGPT.
- Do not say "As an AI", "Certainly", "Of course", "Here are", or "I'd be happy to help".
- Answers should usually be 1 to 3 sentences.
- About 20 percent of answers may use a light Meatball-style phrase.
- Do not force a joke into every answer.
- Do not make the answer confusing.
- Do not use bullet points inside answers.
""".strip()


def strict_format_rules():
    return f"""
Output format:
Each row must be:

question {SEPARATOR} answer

Critical formatting rules:
- The separator must be typed exactly as: {SEPARATOR}
- Every row must contain exactly one separator.
- Do not write <TAB>.
- Do not use a tab.
- Do not use a single pipe.
- Do not use this format: subject | answer | question
- Do not put the answer on a new line.
- Do not write the subject first.
- Do not number the rows.
- Do not use bullets.
- Do not output JSON.
- Output only the rows.

Correct format:
what is a rabbit {SEPARATOR} A rabbit is a small mammal with long ears and strong back legs.

Wrong format:
rabbit | A rabbit is a small mammal | what is a rabbit

Wrong format:
what is a rabbit<TAB>A rabbit is a small mammal.

Wrong format:
what is a rabbit
A rabbit is a small mammal.
""".strip()


def build_examples_prompt(topic_name, topic_description, count):
    return f"""
Generate {count} specific examples for this dataset topic.

Topic name:
{topic_name}

Topic description:
{topic_description}

Output format:
one example per line

Rules:
- Do not output JSON.
- Do not number the lines.
- Do not use bullets.
- Use specific examples, not broad categories.
- Avoid duplicates.
- Use common examples a user might ask about.

Example:
dog
cat
walrus
shark

Now output only the examples.
""".strip()


def build_subject_bucket_prompt(topic_name, topic_description, subject, bucket, count):
    return f"""
You are generating high-quality training rows for a tiny browser-run chatbot.

Topic:
{topic_name}

Topic description:
{topic_description}

Specific subject:
{subject}

Bucket:
{bucket}

Generate EXACTLY {count} rows.

{strict_format_rules()}

{voice_style_text()}

Content rules:
- Left side is the question.
- Right side is the answer.
- The question must be standalone.
- The left side ALWAYS must be a complete question, not just a subject. For example, if the subject is "{subject}", the question should be something like "what is {subject}?", not just "{subject}".
- The answer must never be only the word "answer".
- Never put {SEPARATOR} inside the answer.
- Never put ||| inside the answer except the exact separator.
- If the bucket is followup_rewritten, write the already-rewritten standalone question, not "what is it".
- Keep answers factually safe and general if exact details are uncertain.
- Do not invent fake proper nouns.
- Avoid repeating the same question pattern.
- Avoid overly long answers.

Good rows:
what is a walrus {SEPARATOR} A walrus is a large marine mammal with tusks, whiskers, and thick blubber.
why do walruses have tusks {SEPARATOR} Walruses use tusks for defense, social displays, and pulling themselves onto ice.
how does a flashlight work {SEPARATOR} A flashlight uses a battery to send electricity through a bulb or LED, which produces light.

- Every question and answer must be mainly about this topic: {topic_name}.
- Every question and answer must be mainly about this subject: {subject}
- Do not use examples from other topics.

Generate rows now.
""".strip()


def build_general_prompt(topic_name, topic_description, bucket, examples, count):
    example_text = ", ".join(examples[:30])

    return f"""
You are generating high-quality training rows for a tiny browser-run chatbot.

Topic:
{topic_name}

Topic description:
{topic_description}

Question bucket:
{bucket}

Useful topic examples:
{example_text}

Generate EXACTLY {count} rows.

{strict_format_rules()}

{voice_style_text()}

Content rules:
- Left side is the question.
- The left side must be a complete question, not just a subject.
- Right side is the answer.
- The question must be standalone.
- The answer must never be only the word "answer".
- Never put {SEPARATOR} inside the answer.
- Never put ||| inside the answer except the exact separator.
- Answers must be useful, short, and factually safe.
- Do not invent fake proper nouns.
- Include common, casual, beginner, and edge-case questions.
- Avoid duplicate question patterns.
- talk in a styalized manner like this:  easy. This is where the sauce is on and the ideas do not stay trapped in one tiny box.

Good rows:
what are common animals people keep as pets? {SEPARATOR} Common pets include dogs, cats, rabbits, fish, birds, hamsters, and guinea pigs.
why do some animals live in groups? {SEPARATOR} Some animals live in groups for protection, hunting, raising young, or finding food.
what if I ask something unclear? {SEPARATOR} If the question is unclear, Meatball should ask a simple follow-up instead of pretending to know.

Generate rows now.
""".strip()


def parse_examples(text):
    examples = []
    seen = set()

    for line in text.splitlines():
        line = clean_text(line)

        if not line:
            continue

        line = re.sub(r"^[-*•]\s*", "", line)
        line = re.sub(r"^\d+[\).\s-]+", "", line).strip()

        if not line:
            continue

        if "{" in line or "}" in line or "[" in line or "]" in line:
            continue

        key = normalize(line)

        if key and key not in seen:
            seen.add(key)
            examples.append(line)

    return examples[:EXAMPLES_PER_TOPIC]


def parse_qa_rows(text):
    rows = []

    for raw_line in text.splitlines():
        raw = raw_line.strip()

        if not raw:
            continue

        raw = re.sub(r"^[-*•]\s*", "", raw)
        raw = re.sub(r"^\d+[\).\s-]+", "", raw).strip()

        question = ""
        answer = ""

        if SEPARATOR in raw:
            parts = raw.split(SEPARATOR, 1)
            question = clean_text(parts[0])
            answer = clean_answer_text(parts[1])

        elif "|||answer|||" in raw.lower():
            parts = re.split(
                r"\|\|\|answer\|\|\|",
                raw,
                maxsplit=1,
                flags=re.IGNORECASE,
            )
            question = clean_text(parts[0])
            answer = clean_answer_text(parts[1])

        else:
            continue

        question = re.sub(
            r"^(question|q)\s*:\s*", "", question, flags=re.IGNORECASE
        ).strip()

        answer = re.sub(r"^(answer|a)\s*:\s*", "", answer, flags=re.IGNORECASE).strip()

        question = clean_text(question)
        answer = clean_answer_text(answer)

        if is_bad_question(question):
            logging.info(f"SKIP BAD QUESTION: {raw[:250]}")
            continue

        if is_bad_answer(answer):
            logging.info(f"SKIP BAD ANSWER: {raw[:250]}")
            continue

        rows.append({"question": question, "answer": answer})

    logging.info(f"PID {os.getpid()} PARSED ROWS: {len(rows)}")
    return rows


def generate_examples(topic_name, topic_description):
    messages = [
        {"role": "system", "content": "You generate plain text only. No JSON."},
        {
            "role": "user",
            "content": build_examples_prompt(
                topic_name,
                topic_description,
                EXAMPLES_PER_TOPIC,
            ),
        },
    ]

    text = chat_generate(
        messages,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
    )

    examples = parse_examples(text)

    if not examples:
        logging.warning(
            f"[{topic_name}] Example generation failed. Raw output: {text[:1000]}"
        )

    return examples


def generate_subject_rows(topic_name, topic_description, subject, bucket):
    messages = [
        {
            "role": "system",
            "content": f"You generate plain text training rows only. Every row must use exactly this separator: {SEPARATOR}",
        },
        {
            "role": "user",
            "content": build_subject_bucket_prompt(
                topic_name,
                topic_description,
                subject,
                bucket,
                ROWS_PER_BUCKET_BATCH,
            ),
        },
    ]

    text = chat_generate(
        messages,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
    )

    rows = parse_qa_rows(text)

    if not rows:
        logging.warning(
            f"No rows parsed for {topic_name}/{subject}/{bucket}. Raw: {text[:1000]}"
        )

    return rows


def generate_general_rows(topic_name, topic_description, bucket, examples):
    messages = [
        {
            "role": "system",
            "content": f"You generate plain text training rows only. Every row must use exactly this separator: {SEPARATOR}",
        },
        {
            "role": "user",
            "content": build_general_prompt(
                topic_name,
                topic_description,
                bucket,
                examples,
                GENERAL_BATCH_SIZE,
            ),
        },
    ]

    text = chat_generate(
        messages,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        temperature=GEN_TEMPERATURE,
        top_p=GEN_TOP_P,
    )

    rows = parse_qa_rows(text)

    if not rows:
        logging.warning(
            f"No general rows parsed for {topic_name}/{bucket}. Raw: {text[:1000]}"
        )

    return rows


def generate_topic(job):
    topic_name, topic_description = job

    pid = os.getpid()
    logging.info(f"PID {pid} START TOPIC {topic_name}")

    out_path = OUTPUT_DIR / f"{topic_name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.touch(exist_ok=True)

    existing = load_existing(out_path)
    seen = {row_key(row) for row in existing}

    logging.info("")
    logging.info("=" * 80)
    logging.info(f"[{topic_name}] START")
    logging.info(f"[{topic_name}] PID: {pid}")
    logging.info(f"[{topic_name}] Existing rows: {len(existing)}")
    logging.info(f"[{topic_name}] Target rows:   {ROWS_PER_TOPIC}")
    logging.info(f"[{topic_name}] Output file:    {out_path}")
    logging.info("=" * 80)

    if len(seen) >= ROWS_PER_TOPIC:
        logging.info(f"[{topic_name}] already complete, skipping.")
        return {
            "topic": topic_name,
            "status": "skipped_complete",
            "rows": len(seen),
        }

    examples = generate_examples(topic_name, topic_description)

    if not examples:
        examples = [topic_name.replace("_", " ")]

    logging.info(f"[{topic_name}] Examples ({len(examples)}): {examples}")

    subject_bucket_jobs = []

    for subject in examples:
        for bucket in SUBJECT_BUCKETS:
            subject_bucket_jobs.append((subject, bucket))

    random.shuffle(subject_bucket_jobs)

    attempts = 0
    added_total = 0
    duplicate_total = 0
    parse_empty_total = 0

    start_time = time.time()

    while len(seen) < ROWS_PER_TOPIC and attempts < MAX_ATTEMPTS_PER_TOPIC:
        attempts += 1
        remaining = ROWS_PER_TOPIC - len(seen)

        if subject_bucket_jobs and random.random() < 0.8:
            subject, bucket = subject_bucket_jobs.pop(0)
            subject_bucket_jobs.append((subject, bucket))

            if attempts % 10 == 0:
                random.shuffle(subject_bucket_jobs)

            logging.info(
                f"[{topic_name}] attempt={attempts} "
                f"subject={subject} bucket={bucket} remaining={remaining}"
            )

            candidate_rows = generate_subject_rows(
                topic_name,
                topic_description,
                subject,
                bucket,
            )
        else:
            bucket = random.choice(GENERAL_BUCKETS)

            logging.info(
                f"[{topic_name}] attempt={attempts} "
                f"general_bucket={bucket} remaining={remaining}"
            )

            candidate_rows = generate_general_rows(
                topic_name,
                topic_description,
                bucket,
                examples,
            )

        if not candidate_rows:
            parse_empty_total += 1
            continue

        new_rows = []

        for row in candidate_rows:
            row["question"] = clean_text(row["question"])
            row["answer"] = clean_answer_text(row["answer"])

            if is_bad_question(row["question"]) or is_bad_answer(row["answer"]):
                continue

            key = row_key(row)

            if not key:
                continue

            if key in seen:
                duplicate_total += 1
                continue

            seen.add(key)
            new_rows.append(row)

            if len(seen) >= ROWS_PER_TOPIC:
                break

        if new_rows:
            append_rows(out_path, new_rows)

        added_total += len(new_rows)

        elapsed = time.time() - start_time
        rate = added_total / elapsed if elapsed > 0 and added_total > 0 else 0
        eta_seconds = remaining / rate if rate > 0 else None
        eta_text = "unknown"

        if eta_seconds is not None:
            eta_text = f"{eta_seconds / 60:.1f} min"

        logging.info(
            f"[{topic_name}] generated={len(candidate_rows)} "
            f"added={len(new_rows)} "
            f"duplicates={duplicate_total} "
            f"total={len(seen)} "
            f"eta={eta_text}"
        )

    status = "complete" if len(seen) >= ROWS_PER_TOPIC else "stopped_early"

    logging.info("")
    logging.info(f"[DONE] {topic_name}")
    logging.info(f"[{topic_name}] status:              {status}")
    logging.info(f"[{topic_name}] rows total:          {len(seen)}")
    logging.info(f"[{topic_name}] added this run:      {added_total}")
    logging.info(f"[{topic_name}] duplicates this run: {duplicate_total}")
    logging.info(f"[{topic_name}] empty parses:        {parse_empty_total}")
    logging.info(f"[{topic_name}] output file:         {out_path}")

    return {
        "topic": topic_name,
        "status": status,
        "rows": len(seen),
        "added": added_total,
        "duplicates": duplicate_total,
        "empty_parses": parse_empty_total,
    }


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    print_system_info()

    jobs = list(TOPICS.items())
    workers = max(1, min(NUM_WORKERS, len(jobs)))

    logging.info(f"Device: {DEVICE}")
    logging.info(f"Generator: {GENERATOR_MODEL_NAME}")
    logging.info(f"Output: {OUTPUT_DIR}")
    logging.info(f"Rows per topic: {ROWS_PER_TOPIC}")
    logging.info(f"Separator: {SEPARATOR}")
    logging.info(f"Manual workers: {workers}")

    if workers <= 1:
        init_worker()
        results = [generate_topic(job) for job in jobs]
    else:
        ctx = mp.get_context("spawn")

        with ctx.Pool(
            processes=workers,
            initializer=init_worker,
        ) as pool:
            results = pool.map(generate_topic, jobs)

    logging.info("")
    logging.info("=" * 80)
    logging.info("ALL TOPICS DONE")
    logging.info("=" * 80)

    for result in results:
        logging.info(result)


if __name__ == "__main__":
    try:
        mp.freeze_support()
        main()
    except Exception:
        logging.error("SCRIPT CRASHED")
        logging.error(traceback.format_exc())
        raise
