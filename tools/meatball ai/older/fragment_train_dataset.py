import argparse
import json
import random
import re
import urllib.request
import urllib.error
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from sentence_transformers import SentenceTransformer


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                pass
    return rows


def load_embeddings(path, quantized=False):
    emb = np.load(path)
    if quantized or emb.dtype == np.int8:
        emb = emb.astype("float32") / 127.0
    else:
        emb = emb.astype("float32")

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return emb / norms


def append_jsonl(path, obj):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def clean_question(q):
    q = str(q or "").strip()
    q = re.sub(r"\s+", " ", q)
    return q


def dedupe_questions(questions):
    seen = set()
    out = []

    for q in questions:
        q = clean_question(q)
        if not q:
            continue

        key = q.lower()
        key = re.sub(r"[^\w\s]", "", key)
        key = re.sub(r"\s+", " ", key).strip()

        if key in seen:
            continue

        seen.add(key)
        out.append(q)

    return out


def load_seed_questions(path):
    rows = load_jsonl(path)
    questions = []

    for row in rows:
        q = clean_question(row.get("question"))
        if q:
            questions.append(q)

    questions = dedupe_questions(questions)

    if not questions:
        raise ValueError("No questions found.")

    return questions


SLANG_MUTATIONS = [
    lambda q: q,
    lambda q: q.lower(),
    lambda q: q + "?",
    lambda q: q + " bro",
    lambda q: q + " pls",
    lambda q: q + " in simple words",
    lambda q: q + " in sauce terms",
    lambda q: q.replace("what is", "whats"),
    lambda q: q.replace("What is", "Whats"),
    lambda q: q.replace("you", "u"),
    lambda q: q.replace("your", "ur"),
    lambda q: q.replace("the", "da"),
    lambda q: q.replace("The", "Da"),
    lambda q: q.replace("about", "abt"),
    lambda q: q.replace("Unlim8ted", "unlim8ted"),
    lambda q: q.replace("Unlim8ted", "unlimited"),
    lambda q: q.replace("The Glitch", "the glitch"),
    lambda q: q.replace("The Life of a Meatball", "the meatball movie"),
    lambda q: q.replace("what", "wat"),
    lambda q: q.replace("What", "Wat"),
]


FOLLOWUP_FORMS = [
    "what does that mean",
    "explain that more",
    "why",
    "why though",
    "how",
    "how does that work",
    "say that simpler",
    "give me the short version",
    "no i mean the film",
    "no i mean the project",
    "is that official",
    "are you sure",
    "what is the lore",
    "what is the point",
]


UNKNOWN_QUESTIONS = [
    "what is the exact private budget",
    "how many employees are there right now",
    "what is the official private address",
    "what is the secret password",
    "what are the private investor names",
    "what is the exact unreleased revenue",
    "what is the private payroll",
    "what awards did it win if the archive does not say",
    "what is the exact release date if it is not in the data",
    "what is the internal legal document number",
    "what is the hidden backend key",
    "what is the exact box office number",
    "what is the exact production budget",
]


def slangify_question(q):
    q = clean_question(q)
    q = clean_question(random.choice(SLANG_MUTATIONS)(q))

    if random.random() < 0.08:
        q = q.replace("?", "")
    if random.random() < 0.05:
        q = q + "????"
    if random.random() < 0.04:
        q = q + " what da sauce mean"

    return clean_question(q)


def make_question_pool(seed_questions, copies_per_seed=4):
    pool = []

    for q in seed_questions:
        pool.append(q)
        for _ in range(copies_per_seed):
            pool.append(slangify_question(q))

    for _ in range(max(30, len(pool) // 6)):
        pool.append(slangify_question(random.choice(UNKNOWN_QUESTIONS)))

    for _ in range(max(30, len(pool) // 6)):
        pool.append(slangify_question(random.choice(FOLLOWUP_FORMS)))

    pool = dedupe_questions(pool)
    random.shuffle(pool)
    return pool


def make_history(question, seed_questions):
    q = question.lower()

    if any(x in q for x in ["that", "this", "why", "how", "no i mean", "more"]):
        base = random.choice(seed_questions)
        return [
            {
                "user": base,
                "assistant": "A previous answer used selected fragments from the project knowledge base.",
            }
        ]

    if random.random() < 0.12:
        base = random.choice(seed_questions)
        return [
            {
                "user": base,
                "assistant": "A previous answer used selected fragments from the project knowledge base.",
            }
        ]

    return []


def cosine_top_k(query, embedder, fragments, embeddings, top_k):
    q = embedder.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")[0]

    scores = embeddings @ q
    idxs = np.argsort(-scores)[:top_k]

    out = []
    for i in idxs:
        f = fragments[int(i)]
        out.append(
            {
                "id": f["id"],
                "score": float(scores[int(i)]),
                "topic": f.get("topic", "general"),
                "role": f.get("role", "fact"),
                "text": f.get("text", ""),
            }
        )

    return out


def compact_fragment(f):
    text = str(f.get("text", ""))
    if len(text) > 260:
        text = text[:260] + "..."

    return {
        "id": f["id"],
        "topic": f.get("topic", "general"),
        "role": f.get("role", "fact"),
        "score": round(float(f.get("score", 0)), 4),
        "text": text,
    }


def compact_bridge(b):
    text = str(b.get("text", ""))
    if len(text) > 160:
        text = text[:160] + "..."

    return {
        "id": b["id"],
        "topic": b.get("topic", "general"),
        "role": b.get("role", "bridge"),
        "text": text,
    }


def bridge_pool(index, max_per_topic=20):
    bridges = index.get("bridges", [])
    by_topic = {}

    for b in bridges:
        topic = b.get("topic", "general")
        by_topic.setdefault(topic, []).append(b)

    selected = []
    seen = set()

    priority = [
        "explanation",
        "reasoning",
        "uncertainty",
        "question",
        "emotion",
        "memory",
        "relationship",
    ]

    for topic in priority:
        for b in by_topic.get(topic, [])[:max_per_topic]:
            bid = b.get("id")
            if bid and bid not in seen:
                seen.add(bid)
                selected.append(b)

    if not selected:
        selected = bridges[:140]

    return selected


def ollama_generate(model, prompt, url, timeout):
    print("PROMPT_CHARS:", len(prompt))

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.05,
            "top_p": 0.8,
            "num_ctx": 65536,
            "num_predict": 2048,
            "repeat_penalty": 1.05,
        },
    }

    data = json.dumps(payload).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body[:3000]}")
    except Exception as e:
        raise RuntimeError(f"{type(e).__name__}: {e}")

    try:
        wrapper = json.loads(raw)
    except Exception:
        raise RuntimeError(f"Ollama returned invalid wrapper JSON: {raw[:3000]}")

    if "error" in wrapper:
        raise RuntimeError(f"Ollama error: {wrapper['error']}")

    response = wrapper.get("response", "")
    done_reason = wrapper.get("done_reason", "")

    if not response:
        raise RuntimeError(
            "Ollama response was empty. "
            f"done_reason={done_reason!r}. "
            f"prompt_chars={len(prompt)}. "
            f"Raw first 3000 chars: {raw[:3000]}"
        )

    parsed = extract_json_object(response)
    if parsed is None:
        raise RuntimeError(
            f"Invalid model JSON response. "
            f"done_reason={done_reason!r}. "
            f"Response first 3000 chars: {response[:3000]}"
        )

    return parsed


def extract_json_object(text):
    text = str(text or "").strip()

    text = text.replace("```json", "```")

    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("{") and part.endswith("}"):
                try:
                    return json.loads(part)
                except Exception:
                    pass

    try:
        return json.loads(text)
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if escape:
            escape = False
            continue

        if ch == "\\":
            escape = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None

    return None


ALLOWED_SHAPES = {
    "direct_answer",
    "short_answer",
    "explanation",
    "clarification",
    "correction",
    "followup",
    "unknown_or_not_enough_info",
    "smalltalk",
}


def build_prompt(items, bridges):
    compact_bridges = [compact_bridge(b) for b in bridges]
    bridge_ids = [b["id"] for b in compact_bridges]

    rows = []
    for item in items:
        rows.append(
            {
                "row_id": item["row_id"],
                "input": item["input"],
                "history": item["history"],
                "semantic_candidates": [
                    compact_fragment(f) for f in item["semantic_candidates"]
                ],
                "allowed_bridge_ids": bridge_ids,
            }
        )

    return f"""
You are creating supervised labels for a fragment-composer chatbot.

Critical:
- Do not mention real person names.
- Do not invent facts.
- Do not invent IDs.
- The chatbot only selects IDs.

Select only IDs from:
1. semantic_candidates for the row
2. global_bridge_candidates

Allowed shapes:
direct_answer, short_answer, explanation, clarification, correction, followup, unknown_or_not_enough_info, smalltalk

Rules:
1. Prefer semantic fragment IDs when they actually answer the input.
2. Use bridge IDs to connect, clarify, express uncertainty, or ask a question.
3. If semantic candidates do not answer the input, use shape unknown_or_not_enough_info.
4. For unknown answers, use uncertainty/question/explanation bridges.
5. Keep each fragment sequence 2 to 7 IDs.
6. Output valid JSON only.

Global bridge candidates:
{json.dumps(compact_bridges, ensure_ascii=False)}

Rows:
{json.dumps(rows, ensure_ascii=False)}

Return exactly:
{{
  "rows": [
    {{
      "row_id": "row id here",
      "shape": "direct_answer",
      "fragments": ["id_1", "id_2"]
    }}
  ]
}}
""".strip()


def validate_result(result, items, bridges):
    if not isinstance(result, dict):
        return [], "result_not_dict"

    rows = result.get("rows")
    if not isinstance(rows, list):
        return [], "missing_rows_list"

    item_by_id = {x["row_id"]: x for x in items}
    bridge_ids = {b["id"] for b in bridges}

    valid = []
    errors = []

    for r in rows:
        row_id = r.get("row_id")
        if row_id not in item_by_id:
            errors.append(f"bad_row_id:{row_id}")
            continue

        shape = r.get("shape", "explanation")
        if shape not in ALLOWED_SHAPES:
            shape = "explanation"

        item = item_by_id[row_id]
        semantic_ids = {f["id"] for f in item["semantic_candidates"]}
        allowed = semantic_ids | bridge_ids

        selected = r.get("fragments", [])
        cleaned = []

        if isinstance(selected, list):
            for fid in selected:
                if isinstance(fid, str) and fid in allowed and fid not in cleaned:
                    cleaned.append(fid)

        if not cleaned:
            errors.append(f"no_valid_ids:{row_id}")
            continue

        cleaned = cleaned[:8]

        valid.append(
            {
                "input": item["input"],
                "history": item["history"],
                "retrieved_fragment_ids": [f["id"] for f in item["semantic_candidates"]]
                + [b["id"] for b in bridges],
                "semantic_fragment_ids": [f["id"] for f in item["semantic_candidates"]],
                "bridge_fragment_ids": [b["id"] for b in bridges],
                "retrieval_scores": {
                    f["id"]: round(float(f.get("score", 0)), 5)
                    for f in item["semantic_candidates"]
                },
                "target": {
                    "shape": shape,
                    "fragments": cleaned,
                },
                "dedupe_key": item["dedupe_key"],
            }
        )

    return valid, ";".join(errors[:10])


def make_items(batch_questions, seed_questions, embedder, index, embeddings, top_k):
    fragments = index["fragments"]
    items = []

    for q in batch_questions:
        history = make_history(q, seed_questions)
        semantic = cosine_top_k(q, embedder, fragments, embeddings, top_k)

        key = json.dumps(
            {
                "input": q,
                "history": history,
                "semantic": [x["id"] for x in semantic],
            },
            sort_keys=True,
        )

        row_id = "row_" + str(abs(hash(key)))

        items.append(
            {
                "row_id": row_id,
                "input": q,
                "history": history,
                "semantic_candidates": semantic,
                "dedupe_key": key,
            }
        )

    return items


def process_batch(items, args, bridges):
    prompt = build_prompt(items, bridges)

    result = ollama_generate(
        model=args.model,
        prompt=prompt,
        url=args.ollama_url,
        timeout=args.timeout,
    )

    return validate_result(result, items, bridges)


def load_existing_keys(path):
    p = Path(path)
    if not p.exists():
        return set(), 0

    keys = set()
    count = 0

    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
                key = row.get("dedupe_key")
                if key:
                    keys.add(key)
                count += 1
            except Exception:
                pass

    return keys, count


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--index", default="fragment_index.json")
    parser.add_argument("--embeddings", default="fragment_embeddings.npy")
    parser.add_argument("--questions", default="assets/data/Smart-Meatball-Data.jsonl")
    parser.add_argument("--out", default="fragment_training.jsonl")

    parser.add_argument("--target-rows", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=20)

    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate")
    parser.add_argument("--timeout", type=int, default=240)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--quantized", action="store_true")
    parser.add_argument("--max-bridge-per-topic", type=int, default=20)

    args = parser.parse_args()

    index = load_json(args.index)
    embeddings = load_embeddings(args.embeddings, args.quantized)
    seed_questions = load_seed_questions(args.questions)

    if len(index["fragments"]) != embeddings.shape[0]:
        raise ValueError(
            f"Fragment count does not match embeddings rows: "
            f"{len(index['fragments'])} fragments vs {embeddings.shape[0]} embeddings"
        )

    bridges = bridge_pool(index, args.max_bridge_per_topic)

    existing_keys, existing_count = load_existing_keys(args.out)
    question_pool = make_question_pool(seed_questions)

    print("Loaded index")
    print("Fragments:", len(index["fragments"]))
    print("Embeddings:", embeddings.shape)
    print("Bridge candidates sent per prompt:", len(bridges))
    print("Seed questions:", len(seed_questions))
    print("Question pool:", len(question_pool))
    print("Existing rows:", existing_count)
    print("Target rows:", args.target_rows)
    print("Model:", args.model)
    print("Ollama URL:", args.ollama_url)
    print()

    print("Loading embedder once...")
    embedder = SentenceTransformer(index["embedding_model"])
    print("Embedder loaded.")
    print()

    written = existing_count

    stats = {
        "valid": 0,
        "llm_error": 0,
        "validation_error": 0,
        "deduped": 0,
    }

    def create_batch_items():
        batch_questions = [random.choice(question_pool) for _ in range(args.batch_size)]
        return make_items(
            batch_questions, seed_questions, embedder, index, embeddings, args.top_k
        )

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}

        def submit():
            items = create_batch_items()
            fut = ex.submit(process_batch, items, args, bridges)
            futures[fut] = True

        for _ in range(args.workers * 2):
            submit()

        while futures and written < args.target_rows:
            for fut in as_completed(list(futures.keys())):
                futures.pop(fut)

                try:
                    rows, err = fut.result()
                except Exception as e:
                    stats["llm_error"] += 1
                    print("LLM_ERROR:", repr(e)[:2000])
                    rows = []
                    err = ""

                if err:
                    stats["validation_error"] += 1
                    print("VALIDATION_WARNING:", err[:1000])

                for row in rows:
                    key = row["dedupe_key"]

                    if key in existing_keys:
                        stats["deduped"] += 1
                        continue

                    existing_keys.add(key)
                    append_jsonl(args.out, row)
                    written += 1
                    stats["valid"] += 1

                    if written % 25 == 0:
                        print("Written:", written, "Stats:", stats)

                    if written >= args.target_rows:
                        break

                if written < args.target_rows:
                    submit()

                break

    print()
    print("Final rows:", written)
    print("Stats:", stats)


if __name__ == "__main__":
    main()
