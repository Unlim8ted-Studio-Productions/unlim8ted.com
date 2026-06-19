import json
import re
import time
import random
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# HARDCODED PATHS
# ============================================================

QA_PATH = Path("assets/data/Smart-Meatball-Data.jsonl")
FRAGMENTS_PATH = Path("assets/data/fragments.jsonl")
BRIDGES_PATH = Path("assets/data/fragment_bridges.jsonl")
OUT_PATH = Path("assets/data/fragment_training_verified.jsonl")

CACHE_DIR = Path("assets/models/fragment-label-cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FRAGMENT_EMB_CACHE = CACHE_DIR / "semantic_fragment_embeddings.npz"
QA_EMB_CACHE = CACHE_DIR / "qa_embeddings.npz"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LABELER_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# If Windows pagefile/model loading breaks, use:
# LABELER_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

SEED = 42
BATCH_SIZE = 8

TOP_K_ANSWER = 14
TOP_K_QUESTION = 5
MAX_SEMANTIC_CANDIDATES = 14

# More bridge variety shown to the labeler.
MAX_BRIDGE_CANDIDATES_PER_ROW = 12

# More transitions allowed in targets.
MIN_TARGET_FRAGMENTS = 1
MAX_TARGET_FRAGMENTS = 5
PREFER_BRIDGE_RATE = 0.55

MAX_NEW_TOKENS_LABEL = 110
MAX_NEW_TOKENS_REVIEW = 80

# Reviewer now checks: answers question + grammatical + coherent.
# Not strict official-answer matching.
REVIEW_KEEP_SCORE = 65.0

TARGET_ROWS_PER_MIN = 25.0

DEBUG_REJECTIONS = True

# If labeler outputs junk, still let reviewer judge top retrieved answer.
FALLBACK_TO_TOP_SEMANTIC = True

# Adds candidate diversity so every row is not identical.
SEMANTIC_RANDOM_POOL = 22
SEMANTIC_RANDOM_EXTRA = 3


# ============================================================
# TEXT / JSONL
# ============================================================


def clean_text(x):
    x = str(x or "").strip()
    x = re.sub(r"\s+", " ", x)
    return x


def normalize_key(x):
    x = clean_text(x).lower()
    x = re.sub(r"[^\w\s]", "", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def truncate(text, n):
    text = clean_text(text)
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "..."


def load_jsonl(path):
    rows = []

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8", errors="replace") as f:
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


def append_jsonl(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def get_row_text(row):
    for key in ["text", "content", "answer", "fragment", "value"]:
        value = row.get(key)

        if isinstance(value, str) and value.strip():
            return clean_text(value)

    return ""


# ============================================================
# LOAD SOURCE DATA
# ============================================================


def load_qa():
    raw = load_jsonl(QA_PATH)
    out = []
    seen = set()

    for row in raw:
        q = clean_text(row.get("question"))
        a = clean_text(row.get("answer"))

        if not q or not a:
            continue

        key = normalize_key(q)

        if key in seen:
            continue

        seen.add(key)

        out.append(
            {
                "id": clean_text(row.get("id")),
                "question": q,
                "answer": a,
                "intent": clean_text(row.get("intent")),
                "project": clean_text(row.get("project")),
                "project_key": clean_text(row.get("project_key")),
                "category": clean_text(row.get("category")),
                "tags": row.get("tags", []),
            }
        )

    return out


def load_fragments(path, kind):
    raw = load_jsonl(path)
    out = []
    seen = set()

    for row in raw:
        fid = row.get("id")
        text = get_row_text(row)

        if not isinstance(fid, str) or not fid.strip():
            continue

        if not text:
            continue

        fid = fid.strip()

        if fid in seen:
            continue

        seen.add(fid)

        out.append(
            {
                "id": fid,
                "text": text,
                "kind": kind,
                "topic": clean_text(row.get("topic")),
                "role": clean_text(row.get("role")),
                "tags": row.get("tags", []),
                "raw": row,
            }
        )

    return out


# ============================================================
# EMBEDDINGS
# ============================================================


def embed_texts(embedder, texts, batch_size=128):
    return embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    ).astype("float32")


def get_or_build_fragment_embeddings(embedder, fragments):
    ids = [f["id"] for f in fragments]

    if FRAGMENT_EMB_CACHE.exists():
        data = np.load(FRAGMENT_EMB_CACHE, allow_pickle=True)
        cached_ids = data["ids"].tolist()

        if cached_ids == ids:
            print("Loaded semantic fragment embeddings cache.")
            return data["embeddings"].astype("float32")

        print("Fragment embedding cache mismatch. Rebuilding.")

    print("Embedding semantic fragments...")
    embeddings = embed_texts(embedder, [f["text"] for f in fragments])

    np.savez_compressed(
        FRAGMENT_EMB_CACHE,
        ids=np.array(ids, dtype=object),
        embeddings=embeddings,
    )

    return embeddings


def get_or_build_qa_embeddings(embedder, qa_rows):
    ids = [row["id"] or str(i) for i, row in enumerate(qa_rows)]

    if QA_EMB_CACHE.exists():
        data = np.load(QA_EMB_CACHE, allow_pickle=True)
        cached_ids = data["ids"].tolist()

        if cached_ids == ids:
            print("Loaded Q&A embeddings cache.")
            return (
                data["question_embeddings"].astype("float32"),
                data["answer_embeddings"].astype("float32"),
            )

        print("Q&A embedding cache mismatch. Rebuilding.")

    print("Embedding Q&A questions...")
    q_embeddings = embed_texts(embedder, [x["question"] for x in qa_rows])

    print("Embedding Q&A answers...")
    a_embeddings = embed_texts(embedder, [x["answer"] for x in qa_rows])

    np.savez_compressed(
        QA_EMB_CACHE,
        ids=np.array(ids, dtype=object),
        question_embeddings=q_embeddings,
        answer_embeddings=a_embeddings,
    )

    return q_embeddings, a_embeddings


# ============================================================
# BRIDGES / TRANSITION VARIETY
# ============================================================


def bridge_topic(bridge):
    bid = bridge["id"]

    if bid.startswith("bridge_explanation_"):
        return "explanation"
    if bid.startswith("bridge_reasoning_"):
        return "reasoning"
    if bid.startswith("bridge_uncertainty_"):
        return "uncertainty"
    if bid.startswith("bridge_question_"):
        return "question"
    if bid.startswith("bridge_emotion_"):
        return "emotion"
    if bid.startswith("bridge_memory_"):
        return "memory"
    if bid.startswith("bridge_relationship_"):
        return "relationship"
    if bid.startswith("bridge_smalltalk_"):
        return "smalltalk"

    return clean_text(bridge.get("topic")) or "general"


def group_bridges(bridges):
    groups = {}

    for bridge in bridges:
        topic = bridge_topic(bridge)
        groups.setdefault(topic, []).append(bridge)

    return groups


def bridge_prompt_pool_for_row(bridge_groups, row_index):
    """
    Rotates bridges so targets use more varied transitions instead of the same few bridges.
    """
    rng = random.Random(SEED + row_index * 17)

    priority = [
        "explanation",
        "reasoning",
        "smalltalk",
        "emotion",
        "relationship",
        "memory",
        "uncertainty",
        "question",
        "general",
    ]

    selected = []
    seen = set()

    # Always try to include one explanation/reasoning style transition.
    for topic in ["explanation", "reasoning"]:
        group = bridge_groups.get(topic, [])
        if group:
            b = group[row_index % len(group)]
            if b["id"] not in seen:
                selected.append(b)
                seen.add(b["id"])

    # Then rotate/sample the rest.
    topic_order = priority[:]
    rng.shuffle(topic_order)

    for topic in topic_order:
        group = bridge_groups.get(topic, [])
        if not group:
            continue

        picks = group[:]
        rng.shuffle(picks)

        for b in picks:
            if b["id"] in seen:
                continue

            selected.append(b)
            seen.add(b["id"])

            if len(selected) >= MAX_BRIDGE_CANDIDATES_PER_ROW:
                return selected

    return selected


# ============================================================
# RETRIEVAL
# ============================================================


def retrieve_candidates(
    q_emb, a_emb, semantic_embeddings, semantic_fragments, row_index
):
    q_scores = semantic_embeddings @ q_emb
    a_scores = semantic_embeddings @ a_emb

    q_top = np.argsort(-q_scores)[:TOP_K_QUESTION]
    a_top = np.argsort(-a_scores)[:TOP_K_ANSWER]

    mixed_scores = (a_scores * 0.80) + (q_scores * 0.20)
    random_pool = np.argsort(-mixed_scores)[:SEMANTIC_RANDOM_POOL]

    rng = random.Random(SEED + row_index * 31)
    random_extra = list(random_pool)
    rng.shuffle(random_extra)
    random_extra = random_extra[:SEMANTIC_RANDOM_EXTRA]

    merged = []
    seen = set()

    for idx in list(a_top) + list(q_top) + list(random_extra):
        idx = int(idx)

        if idx in seen:
            continue

        seen.add(idx)

        merged.append(
            {
                **semantic_fragments[idx],
                "score": float(mixed_scores[idx]),
                "answer_score": float(a_scores[idx]),
                "question_score": float(q_scores[idx]),
            }
        )

    merged.sort(key=lambda x: x["score"], reverse=True)

    return merged[:MAX_SEMANTIC_CANDIDATES]


# ============================================================
# ASSEMBLY / JSON
# ============================================================


def assemble(ids, id_to_text):
    parts = []

    for fid in ids:
        text = clean_text(id_to_text.get(fid, ""))

        if text:
            parts.append(text)

    out = ""

    for part in parts:
        if not out:
            out = part
        elif out.endswith((" ", "\n")):
            out += part
        elif part.startswith((".", ",", "!", "?", ";", ":")):
            out += part
        else:
            out += " " + part

    return out


def extract_json_object(text):
    text = str(text or "").strip()
    text = text.replace("```json", "```")

    if "```" in text:
        for part in text.split("```"):
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
    in_str = False
    esc = False

    for i in range(start, len(text)):
        ch = text[i]

        if esc:
            esc = False
            continue

        if ch == "\\":
            esc = True
            continue

        if ch == '"':
            in_str = not in_str
            continue

        if in_str:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1

            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return None

    return None


def fallback_extract_codes(raw):
    raw = str(raw or "")
    found = re.findall(r"\b[SB]\s*[-_]?\s*\d+\b", raw, flags=re.I)
    out = []

    for x in found:
        x = re.sub(r"\s+", "", x.upper())
        x = x.replace("-", "").replace("_", "")

        if x not in out:
            out.append(x)

    return out


# ============================================================
# CODES / TARGET PARSING
# ============================================================


def make_candidate_code_maps(semantic_candidates, bridge_candidates):
    code_to_id = {}
    id_to_code = {}

    for i, frag in enumerate(semantic_candidates, 1):
        code = f"S{i}"
        code_to_id[code] = frag["id"]
        id_to_code[frag["id"]] = code

    for i, bridge in enumerate(bridge_candidates, 1):
        code = f"B{i}"
        code_to_id[code] = bridge["id"]
        id_to_code[bridge["id"]] = code

    return code_to_id, id_to_code


def parse_label_ids(label_result, code_to_id, allowed_real_ids):
    raw = ""
    obj = None

    if isinstance(label_result, dict):
        raw = label_result.get("raw", "")
        obj = label_result.get("json")
    else:
        obj = label_result

    raw_codes = []
    raw_real_ids = []

    if isinstance(obj, dict):
        frags = obj.get("fragments", [])

        if isinstance(frags, list):
            for x in frags:
                if not isinstance(x, str):
                    continue

                x = x.strip()

                normalized_code = re.sub(r"[^A-Za-z0-9]", "", x).upper()

                if normalized_code in code_to_id:
                    raw_codes.append(normalized_code)
                    continue

                if x in allowed_real_ids:
                    raw_real_ids.append(x)
                    continue

    if not raw_codes and not raw_real_ids:
        raw_codes = fallback_extract_codes(raw)

    # Important: if JSON is cut off and it output real IDs, still recover them.
    for fid in allowed_real_ids:
        if fid in raw and fid not in raw_real_ids:
            raw_real_ids.append(fid)

    ids = []

    for code in raw_codes:
        code = re.sub(r"[^A-Za-z0-9]", "", code).upper()
        fid = code_to_id.get(code)

        if fid and fid not in ids:
            ids.append(fid)

        if len(ids) >= MAX_TARGET_FRAGMENTS:
            break

    for fid in raw_real_ids:
        if fid not in ids:
            ids.append(fid)

        if len(ids) >= MAX_TARGET_FRAGMENTS:
            break

    return ids, raw, obj


# ============================================================
# SHAPE / ROW
# ============================================================


def infer_shape(question, selected_ids):
    q = question.lower()

    if any(
        x in q for x in ["hi", "hello", "hey", "yo", "how are you", "how was your day"]
    ):
        return "smalltalk"

    if not selected_ids:
        return "unknown_or_not_enough_info"

    if len(selected_ids) >= 3:
        return "explanation"

    return "direct_answer"


def build_row_from_ids(
    qa,
    selected_ids,
    semantic_candidates,
    all_bridge_ids,
    id_to_text,
    label_raw,
    label_json,
):
    if not selected_ids:
        return None, "empty_target"

    allowed_semantic_ids = {x["id"] for x in semantic_candidates}
    has_semantic = any(fid in allowed_semantic_ids for fid in selected_ids)

    shape = infer_shape(qa["question"], selected_ids)

    if not has_semantic and shape not in {
        "unknown_or_not_enough_info",
        "smalltalk",
        "clarification",
    }:
        return None, "bridge_only_answer"

    preview = assemble(selected_ids, id_to_text)

    if not preview:
        return None, "empty_preview"

    if len(preview.split()) < 3:
        return None, "too_short_preview"

    semantic_candidate_ids = [x["id"] for x in semantic_candidates]
    candidate_fragment_ids = semantic_candidate_ids + list(all_bridge_ids)

    row = {
        "input": qa["question"],
        "history": [],
        "candidate_fragment_ids": candidate_fragment_ids,
        "target": {
            "shape": shape,
            "fragments": selected_ids,
        },
        "assembled_preview": preview,
        "official_answer": qa["answer"],
        "label_debug": {
            "source_id": qa.get("id", ""),
            "labeler_raw": str(label_raw or "")[:500],
            "labeler_json": label_json,
            "semantic_candidates": [
                {
                    "id": x["id"],
                    "score": round(float(x["score"]), 4),
                    "answer_score": round(float(x["answer_score"]), 4),
                    "question_score": round(float(x["question_score"]), 4),
                }
                for x in semantic_candidates
            ],
        },
    }

    return row, ""


# ============================================================
# PROMPTS
# ============================================================


def build_label_prompt(qa, semantic_candidates, bridge_candidates, row_index):
    sem_lines = []
    for i, x in enumerate(semantic_candidates, 1):
        sem_lines.append(f"S{i}: {truncate(x['text'], 115)}")

    bridge_lines = []
    for i, x in enumerate(bridge_candidates, 1):
        bridge_lines.append(f"B{i}: {truncate(x['text'], 75)}")

    # Encourage variety: some rows get transitions, some direct.
    wants_bridge = (row_index % 100) < int(PREFER_BRIDGE_RATE * 100)

    if wants_bridge:
        bridge_rule = """
- Use 1 bridge fragment when it makes the answer sound more natural.
- Bridge fragments should help transition, clarify, or add personality.
- Good pattern: one bridge + one or two semantic fragments.
""".strip()
    else:
        bridge_rule = """
- Use bridge fragments only if they genuinely improve the answer.
- A direct semantic-only answer is allowed.
""".strip()

    return f"""
Choose numbered fragments that create a good chatbot answer.

QUESTION:
{qa["question"]}

OFFICIAL ANSWER / TRUTH SOURCE:
{qa["answer"]}

SEMANTIC FRAGMENTS:
{chr(10).join(sem_lines)}

BRIDGE / TRANSITION FRAGMENTS:
{chr(10).join(bridge_lines)}

Rules:
- Return ONLY JSON.
- Use codes like S1, S2, B1.
- Choose {MIN_TARGET_FRAGMENTS} to {MAX_TARGET_FRAGMENTS} fragments.
- The assembled answer should answer the user's question.
- It does NOT need to copy the official answer exactly.
- It should be grammatical and sound natural.
- Prefer semantic fragments for facts.
{bridge_rule}
- Do not use full fragment IDs.
- Do not explain.
- Do not use markdown.

Format:
{{"fragments":["B1","S1","S2"]}}
""".strip()


def build_review_prompt(question, official_answer, assembled_preview):
    return f"""
You are reviewing a fragment-assembled chatbot answer.

QUESTION:
{question}

TRUTH SOURCE:
{official_answer}

ASSEMBLED ANSWER:
{assembled_preview}

Judge the ASSEMBLED ANSWER only.

Keep it if:
- it answers the QUESTION,
- it is grammatical enough,
- it is coherent,
- it does not obviously contradict the truth source,
- it does not have to match the truth source wording exactly.

Reject it if:
- it does not answer the question,
- it is nonsense,
- it is ungrammatical/broken,
- it contradicts the truth source,
- it is only a vague transition with no real answer.

Score:
100 = excellent answer
80 = good answer
65 = acceptable answer
50 = weak but related
30 = bad
0 = unrelated/nonsense

Return ONLY JSON:
{{"score":80,"keep":true}}
""".strip()


# ============================================================
# LOCAL LLM
# ============================================================


def load_labeler():
    print("Loading labeler:", LABELER_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(
        LABELER_MODEL,
        trust_remote_code=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        LABELER_MODEL,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    if not torch.cuda.is_available():
        model.to("cpu")

    model.eval()

    print("Labeler loaded.")
    return tokenizer, model


def render_prompt(tokenizer, prompt, mode):
    if mode == "review":
        system = "You are a strict answer-quality reviewer. Return only valid JSON."
    else:
        system = "You are a strict JSON fragment selector. Return only valid JSON."

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


@torch.inference_mode()
def generate_json_batch(tokenizer, model, prompts, max_new_tokens, mode):
    rendered = [render_prompt(tokenizer, p, mode=mode) for p in prompts]

    inputs = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    input_len = inputs["input_ids"].shape[1]
    out = []

    for i in range(len(prompts)):
        new_ids = output_ids[i][input_len:]
        raw = tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        out.append(
            {
                "json": extract_json_object(raw),
                "raw": raw,
            }
        )

    return out


def label_batch(tokenizer, model, prompts):
    return generate_json_batch(
        tokenizer=tokenizer,
        model=model,
        prompts=prompts,
        max_new_tokens=MAX_NEW_TOKENS_LABEL,
        mode="label",
    )


def parse_review_score(result):
    raw = ""
    obj = None

    if isinstance(result, dict):
        raw = result.get("raw", "")
        obj = result.get("json")

    if isinstance(obj, dict):
        score = obj.get("score", 0)
        keep = obj.get("keep", False)

        try:
            score = float(score)
        except Exception:
            score = 0.0

        score = max(0.0, min(100.0, score))

        if isinstance(keep, str):
            keep = keep.strip().lower() in {"true", "yes", "1", "keep"}

        return score, bool(keep), raw, obj

    nums = re.findall(r"-?\d+(?:\.\d+)?", str(raw or ""))

    if not nums:
        return 0.0, False, raw, obj

    score = float(nums[0])
    score = max(0.0, min(100.0, score))

    return score, score >= REVIEW_KEEP_SCORE, raw, obj


def review_batch(tokenizer, model, review_items):
    prompts = [
        build_review_prompt(
            question=item["question"],
            official_answer=item["official_answer"],
            assembled_preview=item["assembled_preview"],
        )
        for item in review_items
    ]

    outputs = generate_json_batch(
        tokenizer=tokenizer,
        model=model,
        prompts=prompts,
        max_new_tokens=MAX_NEW_TOKENS_REVIEW,
        mode="review",
    )

    parsed = []

    for result in outputs:
        score, keep, raw, obj = parse_review_score(result)

        parsed.append(
            {
                "score": score,
                "keep": keep,
                "raw": raw,
                "json": obj,
            }
        )

    return parsed


# ============================================================
# BATCH PROCESSING
# ============================================================


def process_batch(
    batch,
    tokenizer,
    model,
    all_bridge_ids,
    id_to_text,
    existing_inputs,
    counters,
    start_time,
):
    prompts = [x["prompt"] for x in batch]

    try:
        label_outputs = label_batch(tokenizer, model, prompts)
    except Exception as e:
        counters["model_errors"] += len(batch)
        print("LABEL_BATCH_ERROR:", repr(e)[:1000], flush=True)
        return

    review_candidates = []

    for item, label_result in zip(batch, label_outputs):
        counters["attempted"] += 1

        code_to_id = item["code_to_id"]
        allowed_real_ids = set(code_to_id.values())

        selected_ids, raw, obj = parse_label_ids(
            label_result=label_result,
            code_to_id=code_to_id,
            allowed_real_ids=allowed_real_ids,
        )

        if (
            not selected_ids
            and FALLBACK_TO_TOP_SEMANTIC
            and item["semantic_candidates"]
        ):
            selected_ids = [item["semantic_candidates"][0]["id"]]
            raw = f"FALLBACK_TOP_SEMANTIC because labeler output was: {raw}"

        out_row, err = build_row_from_ids(
            qa=item["qa"],
            selected_ids=selected_ids,
            semantic_candidates=item["semantic_candidates"],
            all_bridge_ids=all_bridge_ids,
            id_to_text=id_to_text,
            label_raw=raw,
            label_json=obj,
        )

        if out_row is None:
            counters["rejected"] += 1

            if DEBUG_REJECTIONS and (
                counters["rejected"] <= 20 or counters["rejected"] % 100 == 0
            ):
                print()
                print("REJECTED BEFORE REVIEW")
                print("reason:", err)
                print("question:", item["qa"]["question"][:220])
                print("official:", item["qa"]["answer"][:260])
                print("raw labeler output:", str(raw)[:700])
                print("parsed:", obj)
                print("selected_ids:", selected_ids)
                print("candidate codes:", list(code_to_id.keys())[:20])

            continue

        review_candidates.append(
            {
                "row": out_row,
                "question": item["qa"]["question"],
                "official_answer": item["qa"]["answer"],
                "assembled_preview": out_row["assembled_preview"],
            }
        )

    if not review_candidates:
        return

    try:
        reviews = review_batch(tokenizer, model, review_candidates)
    except Exception as e:
        print("REVIEW_BATCH_ERROR:", repr(e)[:1000], flush=True)
        reviews = [
            {
                "score": 0.0,
                "keep": False,
                "raw": "",
                "json": None,
            }
            for _ in review_candidates
        ]

    for item, review in zip(review_candidates, reviews):
        out_row = item["row"]

        score = float(review["score"])
        keep = bool(review["keep"]) or score >= REVIEW_KEEP_SCORE

        out_row["label_debug"]["review_score"] = round(score, 2)
        out_row["label_debug"]["review_keep"] = keep
        out_row["label_debug"]["review_raw"] = str(review.get("raw", ""))[:500]
        out_row["label_debug"]["review_json"] = review.get("json")

        if not keep:
            counters["rejected"] += 1

            if DEBUG_REJECTIONS and (
                counters["rejected"] <= 20 or counters["rejected"] % 100 == 0
            ):
                print()
                print("REJECTED BY REVIEWER")
                print("score:", score)
                print("question:", out_row["input"][:220])
                print("truth source:", out_row["official_answer"][:260])
                print("assembled:", out_row["assembled_preview"][:260])
                print("review raw:", str(review.get("raw", ""))[:500])

            continue

        append_jsonl(OUT_PATH, out_row)
        existing_inputs.add(out_row["input"])

        counters["written"] += 1
        counters["accepted"] += 1

        if counters["accepted"] <= 10 or counters["accepted"] % 25 == 0:
            elapsed = max(time.time() - start_time, 0.001)
            rpm = counters["accepted"] / elapsed * 60.0

            print()
            print(
                f"ACCEPTED {counters['accepted']} / attempted {counters['attempted']}"
            )
            print("rows/min:", round(rpm, 2))
            print("review score:", score)
            print("input:", out_row["input"][:180])
            print("target:", out_row["target"])
            print("preview:", out_row["assembled_preview"][:260])
            print("truth source:", out_row["official_answer"][:260])


# ============================================================
# MAIN
# ============================================================


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Hardcoded input paths:")
    print("QA:", QA_PATH)
    print("Fragments:", FRAGMENTS_PATH)
    print("Bridges:", BRIDGES_PATH)
    print("Out:", OUT_PATH)
    print()

    qa_rows = load_qa()
    semantic_fragments = load_fragments(FRAGMENTS_PATH, "semantic")
    bridge_fragments = load_fragments(BRIDGES_PATH, "bridge")

    print("Q&A rows:", len(qa_rows))
    print("Semantic fragments:", len(semantic_fragments))
    print("Bridge fragments:", len(bridge_fragments))

    all_bridge_ids = [b["id"] for b in bridge_fragments]
    bridge_groups = group_bridges(bridge_fragments)

    print("All bridge IDs included in training candidates:", len(all_bridge_ids))
    print("Bridge prompt candidates per row:", MAX_BRIDGE_CANDIDATES_PER_ROW)
    print()

    id_to_text = {}

    for frag in semantic_fragments + bridge_fragments:
        id_to_text[frag["id"]] = frag["text"]

    print("Loading embedder:", EMBEDDING_MODEL)
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    semantic_embeddings = get_or_build_fragment_embeddings(embedder, semantic_fragments)
    question_embeddings, answer_embeddings = get_or_build_qa_embeddings(
        embedder, qa_rows
    )

    tokenizer, model = load_labeler()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing_inputs = set()
    existing_count = 0

    if OUT_PATH.exists():
        for row in load_jsonl(OUT_PATH):
            inp = clean_text(row.get("input"))

            if inp:
                existing_inputs.add(inp)

            existing_count += 1

    counters = {
        "written": existing_count,
        "attempted": 0,
        "accepted": 0,
        "rejected": 0,
        "model_errors": 0,
    }

    print("Existing output rows:", existing_count)
    print("Continuing without duplicating existing inputs.")
    print()

    start = time.time()
    buffer = []

    for i, qa in enumerate(qa_rows):
        if qa["question"] in existing_inputs:
            continue

        semantic_candidates = retrieve_candidates(
            q_emb=question_embeddings[i],
            a_emb=answer_embeddings[i],
            semantic_embeddings=semantic_embeddings,
            semantic_fragments=semantic_fragments,
            row_index=i,
        )

        bridge_candidates = bridge_prompt_pool_for_row(
            bridge_groups=bridge_groups,
            row_index=i,
        )

        code_to_id, _ = make_candidate_code_maps(
            semantic_candidates=semantic_candidates,
            bridge_candidates=bridge_candidates,
        )

        prompt = build_label_prompt(
            qa=qa,
            semantic_candidates=semantic_candidates,
            bridge_candidates=bridge_candidates,
            row_index=i,
        )

        buffer.append(
            {
                "row_index": i,
                "qa": qa,
                "semantic_candidates": semantic_candidates,
                "bridge_candidates": bridge_candidates,
                "prompt": prompt,
                "code_to_id": code_to_id,
            }
        )

        if len(buffer) < BATCH_SIZE:
            continue

        process_batch(
            batch=buffer,
            tokenizer=tokenizer,
            model=model,
            all_bridge_ids=all_bridge_ids,
            id_to_text=id_to_text,
            existing_inputs=existing_inputs,
            counters=counters,
            start_time=start,
        )

        buffer = []

        elapsed = max(time.time() - start, 0.001)
        rpm = counters["accepted"] / elapsed * 60.0

        print(
            f"PROGRESS accepted={counters['accepted']} "
            f"rejected={counters['rejected']} "
            f"attempted={counters['attempted']} "
            f"written_total={counters['written']} "
            f"model_errors={counters['model_errors']} "
            f"rows/min={rpm:.2f} "
            f"target={TARGET_ROWS_PER_MIN}",
            flush=True,
        )

    if buffer:
        process_batch(
            batch=buffer,
            tokenizer=tokenizer,
            model=model,
            all_bridge_ids=all_bridge_ids,
            id_to_text=id_to_text,
            existing_inputs=existing_inputs,
            counters=counters,
            start_time=start,
        )

    elapsed = max(time.time() - start, 0.001)
    rpm = counters["accepted"] / elapsed * 60.0

    print()
    print("DONE")
    print("accepted:", counters["accepted"])
    print("rejected:", counters["rejected"])
    print("attempted:", counters["attempted"])
    print("written_total:", counters["written"])
    print("model_errors:", counters["model_errors"])
    print("rows/min:", round(rpm, 2))
    print("target rows/min:", TARGET_ROWS_PER_MIN)
    print("output:", OUT_PATH)

    if rpm < TARGET_ROWS_PER_MIN:
        print()
        print("Below 25/min.")
        print("Try BATCH_SIZE = 12 or 16.")
        print("Try MAX_NEW_TOKENS_LABEL = 90.")
        print("Try REVIEW_KEEP_SCORE = 55.0.")
        print("Try MAX_BRIDGE_CANDIDATES_PER_ROW = 8.")


if __name__ == "__main__":
    main()
