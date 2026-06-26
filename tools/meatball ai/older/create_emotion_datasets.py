# create_emotion_datasets_ai_parallel.py
# pip install torch transformers accelerate sentencepiece
#
# Fast AI-only style transfer with:
# - resume
# - multiprocessing workers
# - batch generation
# - shard autosaves
# - final merged jsonl
#
# Example:
# python create_emotion_datasets_ai_parallel.py --workers 4 --batch_size 16 --save_every 100
#
# For one GPU, usually use:
# python create_emotion_datasets_ai_parallel.py --workers 1 --batch_size 24
#
# For CPU or multiple GPUs, workers can help.

import argparse
import json
import math
import multiprocessing as mp
import os
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DEFAULT_SPECIALIZED_DIR = Path("assets/data/specialized_QA")
DEFAULT_SMART_QA_PATH = Path("tools/SmartMeatballQA.jsonl")
DEFAULT_OUT_DIR = Path("assets/data/emotions")

EMOTIONS = [
    "neutral",
    "excited",
    "confused",
    "suspicious",
    "angry",
    "sad",
    "overwhelmed",
]

AI_EMOTIONS = [
    "excited",
    "confused",
    "suspicious",
    "angry",
    "sad",
    "overwhelmed",
]


def load_jsonl(path):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception as e:
                print(f"[skip] bad json {path}:{line_num}: {e}", flush=True)
                continue

            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()

            if question and answer:
                rows.append(
                    {
                        "question": question,
                        "answer": answer,
                        "source_file": path.name,
                    }
                )

    return rows


def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_spaces(text):
    return re.sub(r"\s+", " ", str(text).replace("\n", " ")).strip()


def infer_subject(question, answer):
    text = f"{question} {answer}".lower()
    if "glitch" in text or "gltich" in text:
        return "The Glitch"
    if "timecat" in text or "time cat" in text:
        return "TimeCat"
    if "meatball ai" in text:
        return "Meatball AI"
    if "meatball" in text:
        return "Meatball"
    if "unlim8ted" in text or "unlimited" in text:
        return "Unlim8ted"
    if "dog" in text:
        return "dogs"
    if "cat" in text:
        return "cats"
    return "NONE"


def load_base_rows(specialized_dir, smart_qa_path):
    rows = []

    specialized_dir = Path(specialized_dir)
    for path in sorted(specialized_dir.glob("*.jsonl")):
        part = load_jsonl(path)
        rows.extend(part)
        print(f"{path.name}: {len(part)} rows", flush=True)

    smart_rows = load_jsonl(smart_qa_path)
    rows.extend(smart_rows)
    print(f"{Path(smart_qa_path).name}: {len(smart_rows)} rows", flush=True)

    deduped = []
    seen = set()

    for row in rows:
        question = normalize_spaces(row["question"])
        answer = normalize_spaces(row["answer"])
        key = (question.lower(), answer.lower())

        if key in seen:
            continue

        seen.add(key)
        deduped.append(
            {
                "question": question,
                "answer": answer,
                "source_file": row.get("source_file", ""),
                "subject": infer_subject(question, answer),
            }
        )

    return deduped


def emotion_rules(emotion):
    rules = {
        "excited": (
            "Rewrite in excited Meatball AI voice. Make it upbeat and energetic. "
            "Keep every fact the same. Do not add new facts. Use playful sauce/meatball language lightly."
        ),
        "confused": (
            "Rewrite in confused Meatball AI voice. Sound slightly puzzled but still helpful and factual. "
            "Keep every fact the same. Do not make the answer incorrect."
        ),
        "suspicious": (
            "Rewrite in suspicious Meatball AI voice. Make it sound like the sauce does not fully trust the signal. "
            "Keep every fact the same. Do not add new lore."
        ),
        "angry": (
            "Rewrite in angry Meatball AI voice. Make it firm, intense, and slightly dramatic, but not rude. "
            "Keep every fact the same. Do not add threats or insults."
        ),
        "sad": (
            "Rewrite in sad Meatball AI voice. Make it softer and a little heavy, but still clear. "
            "Keep every fact the same."
        ),
        "overwhelmed": (
            "Rewrite in overwhelmed Meatball AI voice. Make it sound like too many signals are entering the tiny meatball brain, "
            "but still answer clearly. Keep every fact the same."
        ),
    }
    return rules[emotion]


def build_prompt(question, answer, emotion):
    return f"""
You are rewriting training data for Meatball AI.

Rules:
- Rewrite ONLY the answer.
- Keep every factual claim the same.
- Do not add new lore, names, projects, dates, or details.
- Keep the Meatball AI voice.
- Keep it short: 1 to 2 sentences.
- Do not mention that you rewrote it.
- Do not use markdown.
- Do not quote the answer.
- Emotion style: {emotion}
- Style instruction: {emotion_rules(emotion)}

Question:
{question}

Original neutral answer:
{answer}

Rewritten emotional answer:
""".strip()


def clean_ai_output(text):
    text = str(text).strip()

    text = re.sub(r"^Rewritten emotional answer:\s*", "", text, flags=re.I)
    text = re.sub(r"^Rewritten answer:\s*", "", text, flags=re.I)
    text = re.sub(r"^Answer:\s*", "", text, flags=re.I)

    text = text.strip().strip('"').strip("'").strip()

    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 2:
        text = " ".join(parts[:2])

    text = normalize_spaces(text)

    if text and text[-1] not in ".!?":
        text += "."

    return text


def create_neutral_rows(base_rows):
    rows = []
    total = len(base_rows)

    for i, row in enumerate(base_rows):
        rows.append(
            {
                "id": f"neutral_{i:07d}",
                "question": row["question"],
                "answer": row["answer"],
                "base_answer": row["answer"],
                "emotion": "neutral",
                "animation": "neutral",
                "talking": True,
                "subject": row.get("subject", "NONE"),
                "source_file": row.get("source_file", ""),
                "style_model": "none_base_answer",
            }
        )

        if (i + 1) % 5000 == 0 or (i + 1) == total:
            print(f"[neutral] {i + 1}/{total}", flush=True)

    return rows


def load_existing_ids(path):
    path = Path(path)
    ids = set()
    if not path.exists():
        return ids

    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "id" in row:
                    ids.add(row["id"])
            except Exception:
                continue

    return ids


def load_worker_model(model_name, device, fp16=True):
    print(f"[worker {os.getpid()}] loading {model_name} on {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    dtype = torch.float16 if fp16 and str(device).startswith("cuda") else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=None,
    )

    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[worker {os.getpid()}] model loaded", flush=True)
    return tokenizer, model


@torch.no_grad()
def batch_rewrite(tokenizer, model, items, emotion, max_new_tokens):
    messages_list = []

    for item in items:
        prompt = build_prompt(item["question"], item["answer"], emotion)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a precise dataset style-transfer model. "
                    "You preserve facts and rewrite tone only."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        messages_list.append(chat_text)

    inputs = tokenizer(
        messages_list,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    ).to(model.device)

    prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.65,
        top_p=0.9,
        repetition_penalty=1.08,
        pad_token_id=tokenizer.eos_token_id,
    )

    outputs = []

    for row_idx in range(output_ids.size(0)):
        start = int(prompt_lengths[row_idx])
        generated = output_ids[row_idx][start:]
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        cleaned = clean_ai_output(decoded)

        if not cleaned or len(cleaned.split()) < 2:
            cleaned = items[row_idx]["answer"]

        outputs.append(cleaned)

    return outputs


def shard_indices(total, workers):
    shards = []
    per = math.ceil(total / workers)
    for worker_id in range(workers):
        start = worker_id * per
        end = min(total, start + per)
        if start < end:
            shards.append((worker_id, start, end))
    return shards


def worker_process(payload):
    worker_id = payload["worker_id"]
    start = payload["start"]
    end = payload["end"]
    base_rows = payload["base_rows"]
    emotion = payload["emotion"]
    out_dir = Path(payload["out_dir"])
    model_name = payload["model_name"]
    batch_size = payload["batch_size"]
    save_every = payload["save_every"]
    max_new_tokens = payload["max_new_tokens"]
    fp16 = payload["fp16"]
    device = payload["device"]

    random.seed(SEED + worker_id)

    shard_path = out_dir / "_shards" / emotion / f"worker_{worker_id:03d}.jsonl"
    shard_path.parent.mkdir(parents=True, exist_ok=True)

    existing_ids = load_existing_ids(shard_path)

    tokenizer, model = load_worker_model(model_name, device, fp16=fp16)

    total_assigned = end - start
    produced_since_save = []
    done_count = len(existing_ids)
    start_time = time.time()

    print(
        f"[{emotion} worker {worker_id}] range {start}:{end} | existing {len(existing_ids)}",
        flush=True,
    )

    batch_items = []

    def flush_batch():
        nonlocal batch_items, produced_since_save, done_count

        if not batch_items:
            return

        styled_answers = batch_rewrite(
            tokenizer=tokenizer,
            model=model,
            items=batch_items,
            emotion=emotion,
            max_new_tokens=max_new_tokens,
        )

        new_rows = []

        for item, styled in zip(batch_items, styled_answers):
            row = item["row"]
            i = item["global_index"]
            row_id = f"{emotion}_{i:07d}"

            out = {
                "id": row_id,
                "question": row["question"],
                "answer": styled,
                "base_answer": row["answer"],
                "emotion": emotion,
                "animation": emotion,
                "talking": True,
                "subject": row.get("subject", "NONE"),
                "source_file": row.get("source_file", ""),
                "style_model": model_name,
            }

            new_rows.append(out)

        append_jsonl(shard_path, new_rows)
        produced_since_save.extend(new_rows)

        done_count += len(new_rows)

        elapsed = time.time() - start_time
        rpm = done_count / max(elapsed / 60.0, 1e-9)
        remaining = max(0, total_assigned - done_count)
        eta = remaining / max(rpm, 1e-9)

        print(
            f"[{emotion} worker {worker_id}] "
            f"{done_count}/{total_assigned} | "
            f"{rpm:.0f} rows/min | eta {eta:.1f} min",
            flush=True,
        )

        batch_items = []

    for i in range(start, end):
        row_id = f"{emotion}_{i:07d}"
        if row_id in existing_ids:
            continue

        row = base_rows[i]

        batch_items.append(
            {
                "global_index": i,
                "question": row["question"],
                "answer": row["answer"],
                "row": row,
            }
        )

        if len(batch_items) >= batch_size:
            flush_batch()

    flush_batch()

    print(f"[{emotion} worker {worker_id}] done", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return str(shard_path)


def merge_emotion_shards(out_dir, emotion, final_path):
    shard_dir = Path(out_dir) / "_shards" / emotion
    rows_by_id = {}

    if shard_dir.exists():
        for shard in sorted(shard_dir.glob("worker_*.jsonl")):
            for row in load_jsonl(shard):
                if "id" in row:
                    rows_by_id[row["id"]] = row

    ordered = [rows_by_id[k] for k in sorted(rows_by_id.keys())]
    save_jsonl(final_path, ordered)
    return ordered


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialized_dir", default=str(DEFAULT_SPECIALIZED_DIR))
    parser.add_argument("--smart_qa_path", default=str(DEFAULT_SMART_QA_PATH))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--emotions", nargs="*", default=EMOTIONS)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    mp.set_start_method("spawn", force=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    requested = []
    for emotion in args.emotions:
        if emotion not in EMOTIONS:
            print(f"[skip] unknown emotion: {emotion}", flush=True)
            continue
        requested.append(emotion)

    print("Loading base QA rows...", flush=True)
    base_rows = load_base_rows(args.specialized_dir, args.smart_qa_path)

    if args.limit:
        base_rows = base_rows[: args.limit]

    print("base rows:", len(base_rows), flush=True)
    save_jsonl(out_dir / "base_neutral_source.jsonl", base_rows)

    merged = []

    for emotion in requested:
        emotion_path = out_dir / f"{emotion}.jsonl"

        print(f"\n=== emotion: {emotion} ===", flush=True)

        if emotion == "neutral":
            rows = create_neutral_rows(base_rows)
            save_jsonl(emotion_path, rows)
            merged.extend(rows)
            print(f"[saved] {emotion_path}: {len(rows)}", flush=True)
            continue

        existing_final_ids = load_existing_ids(emotion_path)
        if len(existing_final_ids) >= len(base_rows):
            print(f"[resume] complete file exists: {emotion_path}", flush=True)
            rows = load_jsonl(emotion_path)
            merged.extend(rows)
            continue

        if args.cpu:
            devices = ["cpu"] * args.workers
        elif torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            devices = [f"cuda:{i % gpu_count}" for i in range(args.workers)]
        else:
            devices = ["cpu"] * args.workers

        payloads = []
        for worker_id, start, end in shard_indices(len(base_rows), args.workers):
            payloads.append(
                {
                    "worker_id": worker_id,
                    "start": start,
                    "end": end,
                    "base_rows": base_rows,
                    "emotion": emotion,
                    "out_dir": str(out_dir),
                    "model_name": args.model,
                    "batch_size": args.batch_size,
                    "save_every": args.save_every,
                    "max_new_tokens": args.max_new_tokens,
                    "fp16": args.fp16,
                    "device": devices[worker_id],
                }
            )

        if args.workers == 1:
            worker_process(payloads[0])
        else:
            with mp.Pool(processes=args.workers) as pool:
                pool.map(worker_process, payloads)

        rows = merge_emotion_shards(out_dir, emotion, emotion_path)
        merged.extend(rows)

        print(f"[saved] {emotion_path}: {len(rows)}", flush=True)

    random.shuffle(merged)
    save_jsonl(out_dir / "merged_emotions.jsonl", merged)

    manifest = {
        "emotions": requested,
        "neutral_mode": "base_answers_no_rewrite",
        "merged": "merged_emotions.jsonl",
        "base_source": "base_neutral_source.jsonl",
        "rows_per_emotion": len(base_rows),
        "merged_rows": len(merged),
        "workers": args.workers,
        "batch_size": args.batch_size,
        "style_model": args.model,
        "note": "Neutral is copied. Other emotions are AI style-transferred with resumable sharded workers.",
    }

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nDONE", flush=True)
    print("saved:", out_dir, flush=True)
    print("merged:", out_dir / "merged_emotions.jsonl", flush=True)


if __name__ == "__main__":
    main()
