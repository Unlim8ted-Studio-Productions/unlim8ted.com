import json
import re
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import onnx
import onnxruntime as ort

# ============================================================
# CONFIG
# ============================================================

INPUT_NGRAMS = (1, 2, 3)
SELECTOR_MAX_LEN = 48

HIDDEN_SIZE = 192
EMBED_SIZE = 128
DROPOUT = 0.35

MAX_OUTPUT_CHUNKS = 40

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


# ============================================================
# FILE UTILS
# ============================================================


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# INPUT TOKENIZATION / VECTORIZATION
# Same logic as training file
# ============================================================


def input_normalize(text: str) -> str:
    text = str(text).lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[^a-z0-9_!?.,' -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def input_tokenize(text: str):
    text = input_normalize(text)
    if not text:
        return []
    return text.split()


def make_input_ngrams(tokens, ngrams=INPUT_NGRAMS):
    feats = []

    for n in ngrams:
        if len(tokens) < n:
            continue

        for i in range(len(tokens) - n + 1):
            feats.append("_".join(tokens[i : i + n]))

    return feats


def row_to_input_text(row):
    question = row.get("question", "")
    return f"question: {question}"


def selector_text_from_question(question: str):
    return f"question: {question}"


def vectorize_text(text, vocab):
    tokens = input_tokenize(text)
    feats = make_input_ngrams(tokens)

    x = np.zeros(len(vocab), dtype=np.float32)

    counts = Counter(feats)

    for feat, count in counts.items():
        idx = vocab.get(feat, vocab["<UNK>"])
        x[idx] = min(float(count), 5.0)

    return x


def vectorize_selector_question(question, selector_vocab):
    text = selector_text_from_question(question)
    return vectorize_text(text, selector_vocab)


def encode_selector_question_ids(
    question,
    selector_vocab,
    max_len=SELECTOR_MAX_LEN,
    max_token_id=None,
):
    text = selector_text_from_question(question)
    tokens = input_tokenize(text)
    feats = make_input_ngrams(tokens)

    pad_id = int(selector_vocab.get("<PAD>", 0))
    unk_id = int(selector_vocab.get("<UNK>", 1))

    ids = []
    for feat in feats:
        token_id = int(selector_vocab.get(feat, unk_id))
        if max_token_id is not None and token_id > max_token_id:
            token_id = unk_id
        ids.append(token_id)
        if len(ids) >= max_len:
            break

    while len(ids) < max_len:
        ids.append(pad_id)

    return np.asarray(ids, dtype=np.int64)


def vectorize_topic_question(question, input_vocab, history=None):
    row = {
        "question": question,
    }

    text = row_to_input_text(row)
    return vectorize_text(text, input_vocab)


# ============================================================
# OUTPUT DECODING
# Same join logic as training file
# ============================================================


def join_chunk_texts(chunks):
    out = ""

    for chunk in chunks:
        chunk = str(chunk).strip()

        if not chunk:
            continue

        if chunk in [".", ",", "!", "?", ":", ";", "%", ")", "]", "}"]:
            out = out.rstrip() + chunk
        elif chunk in ["(", "[", "{"]:
            if out and not out.endswith(" "):
                out += " "
            out += chunk
        elif chunk.startswith(("'", "’")):
            out = out.rstrip() + chunk
        else:
            if out and not out.endswith((" ", "(", "[", "{", "'", "’")):
                out += " "
            out += chunk

    out = re.sub(r"\s+", " ", out).strip()
    return out


def decode_chunk_ids(ids, output_chunks):
    texts = []

    for idx in ids:
        idx = int(idx)

        if idx == EOS_ID:
            break

        if idx in (PAD_ID, BOS_ID):
            continue

        if idx == UNK_ID:
            continue

        if 0 <= idx < len(output_chunks):
            text = output_chunks[idx]["text"]
            texts.append(text)

    return join_chunk_texts(texts)


# ============================================================
# LOAD MODELS
# ============================================================


def create_session(path):
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(str(path), providers=providers)


def infer_selector_max_token_id(path):
    model = onnx.load(str(path))

    for init in model.graph.initializer:
        name = init.name.lower()
        if "embedding" in name and "weight" in name and init.dims:
            return int(init.dims[0]) - 1

    return None


def cast_input_for_session(session, array):
    input_meta = session.get_inputs()[0]
    input_type = input_meta.type

    if input_type == "tensor(int64)":
        return np.asarray(array, dtype=np.int64)

    if input_type == "tensor(int32)":
        return np.asarray(array, dtype=np.int32)

    if input_type == "tensor(float)":
        return np.asarray(array, dtype=np.float32)

    if input_type == "tensor(double)":
        return np.asarray(array, dtype=np.float64)

    return np.asarray(array)


def run_first_output(session, array):
    input_name = session.get_inputs()[0].name
    cast_array = cast_input_for_session(session, array)
    outputs = session.run(None, {input_name: cast_array})
    return outputs[0]


def load_selector(model_dir, selector_vocab, labels):
    selector_path = model_dir / "selector.onnx"
    if not selector_path.exists():
        raise FileNotFoundError(f"Missing selector ONNX: {selector_path}")
    return {
        "session": create_session(selector_path),
        "max_token_id": infer_selector_max_token_id(selector_path),
    }


def load_topic_model(topic_dir):
    config = load_json(topic_dir / "config.json")
    input_vocab = load_json(topic_dir / "input_vocab.json")
    output_chunks = load_json(topic_dir / "output_chunks.json")

    model_rel = config.get("model_onnx_path", "../model.onnx")
    model_path = (topic_dir / model_rel).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Missing topic ONNX: {model_path}")

    max_output_chunks = config.get("max_output_chunks", MAX_OUTPUT_CHUNKS)

    return create_session(model_path), input_vocab, output_chunks, max_output_chunks


# ============================================================
# PREDICT
# ============================================================


def predict_topic(question, selector_model, selector_vocab, labels, top_k=5):
    session = selector_model["session"]
    max_token_id = selector_model.get("max_token_id")
    input_meta = session.get_inputs()[0]

    if input_meta.type in ("tensor(int64)", "tensor(int32)"):
        shape = input_meta.shape
        seq_len = SELECTOR_MAX_LEN
        if len(shape) >= 2 and isinstance(shape[1], int):
            seq_len = int(shape[1])
        x = encode_selector_question_ids(
            question,
            selector_vocab,
            max_len=seq_len,
            max_token_id=max_token_id,
        ).reshape(1, -1)
    else:
        x = vectorize_selector_question(question, selector_vocab).reshape(1, -1)

    logits = run_first_output(session, x)[0]
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits)
    probs = np.exp(shifted)
    probs /= max(float(np.sum(probs)), 1e-8)

    topic_id = int(np.argmax(probs))
    confidence = float(probs[topic_id])

    topic = labels.get(str(topic_id), labels.get(topic_id))

    top_topics = []
    top_indices = np.argsort(probs)[::-1][: min(top_k, probs.shape[0])]
    for idx in top_indices:
        prob = float(probs[idx])
        t = labels.get(str(int(idx)), labels.get(int(idx)))
        top_topics.append((t, prob))

    return topic, confidence, top_topics


def predict_answer(
    question, model, input_vocab, output_chunks, max_output_chunks, history=None
):
    x = vectorize_topic_question(question, input_vocab, history=history)
    x = x.reshape(1, -1)
    logits_steps = run_first_output(model, x)[0]

    pred_ids = []
    scores = []

    for step_logits in logits_steps[: max_output_chunks + 1]:
        step_logits = np.asarray(step_logits, dtype=np.float32)
        shifted = step_logits - np.max(step_logits)
        probs = np.exp(shifted)
        probs /= max(float(np.sum(probs)), 1e-8)
        token_id = int(np.argmax(probs))
        score_value = float(probs[token_id])

        if token_id == EOS_ID:
            break

        if token_id in (PAD_ID, BOS_ID):
            break

        pred_ids.append(token_id)
        scores.append(score_value)

    answer = decode_chunk_ids(pred_ids + [EOS_ID], output_chunks)

    chunks = []
    for idx, score in zip(pred_ids, scores):
        text = output_chunks[idx]["text"] if 0 <= idx < len(output_chunks) else "???"
        chunks.append((idx, score, text))

    return answer, pred_ids, chunks


# ============================================================
# SIMPLE SMALLTALK GUARD
# ============================================================


def maybe_smalltalk(question):
    q = question.strip().lower()

    greetings = {
        "hi",
        "hello",
        "hey",
        "yo",
        "sup",
        "what's up",
        "whats up",
    }

    exits = {
        "nvm",
        "never mind",
        "nevermind",
        "forget it",
    }

    thanks = {
        "thanks",
        "thank you",
        "thx",
    }

    if q in greetings:
        return "Hey. Meatball is listening."

    if q in exits:
        return "No worries. The meatball rolls on."

    if q in thanks:
        return "You got it."

    return None


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_dir",
        default="assets/models/specialized_meatball_chunks",
        help="Folder with selector.onnx, selector_vocab.json, selector_labels.json, and topics/",
    )

    parser.add_argument(
        "--question",
        default=None,
        help="Optional one-shot question",
    )

    parser.add_argument(
        "--show_top_topics",
        action="store_true",
        help="Show top selector topics",
    )

    parser.add_argument(
        "--force_topic",
        default=None,
        help="Skip selector and force a topic, e.g. --force_topic art",
    )

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    selector_vocab = load_json(model_dir / "selector_vocab.json")
    labels = load_json(model_dir / "selector_labels.json")

    selector_model = load_selector(model_dir, selector_vocab, labels)

    topic_cache = {}

    print("Loaded ONNX Meatball specialized runtime.")
    print(f"model_dir: {model_dir}")
    print()

    def get_topic_runtime(topic):
        if topic not in topic_cache:
            topic_dir = model_dir / "topics" / topic

            if not topic_dir.exists():
                raise FileNotFoundError(f"Missing topic directory: {topic_dir}")

            topic_cache[topic] = load_topic_model(topic_dir)

        return topic_cache[topic]

    def run_one(question):
        smalltalk = maybe_smalltalk(question)

        if smalltalk:
            print("=" * 70)
            print("QUESTION:")
            print(question)
            print()
            print("SMALLTALK:")
            print(smalltalk)
            print("=" * 70)
            print()
            return

        if args.force_topic:
            topic = args.force_topic
            confidence = None
            top_topics = []
        else:
            topic, confidence, top_topics = predict_topic(
                question,
                selector_model,
                selector_vocab,
                labels,
            )

        model, input_vocab, output_chunks, max_output_chunks = get_topic_runtime(topic)

        answer, ids, chunks = predict_answer(
            question,
            model,
            input_vocab,
            output_chunks,
            max_output_chunks,
        )

        print("=" * 70)
        print("QUESTION:")
        print(question)
        print()

        print(f"SELECTED TOPIC: {topic}")

        if confidence is not None:
            print(f"CONFIDENCE: {confidence:.3f}")

        if args.show_top_topics and top_topics:
            print()
            print("TOP TOPICS:")
            for t, p in top_topics:
                print(f"  {t}: {p:.3f}")

        print()
        print("ANSWER:")
        print(answer)

        print()
        print("CHUNK IDS:")
        print(ids)

        print()
        print("CHUNKS:")
        for idx, score, text in chunks:
            print(f"  {score:.3f}  {idx:5d}  {text}")

        print("=" * 70)
        print()

    if args.question:
        run_one(args.question)
        return

    while True:
        q = input("You: ").strip()

        if q.lower() in ["q", "quit", "exit"]:
            break

        if not q:
            continue

        run_one(q)


if __name__ == "__main__":
    main()
