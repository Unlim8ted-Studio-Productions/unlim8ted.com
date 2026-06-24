import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

INPUT_NGRAMS = (1, 2, 3)
MAX_OUTPUT_CHUNKS = 40

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def load_json(path: Path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if question and answer:
                rows.append({"question": question, "answer": answer})
    return rows


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


def vectorize_question(question: str, input_vocab):
    text = f"question: {question}"
    tokens = input_tokenize(text)
    feats = make_input_ngrams(tokens)

    x = np.zeros(len(input_vocab), dtype=np.float32)
    counts = Counter(feats)

    for feat, count in counts.items():
        idx = input_vocab.get(feat, input_vocab["<UNK>"])
        x[idx] = min(float(count), 5.0)

    return x.reshape(1, -1)


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

    return re.sub(r"\s+", " ", out).strip()


def decode_chunk_ids(ids, output_chunks):
    texts = []

    for idx in ids:
        idx = int(idx)

        if idx == EOS_ID:
            break
        if idx in (PAD_ID, BOS_ID, UNK_ID):
            continue
        if 0 <= idx < len(output_chunks):
            texts.append(output_chunks[idx]["text"])

    return join_chunk_texts(texts)


def create_session(path: Path):
    if ort is None:
        raise RuntimeError(
            "onnxruntime is not installed in the current Python environment. "
            "Install it with: pip install onnxruntime"
        )
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def run_first_output(session, array):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: np.asarray(array, dtype=np.float32)})
    return outputs[0]


def predict_answer(question, session, input_vocab, output_chunks, max_output_chunks):
    x = vectorize_question(question, input_vocab)
    logits_steps = run_first_output(session, x)[0]

    pred_ids = []
    chunks = []

    for step_logits in logits_steps[: max_output_chunks + 1]:
        step_logits = np.asarray(step_logits, dtype=np.float32)
        shifted = step_logits - np.max(step_logits)
        probs = np.exp(shifted)
        probs /= max(float(np.sum(probs)), 1e-8)

        token_id = int(np.argmax(probs))
        score = float(probs[token_id])

        if token_id == EOS_ID:
            break
        if token_id in (PAD_ID, BOS_ID):
            break

        pred_ids.append(token_id)
        text = output_chunks[token_id]["text"] if 0 <= token_id < len(output_chunks) else "???"
        chunks.append((token_id, score, text))

    answer = decode_chunk_ids(pred_ids + [EOS_ID], output_chunks)
    return answer, pred_ids, chunks


def print_result(question, expected, predicted, pred_ids, chunks):
    print("=" * 70)
    print("QUESTION:")
    print(question)
    print()

    if expected is not None:
        print("EXPECTED:")
        print(expected)
        print()

    print("PREDICTED:")
    print(predicted)
    print()

    print("CHUNK IDS:")
    print(pred_ids)
    print()

    print("CHUNKS:")
    for idx, score, text in chunks:
        print(f"  {score:.3f}  {idx:5d}  {text}")

    print("=" * 70)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        default="assets/models/specialized_meatball_chunks/topics/general",
        help="Path to the trained general topic model directory.",
    )
    parser.add_argument(
        "--dataset",
        default="assets/data/specialized_QA/general.jsonl",
        help="General dataset to sample test questions from.",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Optional one-shot question to test.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="How many dataset samples to test when --question is not provided.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset sampling.",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    model_dir = Path(args.model_dir)
    dataset_path = Path(args.dataset)

    config = load_json(model_dir / "config.json")
    input_vocab = load_json(model_dir / "input_vocab.json")
    output_chunks = load_json(model_dir / "output_chunks.json")
    session = create_session(model_dir / config.get("model_onnx_path", "model.onnx"))
    max_output_chunks = int(config.get("max_output_chunks", MAX_OUTPUT_CHUNKS))

    print("Loaded general specialist runtime.")
    print(f"model_dir: {model_dir}")
    print(f"dataset:   {dataset_path}")
    print()

    if args.question:
        predicted, pred_ids, chunks = predict_answer(
            args.question,
            session,
            input_vocab,
            output_chunks,
            max_output_chunks,
        )
        print_result(args.question, None, predicted, pred_ids, chunks)
        return

    rows = load_jsonl(dataset_path)
    if not rows:
        raise RuntimeError(f"No usable rows found in {dataset_path}")

    samples = random.sample(rows, min(args.samples, len(rows)))

    for row in samples:
        predicted, pred_ids, chunks = predict_answer(
            row["question"],
            session,
            input_vocab,
            output_chunks,
            max_output_chunks,
        )
        print_result(row["question"], row["answer"], predicted, pred_ids, chunks)


if __name__ == "__main__":
    main()
