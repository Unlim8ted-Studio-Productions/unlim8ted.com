# test_general_cover_chunks_onnx.py

import argparse
import json
import re
from pathlib import Path
from collections import Counter

import numpy as np
import onnxruntime as ort

DEFAULT_MODEL_PATH = Path(r"assets/models/general_cover_chunks_test/model.onnx")

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def normalize_spaces(text):
    return re.sub(r"\s+", " ", str(text).replace("\n", " ")).strip()


def input_normalize(text):
    text = str(text).lower().replace("\n", " ")
    text = re.sub(r"[^a-z0-9_!?.,' -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def input_tokenize(text):
    text = input_normalize(text)
    return text.split() if text else []


def make_input_ngrams(tokens, ngrams=(1, 2, 3)):
    feats = []
    for n in ngrams:
        for i in range(len(tokens) - n + 1):
            feats.append("_".join(tokens[i : i + n]))
    return feats


def vectorize_question(question, input_vocab):
    text = f"question: {question}"
    tokens = input_tokenize(text)
    feats = make_input_ngrams(tokens)

    x = np.zeros((1, len(input_vocab)), dtype=np.float32)
    counts = Counter(feats)

    unk = input_vocab.get("<UNK>", 1)

    for feat, count in counts.items():
        idx = input_vocab.get(feat, unk)
        x[0, idx] = min(float(count), 5.0)

    return x


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


def decode_ids(ids, output_chunks):
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


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_runtime(model_path):
    model_dir = model_path.parent

    input_vocab = load_json(model_dir / "input_vocab.json")
    output_chunks = load_json(model_dir / "output_chunks.json")

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    return session, input_name, input_vocab, output_chunks


def predict(
    question, session, input_name, input_vocab, output_chunks, show_chunks=False
):
    x = vectorize_question(question, input_vocab)

    logits = session.run(None, {input_name: x})[0]

    # Shape should be [batch, time, vocab]
    ids = np.argmax(logits[0], axis=-1).tolist()
    answer = decode_ids(ids, output_chunks)

    print()
    print("=" * 80)
    print("QUESTION:")
    print(question)
    print()
    print("ANSWER:")
    print(answer)

    if show_chunks:
        print()
        print("CHUNKS:")
        for idx in ids:
            if idx == EOS_ID:
                print(f"{idx:6d}  <EOS>")
                break
            if idx in (PAD_ID, BOS_ID, UNK_ID):
                continue
            text = (
                output_chunks[idx]["text"] if 0 <= idx < len(output_chunks) else "???"
            )
            print(f"{idx:6d}  {text}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--question", default=None)
    parser.add_argument("--show_chunks", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    session, input_name, input_vocab, output_chunks = load_runtime(model_path)

    print("Loaded:", model_path)
    print("Input vocab:", len(input_vocab))
    print("Output chunks:", len(output_chunks))

    if args.question:
        predict(
            args.question,
            session,
            input_name,
            input_vocab,
            output_chunks,
            args.show_chunks,
        )
        return

    print()
    print("Interactive mode. Type quit / exit / stop.")
    while True:
        q = input("You: ").strip()
        if q.lower() in {"quit", "exit", "stop"}:
            break
        if q:
            predict(
                q, session, input_name, input_vocab, output_chunks, args.show_chunks
            )


if __name__ == "__main__":
    main()
