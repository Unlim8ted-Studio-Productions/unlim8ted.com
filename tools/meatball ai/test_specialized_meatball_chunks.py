import json
import re
import argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_NGRAMS = (1, 2, 3)

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

    history = row.get("history", [])
    if isinstance(history, list):
        history_text = " ".join(str(x) for x in history)
    else:
        history_text = str(history)

    return f"question: {question} history: {history_text}"


def selector_text_from_question(question: str):
    return f"question: {question}"


def vectorize_text(text, vocab):
    tokens = input_tokenize(text)
    feats = make_input_ngrams(tokens)

    x = torch.zeros(len(vocab), dtype=torch.float32)

    counts = Counter(feats)

    for feat, count in counts.items():
        idx = vocab.get(feat, vocab["<UNK>"])
        x[idx] = min(float(count), 5.0)

    return x


def vectorize_selector_question(question, selector_vocab):
    text = selector_text_from_question(question)
    return vectorize_text(text, selector_vocab)


def vectorize_topic_question(question, input_vocab, history=None):
    if history is None:
        history = []

    row = {
        "question": question,
        "history": history,
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
# MODELS
# Must match train_selector_and_specialists_old_chunks.py
# ============================================================


class SelectorModel(nn.Module):
    def __init__(
        self, input_size, num_topics, hidden_size=HIDDEN_SIZE, dropout=DROPOUT
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_topics),
        )

    def forward(self, x):
        return self.model(x)


class ChunkAnswerModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_vocab_size,
        hidden_size=HIDDEN_SIZE,
        embed_size=EMBED_SIZE,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_vocab_size = output_vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.embedding = nn.Embedding(output_vocab_size, embed_size)

        self.decoder_cell = nn.GRUCell(embed_size, hidden_size)

        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_vocab_size),
        )

    def encode(self, x):
        return self.encoder(x)

    def decoder_step(self, prev_token, hidden):
        emb = self.embedding(prev_token)
        hidden = self.decoder_cell(emb, hidden)
        logits = self.output(hidden)
        return logits, hidden

    def forward(
        self, x, target=None, teacher_forcing=True, max_len=MAX_OUTPUT_CHUNKS + 1
    ):
        batch_size = x.size(0)

        hidden = self.encode(x)

        prev_token = torch.full(
            (batch_size,),
            BOS_ID,
            dtype=torch.long,
            device=x.device,
        )

        logits_steps = []

        for t in range(max_len):
            logits, hidden = self.decoder_step(prev_token, hidden)
            logits_steps.append(logits.unsqueeze(1))

            if teacher_forcing and target is not None:
                prev_token = target[:, t]
            else:
                prev_token = torch.argmax(logits, dim=-1)

        return torch.cat(logits_steps, dim=1)


# ============================================================
# LOAD MODELS
# ============================================================


def load_selector(model_dir, selector_vocab, labels):
    selector_path = model_dir / "selector.pt"

    checkpoint = torch.load(selector_path, map_location=DEVICE)

    model = SelectorModel(
        input_size=checkpoint.get("input_size", len(selector_vocab)),
        num_topics=checkpoint.get("num_topics", len(labels)),
        hidden_size=checkpoint.get("hidden_size", HIDDEN_SIZE),
        dropout=checkpoint.get("dropout", DROPOUT),
    ).to(DEVICE)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def load_topic_model(topic_dir):
    config = load_json(topic_dir / "config.json")
    input_vocab = load_json(topic_dir / "input_vocab.json")
    output_chunks = load_json(topic_dir / "output_chunks.json")

    model_path = topic_dir / config.get("model_pt_path", "model.pt")

    checkpoint = torch.load(model_path, map_location=DEVICE)

    model = ChunkAnswerModel(
        input_size=checkpoint.get("input_vocab_size", len(input_vocab)),
        output_vocab_size=checkpoint.get("output_vocab_size", len(output_chunks)),
        hidden_size=checkpoint.get(
            "hidden_size", config.get("hidden_size", HIDDEN_SIZE)
        ),
        embed_size=checkpoint.get("embed_size", config.get("embed_size", EMBED_SIZE)),
        dropout=checkpoint.get("dropout", config.get("dropout", DROPOUT)),
    ).to(DEVICE)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    max_output_chunks = checkpoint.get(
        "max_output_chunks",
        config.get("max_output_chunks", MAX_OUTPUT_CHUNKS),
    )

    return model, input_vocab, output_chunks, max_output_chunks


# ============================================================
# PREDICT
# ============================================================


@torch.no_grad()
def predict_topic(question, selector_model, selector_vocab, labels, top_k=5):
    selector_model.eval()

    x = vectorize_selector_question(question, selector_vocab).unsqueeze(0).to(DEVICE)

    logits = selector_model(x)[0]
    probs = torch.softmax(logits, dim=-1)

    topic_id = int(torch.argmax(probs).item())
    confidence = float(probs[topic_id].item())

    topic = labels.get(str(topic_id), labels.get(topic_id))

    top = torch.topk(probs, k=min(top_k, probs.numel()))

    top_topics = []
    for prob, idx in zip(top.values.tolist(), top.indices.tolist()):
        t = labels.get(str(int(idx)), labels.get(int(idx)))
        top_topics.append((t, float(prob)))

    return topic, confidence, top_topics


@torch.no_grad()
def predict_answer(
    question, model, input_vocab, output_chunks, max_output_chunks, history=None
):
    model.eval()

    x = vectorize_topic_question(question, input_vocab, history=history)
    x = x.unsqueeze(0).to(DEVICE)

    hidden = model.encode(x)

    prev_token = torch.tensor([BOS_ID], dtype=torch.long, device=DEVICE)

    pred_ids = []
    scores = []

    for _ in range(max_output_chunks + 1):
        logits, hidden = model.decoder_step(prev_token, hidden)

        probs = torch.softmax(logits, dim=-1)
        score, token = torch.max(probs, dim=-1)

        token_id = int(token.item())
        score_value = float(score.item())

        if token_id == EOS_ID:
            break

        if token_id in (PAD_ID, BOS_ID):
            break

        pred_ids.append(token_id)
        scores.append(score_value)

        prev_token = token

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
        help="Folder with selector.pt, selector_vocab.json, selector_labels.json, and topics/",
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

    print(f"device: {DEVICE}")
    print("Loaded PT Meatball specialized runtime.")
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
