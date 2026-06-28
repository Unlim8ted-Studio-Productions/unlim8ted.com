import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPECIALIZED_DIR = Path("assets/data/specialized_QA")
SMART_QA_PATH = Path("tools/SmartMeatballQA.jsonl")
OUT_DIR = Path("assets/models/output_sanity_checker")

LABELS = ["accept", "confused_fallback"]
CHAR_NGRAMS = (2, 3, 4, 5)
WORD_NGRAMS = (1, 2, 3)
MAX_VOCAB = 18000
HIDDEN = 256
DROPOUT = 0.22
BATCH_SIZE = 128
EPOCHS = 28
PATIENCE = 6
LR = 8e-4
WEIGHT_DECAY = 2e-3

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_jsonl(path):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def normalize(text):
    text = str(text or "").lower().replace("\n", " ")
    text = re.sub(r"[^a-z0-9!?.,' +\-*/=()%$]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def pair_features(question, answer):
    q = normalize(question)
    a = normalize(answer)
    merged = f"question: {q} answer: {a}"
    wrapped = f"<{merged}>"
    out = []
    for n in CHAR_NGRAMS:
        for i in range(0, max(0, len(wrapped) - n + 1)):
            out.append(f"c:{wrapped[i:i+n]}")
    words = merged.split()
    for n in WORD_NGRAMS:
        for i in range(0, max(0, len(words) - n + 1)):
            out.append(f"w:{'_'.join(words[i:i+n])}")

    q_words = set(re.findall(r"[a-z0-9']+", q))
    a_words = set(re.findall(r"[a-z0-9']+", a))
    overlap = len(
        (q_words & a_words) - {"what", "is", "the", "a", "an", "tell", "me", "about"}
    )
    if overlap == 0:
        out.append("flag:no_overlap")
    if overlap >= 2:
        out.append("flag:good_overlap")
    if a.startswith("- "):
        out.append("flag:list")
    if a in {"i'm not", "the meatball chooses to interpret that as"}:
        out.append("flag:known_bad")
    if len(a.split()) <= 2:
        out.append("flag:very_short")
    if a.count("?") > 2:
        out.append("flag:many_questions")
    if re.fullmatch(r"(yes|yeah|yep|no|nope)\.?", a):
        out.append("flag:bare_yes_no")
    if (
        a.startswith("i don't know")
        or a.startswith("im not")
        or a.startswith("i'm not")
    ):
        out.append("flag:weak_fallback")
    return out


def build_vocab(rows):
    counts = Counter()
    for row in rows:
        counts.update(pair_features(row["question"], row["answer"]))
    vocab = {"<unk>": 0}
    for token, _ in counts.most_common(MAX_VOCAB - 1):
        vocab[token] = len(vocab)
    return vocab


def vectorize(question, answer, vocab):
    vec = torch.zeros(len(vocab), dtype=torch.float32)
    counts = Counter(pair_features(question, answer))
    for token, count in counts.items():
        vec[vocab.get(token, 0)] = min(float(count), 5.0)
    return vec


class RowDataset(Dataset):
    def __init__(self, rows, vocab):
        self.rows = rows
        self.vocab = vocab

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        x = vectorize(row["question"], row["answer"], self.vocab)
        y = LABELS.index(row["label"])
        return x, torch.tensor(y, dtype=torch.long)


class TinyClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def gather_good_pairs(limit=18000):
    pairs = []
    seen = set()

    for path in sorted(SPECIALIZED_DIR.glob("*.jsonl")):
        for row in load_jsonl(path):
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            if not question or not answer:
                continue
            key = (question.lower(), answer.lower())
            if key in seen:
                continue
            seen.add(key)
            pairs.append({"question": question, "answer": answer})
            if len(pairs) >= limit:
                return pairs

    for row in load_jsonl(SMART_QA_PATH):
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if not question or not answer:
            continue
        key = (question.lower(), answer.lower())
        if key in seen:
            continue
        seen.add(key)
        pairs.append({"question": question, "answer": answer})
        if len(pairs) >= limit:
            break

    return pairs


def corrupt_answer(answer):
    text = str(answer).strip()
    mode = random.choice(["truncate", "prefix_only", "repeat", "garble", "bare_fragment"])

    if mode == "truncate":
        words = text.split()
        return " ".join(words[: max(1, len(words) // 3)])
    if mode == "prefix_only":
        return random.choice([
            "The Meatball chooses to interpret that as",
            "I'm not",
            "Thank you. The sauce accepts the compliment.",
        ])
    if mode == "repeat":
        sentence = text.split(".")[0].strip()
        return f"{sentence}. {sentence}."
    if mode == "garble":
        text = re.sub(r"[aeiou]", "", text[:48], flags=re.I)
        return text or "??"
    return random.choice([
        "What.",
        "The sauce...",
        "Maybe maybe maybe.",
        "It is because it is.",
        "I don't know.",
    ])


def build_rows(limit):
    good = gather_good_pairs(limit)
    if not good:
        raise RuntimeError("No question/answer pairs found for output sanity training.")

    rows = [
        {"question": pair["question"], "answer": pair["answer"], "label": "accept"}
        for pair in good
    ]

    all_answers = [pair["answer"] for pair in good]

    for idx, pair in enumerate(good):
        wrong_answer = all_answers[(idx * 7 + 13) % len(all_answers)]
        if wrong_answer != pair["answer"]:
            rows.append(
                {
                    "question": pair["question"],
                    "answer": wrong_answer,
                    "label": "confused_fallback",
                }
            )

    for pair in good:
        rows.append(
            {
                "question": pair["question"],
                "answer": corrupt_answer(pair["answer"]),
                "label": "confused_fallback",
            }
        )

    rows.extend([
        {
            "question": "what is The Glitch",
            "answer": "The Meatball chooses to interpret that as",
            "label": "confused_fallback",
        },
        {
            "question": "what is The Glitch",
            "answer": "I'm not",
            "label": "confused_fallback",
        },
        {
            "question": "thanks",
            "answer": "Thank you. The sauce accepts the compliment.",
            "label": "accept",
        },
        {
            "question": "yes, what year was seinfield started",
            "answer": "the speed of light is approximately 299,792,458 meters per second.",
            "label": "confused_fallback",
        },
    ])

    random.shuffle(rows)
    return rows


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += y.size(0)
    return {"loss": total_loss / max(1, total), "acc": correct / max(1, total)}


def train(args):
    rows = build_rows(args.limit)
    split = max(1, int(len(rows) * 0.12))
    val_rows = rows[:split]
    train_rows = rows[split:]
    vocab = build_vocab(train_rows)

    train_ds = RowDataset(train_rows, vocab)
    val_ds = RowDataset(val_rows, vocab)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = TinyClassifier(len(vocab), len(LABELS)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * y.size(0)
            total += y.size(0)

        train_loss = total_loss / max(1, total)
        metrics = evaluate(model, val_loader, loss_fn)
        print(
            f"epoch {epoch:03d} | train_loss {train_loss:.4f} | "
            f"val_loss {metrics['loss']:.4f} | acc {metrics['acc']:.4f}"
        )

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": best_state,
                    "input_size": len(vocab),
                    "num_classes": len(LABELS),
                },
                OUT_DIR / "output_sanity_checker.pt",
            )
            print("[saved best]")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("[early stop]")
                break

    save_json(OUT_DIR / "input_vocab.json", vocab)
    save_json(OUT_DIR / "labels.json", LABELS)
    save_json(
        OUT_DIR / "config.json",
        {
            "model_type": "qa_pair_alignment_classifier",
            "feature_mode": "qa_pair",
            "char_ngrams": list(CHAR_NGRAMS),
            "word_ngrams": list(WORD_NGRAMS),
            "hidden": HIDDEN,
            "dropout": DROPOUT,
            "note": "Classifies whether an answer fits the input question using question and answer pairs.",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--limit", type=int, default=18000)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
