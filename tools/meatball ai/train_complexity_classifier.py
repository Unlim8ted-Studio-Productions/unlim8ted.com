# train_complexity_classifier.py
#
# Trains a tiny classifier for question routing/complexity:
#
# normal_qa
# list
# compare
# multi_part
# followup
# smalltalk
# unknown
#
# This does NOT do math. Your existing math classifier should still run separately.
#
# Output:
# assets/models/complexity_classifier/complexity_classifier.pt
# assets/models/complexity_classifier/input_vocab.json
# assets/models/complexity_classifier/labels.json
# assets/models/complexity_classifier/config.json
#
# Run:
# python train_complexity_classifier.py

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DIR = Path("assets/models/complexity_classifier")

SPECIALIZED_DIR = Path("assets/data/specialized_QA")
SMART_QA_PATH = Path("tools/SmartMeatballQA.jsonl")

LABELS = [
    "normal_qa",
    "list",
    "compare",
    "multi_part",
    "followup",
    "smalltalk",
    "unknown",
]

CHAR_NGRAMS = (2, 3, 4, 5)
WORD_NGRAMS = (1, 2, 3)

MAX_VOCAB = 14000
HIDDEN = 320
DROPOUT = 0.22
BATCH_SIZE = 128
EPOCHS = 35
VAL_SPLIT = 0.12
LR = 8e-4
WEIGHT_DECAY = 2e-3
GRAD_CLIP = 1.0

KNOWN_SUBJECTS = [
    "dogs",
    "cats",
    "timecat",
    "time cat",
    "the glitch",
    "glitch",
    "unlim8ted",
    "unlimited",
    "meatball",
    "meatball ai",
    "subway trains",
    "cars",
    "trees",
    "space",
    "games",
    "movies",
]

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
    text = str(text).lower().replace("\n", " ")
    text = re.sub(r"[^a-z0-9!?.,' +\-*/=()%$]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_no_punc(text):
    text = normalize(text)
    text = re.sub(r"[!?.,:;\"'`()\[\]{}]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def corrupt(text):
    text = normalize(text)
    out = []

    for ch in text:
        r = random.random()

        if ch.isalpha() and r < 0.018:
            continue

        if ch.isalpha() and 0.018 <= r < 0.035:
            out.append(ch)
            out.append(ch)
            continue

        if ch.isalpha() and 0.035 <= r < 0.055:
            subs = {
                "a": "s",
                "s": "a",
                "e": "r",
                "r": "e",
                "i": "o",
                "o": "i",
                "t": "y",
                "y": "t",
                "n": "m",
                "m": "n",
                "u": "i",
                "p": "o",
                "c": "v",
                "v": "c",
            }
            out.append(subs.get(ch, ch))
            continue

        out.append(ch)

    text = "".join(out)

    if random.random() < 0.35:
        text = re.sub(r"[!?.,:;\"'`()\[\]{}]", " ", text)

    return re.sub(r"\s+", " ", text).strip()


def add(rows, label, examples, augment=8):
    for text in examples:
        rows.append({"text": text, "label": label})
        for _ in range(augment):
            rows.append({"text": corrupt(text), "label": label})


def load_project_normal_rows(limit=12000):
    out = []
    seen = set()

    files = sorted(SPECIALIZED_DIR.glob("*.jsonl"))
    for path in files:
        for r in load_jsonl(path):
            q = str(r.get("question", "")).strip()
            if not q:
                continue

            key = q.lower()
            if key in seen:
                continue
            seen.add(key)

            out.append({"text": q, "label": "normal_qa"})

            if len(out) >= limit:
                return out

    for r in load_jsonl(SMART_QA_PATH):
        q = str(r.get("question", "")).strip()
        if not q:
            continue

        key = q.lower()
        if key in seen:
            continue

        seen.add(key)
        out.append({"text": q, "label": "normal_qa"})

        if len(out) >= limit:
            break

    return out


def build_synthetic_rows(extra_normal=7000):
    rows = []

    smalltalk = [
        "hi",
        "hello",
        "hey",
        "yo",
        "sup",
        "how are you",
        "are you awake",
        "good morning",
        "good night",
        "thanks",
        "thank you",
        "ok",
        "okay",
        "cool",
    ]

    unknown = [
        "",
        "???",
        "asdf qwer",
        "zzzzzz",
        "what",
        "uh",
        "??? ???",
        "thing stuff",
        "no context",
        "words words words",
    ]

    followup = [
        "what does that mean",
        "what did that mean",
        "explain that",
        "what do you mean",
        "what was that",
        "why did you say that",
        "tell me more",
        "more",
        "what about it",
        "does it have clothes",
        "who made it",
        "why does it matter",
        "is it real",
        "does that have lore",
        "what about that",
        "and who made it",
    ]

    list_q = [
        "facts about dogs",
        "list facts about cats",
        "give me examples of games",
        "show me features of timecat",
        "what are some facts about the glitch",
        "list projects",
        "what are the main things about unlim8ted",
        "give me a list of facts about meatball ai",
        "examples of animals",
        "types of vehicles",
        "what are some examples of trees",
        "list the features",
    ]

    compare = [
        "compare cats and dogs",
        "cats vs dogs",
        "timecat versus the glitch",
        "what is the difference between cats and dogs",
        "is there a difference between timecat and the glitch",
        "compare unlim8ted and meatball ai",
        "contrast dogs with cats",
        "which is better cats or dogs",
        "how are cats and dogs different",
    ]

    multi = [
        "what is timecat and who made it",
        "tell me about the glitch and does it have lore",
        "what is unlim8ted also who made it",
        "facts about dogs and compare cats and dogs",
        "what is meatball ai plus tell me about timecat",
        "who made the glitch and what is it",
        "what is a dog and what is a cat",
        "tell me a joke and explain it",
        "what is timecat and does it have clothes",
    ]

    normal = [
        "what is dogs",
        "what are dogs",
        "what is a dog",
        "tell me about dogs",
        "what is timecat",
        "tell me about the glitch",
        "who made meatball ai",
        "explain unlim8ted",
        "does timecat have lore",
        "what is a subway train",
        "how does a car work",
        "what is a tree",
        "tell a joke",
        "why did the meatball study astronomy",
    ]

    add(rows, "smalltalk", smalltalk, augment=12)
    add(rows, "unknown", unknown, augment=12)
    add(rows, "followup", followup, augment=14)
    add(rows, "list", list_q, augment=14)
    add(rows, "compare", compare, augment=14)
    add(rows, "multi_part", multi, augment=14)
    add(rows, "normal_qa", normal, augment=10)

    normal_templates = [
        "what is {s}",
        "what are {s}",
        "tell me about {s}",
        "who made {s}",
        "does {s} have lore",
        "explain {s}",
        "how does {s} work",
        "why is {s} important",
    ]

    for _ in range(extra_normal):
        text = random.choice(normal_templates).format(s=random.choice(KNOWN_SUBJECTS))
        if random.random() < 0.30:
            text = corrupt(text)
        rows.append({"text": text, "label": "normal_qa"})

    random.shuffle(rows)
    return rows


def make_features(text):
    text = normalize(text)
    no_punc = normalize_no_punc(text)

    feats = []

    s = f"<{text}>"
    for n in CHAR_NGRAMS:
        for i in range(len(s) - n + 1):
            feats.append("c:" + s[i : i + n])

    words = text.split()
    for n in WORD_NGRAMS:
        for i in range(len(words) - n + 1):
            feats.append("w:" + "_".join(words[i : i + n]))

    if not no_punc:
        feats.append("flag:empty")
    if "?" in text:
        feats.append("flag:question")
    if "!" in text:
        feats.append("flag:bang")
    if re.search(r"\d", text):
        feats.append("flag:number")
    if re.search(r"[+\-*/=]", text):
        feats.append("flag:operator")

    if re.search(r"\b(facts|list|examples|features|types|projects)\b", no_punc):
        feats.append("flag:list_word")
    if re.search(
        r"\b(compare|contrast|vs|versus|difference|different|better)\b", no_punc
    ):
        feats.append("flag:compare_word")
    if re.search(r"\b(and|also|plus)\b", no_punc):
        feats.append("flag:connector")
    if re.search(r"\b(it|that|this|more|they|them|their)\b", no_punc):
        feats.append("flag:followup_pronoun")
    if no_punc in {"hi", "hello", "hey", "yo", "sup", "thanks", "thank you"}:
        feats.append("flag:smalltalk_exact")
    if no_punc in {
        "what does that mean",
        "what do you mean",
        "explain that",
        "what was that",
        "tell me more",
        "more",
    }:
        feats.append("flag:followup_exact")

    return feats


def build_vocab(rows):
    counter = Counter()
    for r in rows:
        counter.update(make_features(r["text"]))

    vocab = {"<UNK>": 0}

    for feat, _ in counter.most_common(MAX_VOCAB - 1):
        vocab[feat] = len(vocab)

    return vocab


def vectorize(text, vocab):
    x = torch.zeros(len(vocab), dtype=torch.float32)
    counts = Counter(make_features(text))

    for feat, count in counts.items():
        x[vocab.get(feat, 0)] = min(float(count), 5.0)

    return x


class ComplexityDataset(Dataset):
    def __init__(self, rows, vocab, label_map):
        self.rows = rows
        self.vocab = vocab
        self.label_map = label_map

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        x = vectorize(row["text"], self.vocab)
        y = torch.tensor(self.label_map[row["label"]], dtype=torch.long)
        return x, y


class ComplexityClassifier(nn.Module):
    def __init__(self, input_size, classes):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN // 2),
            nn.LayerNorm(HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN // 2, classes),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    batches = 0
    correct = 0
    total = 0

    per_label_correct = Counter()
    per_label_total = Counter()

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += float(loss.item())
        batches += 1

        pred = torch.argmax(logits, dim=-1)

        correct += int((pred == y).sum().item())
        total += int(y.numel())

        for p, t in zip(pred.detach().cpu().tolist(), y.detach().cpu().tolist()):
            per_label_total[t] += 1
            if p == t:
                per_label_correct[t] += 1

    metrics = {
        "loss": total_loss / max(1, batches),
        "acc": correct / max(1, total),
    }

    for idx, label in enumerate(LABELS):
        metrics[f"{label}_acc"] = per_label_correct[idx] / max(1, per_label_total[idx])

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--project_limit", type=int, default=12000)
    parser.add_argument("--extra_normal", type=int, default=7000)
    parser.add_argument("--no_project_data", action="store_true")
    args = parser.parse_args()

    print("device:", DEVICE, flush=True)

    rows = []

    if not args.no_project_data:
        project_rows = load_project_normal_rows(limit=args.project_limit)
        rows.extend(project_rows)
        print("project normal rows:", len(project_rows), flush=True)

    synthetic_rows = build_synthetic_rows(extra_normal=args.extra_normal)
    rows.extend(synthetic_rows)
    print("synthetic rows:", len(synthetic_rows), flush=True)

    random.shuffle(rows)

    label_counts = Counter(r["label"] for r in rows)
    print("label counts:", dict(label_counts), flush=True)

    split = int(len(rows) * (1.0 - VAL_SPLIT))
    train_rows = rows[:split]
    val_rows = rows[split:] or rows[:]

    vocab = build_vocab(train_rows)
    label_map = {label: i for i, label in enumerate(LABELS)}

    train_ds = ComplexityDataset(train_rows, vocab, label_map)
    val_ds = ComplexityDataset(val_rows, vocab, label_map)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ComplexityClassifier(len(vocab), len(LABELS)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    best = math.inf

    print("rows:", len(rows), flush=True)
    print("vocab:", len(vocab), flush=True)
    print("labels:", LABELS, flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        batches = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            opt.zero_grad(set_to_none=True)

            logits = model(x)
            loss = loss_fn(logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            total_loss += float(loss.item())
            batches += 1

        metrics = evaluate(model, val_loader)
        train_loss = total_loss / max(1, batches)

        print(
            f"epoch {epoch:03d} | "
            f"train {train_loss:.4f} | "
            f"val {metrics['loss']:.4f} | "
            f"acc {metrics['acc']:.4f} | "
            f"normal {metrics['normal_qa_acc']:.3f} | "
            f"list {metrics['list_acc']:.3f} | "
            f"compare {metrics['compare_acc']:.3f} | "
            f"multi {metrics['multi_part_acc']:.3f} | "
            f"followup {metrics['followup_acc']:.3f}",
            flush=True,
        )

        if metrics["loss"] < best:
            best = metrics["loss"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": len(vocab),
                    "classes": len(LABELS),
                    "best_loss": best,
                },
                OUT_DIR / "complexity_classifier.pt",
            )

            save_json(OUT_DIR / "input_vocab.json", vocab)
            save_json(OUT_DIR / "labels.json", LABELS)
            save_json(
                OUT_DIR / "config.json",
                {
                    "model_type": "complexity_classifier",
                    "meaning": "Predicts routing complexity: normal/list/compare/multi/followup/smalltalk/unknown. Math is handled by the separate existing math classifier.",
                    "char_ngrams": list(CHAR_NGRAMS),
                    "word_ngrams": list(WORD_NGRAMS),
                    "hidden": HIDDEN,
                    "dropout": DROPOUT,
                },
            )

            print("[saved best]", flush=True)

    print("DONE:", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
