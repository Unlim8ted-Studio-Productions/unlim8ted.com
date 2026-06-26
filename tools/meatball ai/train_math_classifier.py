import argparse, json, random, re, math
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHAR_NGRAMS = (2, 3, 4)
WORD_NGRAMS = (1, 2, 3)

MAX_VOCAB = 16000
BATCH_SIZE = 128
EPOCHS = 30
LR = 8e-4
WEIGHT_DECAY = 2e-3
DROPOUT = 0.25
HIDDEN = 384
VAL_SPLIT = 0.12

OUT_DIR = Path("assets/models/math_classifier")
MATH_DATA_DIR = Path("assets/data/math")
LABELS = ["general", "math"]


random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def load_jsonl(path):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize(text):
    text = str(text).lower().replace("\n", " ")
    text = re.sub(r"[^a-z0-9_+\-*/^().,?:;$%=\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def make_features(text):
    text = normalize(text)
    feats = []

    s = f"<{text}>"
    for n in CHAR_NGRAMS:
        for i in range(len(s) - n + 1):
            feats.append("c:" + s[i : i + n])

    words = text.split()
    for n in WORD_NGRAMS:
        for i in range(len(words) - n + 1):
            feats.append("w:" + "_".join(words[i : i + n]))

    return feats


def load_general_questions(specialized_dir, smart_qa_path):
    rows = []

    for path in sorted(Path(specialized_dir).glob("*.jsonl")):
        if path.stem.lower() == "math":
            continue
        rows.extend(load_jsonl(path))

    rows.extend(load_jsonl(smart_qa_path))

    out = []
    for r in rows:
        q = str(r.get("question", "")).strip()
        if q:
            out.append({"text": q, "label": 0})
    return out


def load_math_questions(math_dir):
    out = []
    for path in sorted(Path(math_dir).glob("*.jsonl")):
        for r in load_jsonl(path):
            q = str(r.get("question", "")).strip()
            if q:
                out.append({"text": q, "label": 1})
    return out


def dedupe_and_drop_conflicts(rows):
    by_text = {}
    conflicted = set()

    for row in rows:
        text = normalize(row["text"])
        if not text:
            continue
        label = int(row["label"])
        prev = by_text.get(text)
        if prev is None:
            by_text[text] = label
        elif prev != label:
            conflicted.add(text)

    out = []
    seen = set()
    for row in rows:
        text = normalize(row["text"])
        if not text or text in conflicted or text in seen:
            continue
        seen.add(text)
        out.append({"text": row["text"], "label": int(row["label"])})

    if conflicted:
        print(f"[dedupe] dropped conflicted texts: {len(conflicted)}", flush=True)
    print(f"[dedupe] kept unique texts: {len(out)}", flush=True)
    return out


def balance_rows(rows):
    grouped = {0: [], 1: []}
    for row in rows:
        grouped[int(row["label"])].append(row)

    if not grouped[0] or not grouped[1]:
        raise RuntimeError("Need both general and math rows to train the classifier.")

    target = min(len(grouped[0]), len(grouped[1]))
    rng = random.Random(SEED)
    balanced = rng.sample(grouped[0], target) + rng.sample(grouped[1], target)
    rng.shuffle(balanced)
    return balanced


def stratified_split(rows, val_split=VAL_SPLIT):
    grouped = {0: [], 1: []}
    for row in rows:
        grouped[int(row["label"])].append(row)

    rng = random.Random(SEED)
    train_rows = []
    val_rows = []

    for label_rows in grouped.values():
        rng.shuffle(label_rows)
        cut = int(len(label_rows) * (1 - val_split))
        if len(label_rows) > 1:
            cut = max(1, min(cut, len(label_rows) - 1))
        train_rows.extend(label_rows[:cut])
        val_rows.extend(label_rows[cut:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def add_synthetic_math(n=25000):
    rows = []
    ops = ["+", "-", "*", "/", "plus", "minus", "times", "divided by"]
    templates = [
        "what is {a} {op} {b}",
        "calculate {a} {op} {b}",
        "solve {a} {op} {b}",
        "if I have {a} apples and get {b} more how many",
        "what is {a} percent of {b}",
        "solve for x {a}x + {b} = {c}",
    ]

    for _ in range(n):
        a = random.randint(1, 200)
        b = random.randint(1, 200)
        c = random.randint(1, 500)
        op = random.choice(ops)
        t = random.choice(templates)
        rows.append({"text": t.format(a=a, b=b, c=c, op=op), "label": 1})
    return rows


def add_synthetic_general(n=25000):
    topics = [
        "dogs",
        "cats",
        "timecat",
        "the glitch",
        "unlim8ted",
        "meatball ai",
        "movies",
        "games",
    ]
    templates = [
        "what is {t}",
        "tell me about {t}",
        "facts about {t}",
        "who made {t}",
        "what are {t}",
        "hi",
        "how are you",
    ]
    return [
        {"text": random.choice(templates).format(t=random.choice(topics)), "label": 0}
        for _ in range(n)
    ]


def build_vocab(rows):
    c = Counter()
    for r in rows:
        c.update(make_features(r["text"]))

    vocab = {"<UNK>": 0}
    for feat, _ in c.most_common(MAX_VOCAB - 1):
        vocab[feat] = len(vocab)
    return vocab


def vectorize(text, vocab):
    x = torch.zeros(len(vocab), dtype=torch.float32)
    counts = Counter(make_features(text))
    for feat, count in counts.items():
        idx = vocab.get(feat, 0)
        x[idx] = min(float(count), 5.0)
    return x


class MathClassifierDataset(Dataset):
    def __init__(self, rows, vocab):
        self.rows = rows
        self.vocab = vocab

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return vectorize(r["text"], self.vocab), torch.tensor(
            r["label"], dtype=torch.long
        )


class MathClassifier(nn.Module):
    def __init__(self, input_size):
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
            nn.Linear(HIDDEN // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total = 0
    correct = 0
    loss_total = 0
    batches = 0
    per_label_total = [0, 0]
    per_label_correct = [0, 0]

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss_total += float(loss.item())
        batches += 1
        pred = torch.argmax(logits, dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        for label in range(len(LABELS)):
            mask = y == label
            if mask.any():
                per_label_total[label] += int(mask.sum().item())
                per_label_correct[label] += int((pred[mask] == y[mask]).sum().item())

    return {
        "loss": loss_total / max(1, batches),
        "acc": correct / max(1, total),
        "general_acc": per_label_correct[0] / max(1, per_label_total[0]),
        "math_acc": per_label_correct[1] / max(1, per_label_total[1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialized_dir", default="assets/data/specialized_QA")
    parser.add_argument("--smart_qa_path", default="tools/SmartMeatballQA.jsonl")
    parser.add_argument("--math_dir", default=str(MATH_DATA_DIR))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    print("device:", DEVICE, flush=True)

    rows = []
    rows.extend(load_general_questions(args.specialized_dir, args.smart_qa_path))
    rows.extend(load_math_questions(args.math_dir))
    rows.extend(add_synthetic_math())
    rows.extend(add_synthetic_general())

    rows = dedupe_and_drop_conflicts(rows)
    if args.limit:
        random.Random(SEED).shuffle(rows)
        rows = rows[: args.limit]

    rows = balance_rows(rows)

    print("rows:", len(rows), flush=True)
    print("math:", sum(r["label"] == 1 for r in rows), flush=True)
    print("general:", sum(r["label"] == 0 for r in rows), flush=True)

    train_rows, val_rows = stratified_split(rows)

    vocab = build_vocab(train_rows)

    train_loader = DataLoader(
        MathClassifierDataset(train_rows, vocab), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        MathClassifierDataset(val_rows, vocab), batch_size=BATCH_SIZE
    )

    model = MathClassifier(len(vocab)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best = math.inf
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0
        batches = 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item())
            batches += 1

        metrics = evaluate(model, val_loader, loss_fn)
        print(
            f"epoch {epoch:03d} | train {total/max(1,batches):.4f} | val {metrics['loss']:.4f} | acc {metrics['acc']:.4f} | general {metrics['general_acc']:.4f} | math {metrics['math_acc']:.4f}",
            flush=True,
        )

        if metrics["loss"] < best:
            best = metrics["loss"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": len(vocab),
                    "best_loss": best,
                },
                OUT_DIR / "math_classifier.pt",
            )
            save_json(OUT_DIR / "input_vocab.json", vocab)
            save_json(OUT_DIR / "labels.json", LABELS)
            save_json(
                OUT_DIR / "config.json",
                {
                    "model_type": "math_vs_general_classifier",
                    "char_ngrams": list(CHAR_NGRAMS),
                    "word_ngrams": list(WORD_NGRAMS),
                    "hidden": HIDDEN,
                    "dropout": DROPOUT,
                    "labels": LABELS,
                    "labels_path": str(OUT_DIR / "labels.json").replace("\\", "/"),
                },
            )
            print("[saved best]", flush=True)

    print("DONE", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()
