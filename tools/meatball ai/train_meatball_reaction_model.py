# train_meatball_reaction_model.py

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

OUT_DIR = Path("assets/models/meatball_reaction_model")

REACTIONS = [
    "neutral",
    "excited",
    "confused",
    "suspicious",
    "angry",
    "sad",
    "overwhelmed",
]

CHAR_NGRAMS = (2, 3, 4, 5)
WORD_NGRAMS = (1, 2, 3)

MAX_VOCAB = 12000
HIDDEN = 256
DROPOUT = 0.2
BATCH_SIZE = 128
EPOCHS = 35
VAL_SPLIT = 0.12
LR = 8e-4
WEIGHT_DECAY = 2e-3
GRAD_CLIP = 1.0

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def normalize(text):
    text = str(text).lower().replace("\n", " ")
    text = re.sub(r"[^a-z0-9!?.,' +\-*/=()%$]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def corrupt(text):
    text = normalize(text)
    out = []
    for ch in text:
        r = random.random()
        if ch.isalpha() and r < 0.025:
            continue
        if ch.isalpha() and 0.025 <= r < 0.045:
            out.append(ch)
            out.append(ch)
            continue
        if ch.isalpha() and 0.045 <= r < 0.070:
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
    return re.sub(r"\s+", " ", "".join(out)).strip()


def add(rows, reaction, examples, augment=10):
    for text in examples:
        rows.append({"text": text, "reaction": reaction})
        for _ in range(augment):
            rows.append({"text": corrupt(text), "reaction": reaction})


def build_rows(extra_neutral=5000):
    rows = []

    add(
        rows,
        "angry",
        [
            "i hate you",
            "you are stupid",
            "youre stupid",
            "your stupid",
            "you suck",
            "shut up",
            "dumb meatball",
            "stupid sauce",
            "your stupii",
            "you are useless",
            "that is wrong",
            "no that is wrong",
            "stop being dumb",
            "be angry",
            "get angry",
            "rage mode",
            "sauce attack",
            "do the forbidden pasta technique",
        ],
        augment=18,
    )

    add(
        rows,
        "sad",
        [
            "i am sad",
            "this is sad",
            "that made me sad",
            "i feel bad",
            "i am lonely",
            "this hurts",
            "soft answer please",
            "sad sauce",
            "i miss it",
            "that is depressing",
            "why is this so sad",
            "be sad",
        ],
        augment=14,
    )

    add(
        rows,
        "confused",
        [
            "what",
            "huh",
            "what do you mean",
            "what does that mean",
            "i dont get it",
            "that makes no sense",
            "???",
            "asdf qwer",
            "whaatat ararercts",
            "im confused",
            "confused",
            "why",
            "explain that",
        ],
        augment=18,
    )

    add(
        rows,
        "overwhelmed",
        [
            "too much",
            "that is too much",
            "slow down",
            "i cant handle this",
            "ahhhhh",
            "too many things",
            "this is overwhelming",
            "stop too much",
            "my brain exploded",
            "everything at once",
            "compare cats and dogs and solve 1+1 and list facts",
            "too many signals",
        ],
        augment=14,
    )

    add(
        rows,
        "excited",
        [
            "yes",
            "awesome",
            "thats cool",
            "amazing",
            "lets go",
            "this is exciting",
            "i love it",
            "great",
            "perfect",
            "nice",
            "that is so cool",
            "woah",
            "wow",
        ],
        augment=14,
    )

    add(
        rows,
        "suspicious",
        [
            "hmm",
            "that seems weird",
            "i dont trust it",
            "suspicious",
            "that signal is weird",
            "is that true",
            "are you sure",
            "something is off",
            "the glitch is watching",
            "what is the glitch",
            "tell me about the glitch",
            "is the sauce lying",
        ],
        augment=14,
    )

    add(
        rows,
        "neutral",
        [
            "hi",
            "hello",
            "what is timecat",
            "facts about dogs",
            "tell me about the glitch",
            "who made meatball ai",
            "what are cats",
            "what is seven plus eight",
            "explain unlim8ted",
            "does it have lore",
            "tell a joke",
            "what is a subway train",
        ],
        augment=10,
    )

    subjects = [
        "dogs",
        "cats",
        "timecat",
        "the glitch",
        "unlim8ted",
        "meatball ai",
        "subway trains",
        "cars",
        "trees",
        "games",
        "movies",
        "space",
    ]
    templates = [
        "what is {s}",
        "tell me about {s}",
        "facts about {s}",
        "who made {s}",
        "does {s} have lore",
        "what are {s}",
        "explain {s}",
        "how does {s} work",
    ]

    for _ in range(extra_neutral):
        t = random.choice(templates).format(s=random.choice(subjects))
        if random.random() < 0.35:
            t = corrupt(t)
        rows.append({"text": t, "reaction": "neutral"})

    random.shuffle(rows)
    return rows


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

    if not text:
        feats.append("flag:empty")
    if "?" in text:
        feats.append("flag:question")
    if "!" in text:
        feats.append("flag:bang")
    if re.search(r"(.)\1\1", text):
        feats.append("flag:repeated_chars")
    if re.search(r"\d", text):
        feats.append("flag:number")
    if re.search(r"[+\-*/=]", text):
        feats.append("flag:operator")

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


class ReactionDataset(Dataset):
    def __init__(self, rows, vocab, label_map):
        self.rows = rows
        self.vocab = vocab
        self.label_map = label_map

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return vectorize(r["text"], self.vocab), torch.tensor(
            self.label_map[r["reaction"]], dtype=torch.long
        )


class ReactionModel(nn.Module):
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

    return total_loss / max(1, batches), correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--extra_neutral", type=int, default=5000)
    args = parser.parse_args()

    print("device:", DEVICE, flush=True)

    rows = build_rows(extra_neutral=args.extra_neutral)
    random.shuffle(rows)

    split = int(len(rows) * (1 - VAL_SPLIT))
    train_rows = rows[:split]
    val_rows = rows[split:] or rows[:]

    vocab = build_vocab(train_rows)
    label_map = {x: i for i, x in enumerate(REACTIONS)}

    train_loader = DataLoader(
        ReactionDataset(train_rows, vocab, label_map),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        ReactionDataset(val_rows, vocab, label_map), batch_size=BATCH_SIZE
    )

    model = ReactionModel(len(vocab), len(REACTIONS)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    best = math.inf
    print("rows:", len(rows), flush=True)
    print("vocab:", len(vocab), flush=True)

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

        val_loss, val_acc = evaluate(model, val_loader)

        print(
            f"epoch {epoch:03d} | train {total_loss/max(1,batches):.4f} | "
            f"val {val_loss:.4f} | acc {val_acc:.4f}",
            flush=True,
        )

        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": len(vocab),
                    "classes": len(REACTIONS),
                    "best_loss": best,
                },
                OUT_DIR / "reaction_model.pt",
            )
            save_json(OUT_DIR / "input_vocab.json", vocab)
            save_json(OUT_DIR / "labels.json", REACTIONS)
            save_json(
                OUT_DIR / "config.json",
                {
                    "model_type": "meatball_reaction_model",
                    "meaning": "Predicts what Meatball should feel/show in response to the message, not the user's emotion.",
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
