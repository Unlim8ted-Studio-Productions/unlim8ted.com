# train_autocorrect_entity_model.py
# Trains a tiny autocorrect/entity normalizer.
#
# Output:
# assets/models/autocorrect_entity/autocorrect_entity.pt
# assets/models/autocorrect_entity/input_vocab.json
# assets/models/autocorrect_entity/labels.json
# assets/models/autocorrect_entity/config.json

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

CHAR_NGRAMS = (2, 3, 4, 5)
WORD_NGRAMS = (1, 2, 3)

MAX_VOCAB = 20000
BATCH_SIZE = 128
EPOCHS = 40
PATIENCE = 8

LR = 8e-4
WEIGHT_DECAY = 2e-3
GRAD_CLIP = 1.0

HIDDEN = 512
DROPOUT = 0.25

VAL_SPLIT = 0.12

OUT_DIR = Path("assets/models/autocorrect_entity")

KNOWN_ENTITIES = {
    "Unlim8ted": [
        "unlim8ted",
        "unlimited",
        "unlimted",
        "unlim8ed",
        "unlimeted",
        "unlim8ted studios",
        "unlimited studios",
        "unlimted studios",
    ],
    "TimeCat": [
        "timecat",
        "time cat",
        "time-cat",
        "tmecat",
        "timect",
        "cat game",
        "the cat game",
    ],
    "The Glitch": [
        "the glitch",
        "glitch",
        "gltich",
        "glotch",
        "glitc",
        "the gltich",
    ],
    "Meatball": [
        "meatball",
        "meetball",
        "meat ball",
        "meatbal",
        "the meatball",
    ],
    "Meatball AI": [
        "meatball ai",
        "meatball bot",
        "meat ball ai",
        "meetball ai",
        "meatball assistant",
    ],
}

SMALL_WORD_FIXES = {
    "teh": "the",
    "thier": "their",
    "recieve": "receive",
    "wierd": "weird",
    "becuase": "because",
    "bc": "because",
    "b/c": "because",
    "pls": "please",
    "plz": "please",
    "wut": "what",
    "wat": "what",
    "whats": "what is",
    "ur": "your",
    "u": "you",
}


random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


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

            obj = json.loads(line)
            q = str(obj.get("question", "")).strip()

            if q:
                rows.append(q)

    return rows


def load_questions(specialized_dir, smart_qa_path):
    questions = []

    specialized_dir = Path(specialized_dir)
    for path in sorted(specialized_dir.glob("*.jsonl")):
        part = load_jsonl(path)
        questions.extend(part)
        print(f"{path.name}: {len(part)} questions")

    smart = load_jsonl(smart_qa_path)
    questions.extend(smart)
    print(f"{Path(smart_qa_path).name}: {len(smart)} questions")

    seen = set()
    out = []

    for q in questions:
        key = q.lower().strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(q.strip())

    return out


def normalize_spaces(text):
    return re.sub(r"\s+", " ", str(text).replace("\n", " ")).strip()


def basic_clean(text):
    text = str(text).lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[^a-z0-9_!?.,' -]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_punctuation(text):
    text = re.sub(r"[!?.,:;\"'`“”‘’()\[\]{}]", " ", str(text))
    return normalize_spaces(text)


def corrupt_chars(text):
    text = str(text)
    chars = []

    for ch in text:
        r = random.random()

        if ch.isalpha() and r < 0.025:
            continue

        if ch.isalpha() and 0.025 <= r < 0.045:
            chars.append(ch)
            chars.append(ch)
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
                "c": "v",
                "v": "c",
                "g": "h",
                "h": "g",
            }
            chars.append(subs.get(ch, ch))
            continue

        chars.append(ch)

    return "".join(chars)


def corrupt_question(clean):
    q = basic_clean(clean)

    for canonical, aliases in KNOWN_ENTITIES.items():
        canon_low = canonical.lower()
        if canon_low in q and random.random() < 0.8:
            q = q.replace(canon_low, random.choice(aliases))

    for wrong, right in SMALL_WORD_FIXES.items():
        if right in q and random.random() < 0.25:
            q = re.sub(rf"\b{re.escape(right)}\b", wrong, q)

    if random.random() < 0.75:
        q = remove_punctuation(q)

    if random.random() < 0.65:
        q = corrupt_chars(q)

    if random.random() < 0.35:
        toks = q.split()
        toks = [
            t
            for t in toks
            if not (
                t in {"the", "a", "an", "is", "are", "do", "does", "to", "of"}
                and random.random() < 0.45
            )
        ]
        q = " ".join(toks)

    if random.random() < 0.25:
        q += random.choice(["??", " pls", " lol", " rn", "!!!"])

    return normalize_spaces(q)


def canonicalize_text(text):
    q = basic_clean(text)

    for wrong, right in SMALL_WORD_FIXES.items():
        q = re.sub(rf"\b{re.escape(wrong)}\b", right, q)

    # longest aliases first
    alias_pairs = []
    for canonical, aliases in KNOWN_ENTITIES.items():
        for alias in aliases:
            alias_pairs.append((alias.lower(), canonical))

    alias_pairs.sort(key=lambda x: len(x[0]), reverse=True)

    for alias, canonical in alias_pairs:
        q = re.sub(rf"\b{re.escape(alias)}\b", canonical, q, flags=re.I)

    q = normalize_spaces(q)

    # light capitalization for canonical entities
    for canonical in KNOWN_ENTITIES:
        q = re.sub(rf"\b{re.escape(canonical.lower())}\b", canonical, q, flags=re.I)

    return q


def detect_entity(text):
    q = basic_clean(text)

    best = "NONE"
    best_len = 0

    for canonical, aliases in KNOWN_ENTITIES.items():
        terms = [canonical.lower()] + [a.lower() for a in aliases]
        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", q):
                if len(term) > best_len:
                    best = canonical
                    best_len = len(term)

    return best


def make_training_pairs(questions, augment_per_question=4):
    rows = []

    seed_questions = [
        "what is unlim8ted",
        "tell me about timecat",
        "what is the glitch",
        "who made meatball ai",
        "what is meatball",
        "does the glitch have clothes",
        "what is the cat game",
        "tell me about unlimited studios",
    ]

    all_questions = questions + seed_questions

    for clean in all_questions:
        clean_out = canonicalize_text(clean)
        entity = detect_entity(clean_out)

        rows.append(
            {
                "input": basic_clean(clean),
                "clean": clean_out,
                "entity": entity,
                "needs_correction": int(basic_clean(clean) != basic_clean(clean_out)),
            }
        )

        for _ in range(augment_per_question):
            noisy = corrupt_question(clean_out)
            fixed = canonicalize_text(noisy)
            entity = detect_entity(fixed)

            rows.append(
                {
                    "input": noisy,
                    "clean": fixed,
                    "entity": entity,
                    "needs_correction": int(basic_clean(noisy) != basic_clean(fixed)),
                }
            )

    # direct alias rows
    for canonical, aliases in KNOWN_ENTITIES.items():
        for alias in aliases:
            templates = [
                f"what is {alias}",
                f"tell me about {alias}",
                f"who made {alias}",
                f"does {alias} exist",
                f"info on {alias}",
            ]

            for t in templates:
                rows.append(
                    {
                        "input": t,
                        "clean": canonicalize_text(t),
                        "entity": canonical,
                        "needs_correction": int(alias.lower() != canonical.lower()),
                    }
                )

    random.shuffle(rows)
    return rows


def char_ngrams(text):
    s = f"<{basic_clean(text)}>"
    feats = []

    for n in CHAR_NGRAMS:
        for i in range(len(s) - n + 1):
            feats.append("char:" + s[i : i + n])

    return feats


def word_ngrams(text):
    toks = basic_clean(text).split()
    feats = []

    for n in WORD_NGRAMS:
        for i in range(len(toks) - n + 1):
            feats.append("word:" + "_".join(toks[i : i + n]))

    return feats


def make_features(text):
    return char_ngrams(text) + word_ngrams(text)


def build_vocab(rows):
    counter = Counter()

    for row in rows:
        counter.update(make_features(row["input"]))

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for feat, count in counter.most_common(MAX_VOCAB - len(vocab)):
        vocab[feat] = len(vocab)

    return vocab


def vectorize(text, vocab):
    x = torch.zeros(len(vocab), dtype=torch.float32)
    counts = Counter(make_features(text))
    unk = vocab["<UNK>"]

    for feat, count in counts.items():
        idx = vocab.get(feat, unk)
        x[idx] = min(float(count), 5.0)

    return x


class AutoCorrectDataset(Dataset):
    def __init__(self, rows, vocab, entity_to_id):
        self.rows = rows
        self.vocab = vocab
        self.entity_to_id = entity_to_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        x = vectorize(row["input"], self.vocab)

        y_entity = torch.tensor(
            self.entity_to_id[row["entity"]],
            dtype=torch.long,
        )

        y_needs = torch.tensor(
            float(row["needs_correction"]),
            dtype=torch.float32,
        )

        return x, y_entity, y_needs


class AutoCorrectEntityModel(nn.Module):
    def __init__(self, input_size, entity_classes):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_size, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN // 2),
            nn.LayerNorm(HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        self.entity_head = nn.Linear(HIDDEN // 2, entity_classes)
        self.needs_head = nn.Linear(HIDDEN // 2, 1)

    def forward(self, x):
        h = self.backbone(x)
        entity_logits = self.entity_head(h)
        needs_logits = self.needs_head(h).squeeze(-1)
        return entity_logits, needs_logits


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    entity_correct = 0
    entity_total = 0

    needs_correct = 0
    needs_total = 0

    total_loss = 0.0
    batches = 0

    entity_loss_fn = nn.CrossEntropyLoss()
    needs_loss_fn = nn.BCEWithLogitsLoss()

    for x, y_entity, y_needs in loader:
        x = x.to(DEVICE)
        y_entity = y_entity.to(DEVICE)
        y_needs = y_needs.to(DEVICE)

        entity_logits, needs_logits = model(x)

        entity_loss = entity_loss_fn(entity_logits, y_entity)
        needs_loss = needs_loss_fn(needs_logits, y_needs)
        loss = entity_loss + 0.25 * needs_loss

        total_loss += float(loss.item())
        batches += 1

        pred_entity = torch.argmax(entity_logits, dim=-1)
        entity_correct += int((pred_entity == y_entity).sum().item())
        entity_total += int(y_entity.numel())

        pred_needs = (torch.sigmoid(needs_logits) >= 0.5).float()
        needs_correct += int((pred_needs == y_needs).sum().item())
        needs_total += int(y_needs.numel())

    return {
        "loss": total_loss / max(1, batches),
        "entity_acc": entity_correct / max(1, entity_total),
        "needs_acc": needs_correct / max(1, needs_total),
    }


def apply_autocorrect(text, predicted_entity="NONE"):
    fixed = canonicalize_text(text)

    if predicted_entity != "NONE":
        # If the input was basically just asking about an alias,
        # force the canonical entity into the cleaned text.
        current = detect_entity(fixed)
        if current == "NONE":
            fixed = f"{fixed} {predicted_entity}"

    fixed = normalize_spaces(fixed)
    return fixed


def train(args):
    print("device:", DEVICE)

    questions = load_questions(args.specialized_dir, args.smart_qa_path)
    print("base questions:", len(questions))

    rows = make_training_pairs(
        questions,
        augment_per_question=args.augment_per_question,
    )

    if args.limit and len(rows) > args.limit:
        rows = rows[: args.limit]

    print("training rows:", len(rows))

    entities = ["NONE"] + sorted(KNOWN_ENTITIES.keys())
    entity_to_id = {e: i for i, e in enumerate(entities)}
    id_to_entity = {i: e for e, i in entity_to_id.items()}

    random.shuffle(rows)
    split = int(len(rows) * (1.0 - VAL_SPLIT))
    train_rows = rows[:split]
    val_rows = rows[split:] or rows[:]

    vocab = build_vocab(train_rows)

    train_ds = AutoCorrectDataset(train_rows, vocab, entity_to_id)
    val_ds = AutoCorrectDataset(val_rows, vocab, entity_to_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    model = AutoCorrectEntityModel(
        input_size=len(vocab),
        entity_classes=len(entities),
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    entity_loss_fn = nn.CrossEntropyLoss()
    needs_loss_fn = nn.BCEWithLogitsLoss()

    best_loss = math.inf
    bad_epochs = 0
    best_state = None

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("input vocab:", len(vocab))
    print("entities:", entities)

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        batches = 0

        for x, y_entity, y_needs in train_loader:
            x = x.to(DEVICE)
            y_entity = y_entity.to(DEVICE)
            y_needs = y_needs.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            entity_logits, needs_logits = model(x)

            entity_loss = entity_loss_fn(entity_logits, y_entity)
            needs_loss = needs_loss_fn(needs_logits, y_needs)

            loss = entity_loss + 0.25 * needs_loss

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += float(loss.item())
            batches += 1

        train_loss = total_loss / max(1, batches)
        metrics = evaluate(model, val_loader)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {metrics['loss']:.4f} | "
            f"entity_acc {metrics['entity_acc']:.4f} | "
            f"needs_acc {metrics['needs_acc']:.4f}"
        )

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            bad_epochs = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

            torch.save(
                {
                    "model_state_dict": best_state,
                    "input_size": len(vocab),
                    "entity_classes": len(entities),
                    "best_loss": best_loss,
                },
                OUT_DIR / "autocorrect_entity.pt",
            )

            print("[saved best]")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("[early stop]")
                break

    save_json(OUT_DIR / "input_vocab.json", vocab)
    save_json(OUT_DIR / "labels.json", entities)
    save_json(
        OUT_DIR / "config.json",
        {
            "model_type": "tiny_autocorrect_entity_classifier",
            "char_ngrams": list(CHAR_NGRAMS),
            "word_ngrams": list(WORD_NGRAMS),
            "max_vocab": MAX_VOCAB,
            "hidden": HIDDEN,
            "dropout": DROPOUT,
            "known_entities": KNOWN_ENTITIES,
            "small_word_fixes": SMALL_WORD_FIXES,
            "note": (
                "This model predicts canonical entity and whether correction is needed. "
                "Actual corrected text is rendered by deterministic rules."
            ),
        },
    )

    print("DONE")
    print("saved:", OUT_DIR)


def load_runtime(model_dir):
    model_dir = Path(model_dir)

    vocab = json.loads((model_dir / "input_vocab.json").read_text(encoding="utf-8"))
    labels = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
    ckpt = torch.load(model_dir / "autocorrect_entity.pt", map_location=DEVICE)

    model = AutoCorrectEntityModel(
        input_size=len(vocab),
        entity_classes=len(labels),
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return model, vocab, labels


@torch.no_grad()
def predict(text, model, vocab, labels):
    x = vectorize(text, vocab).unsqueeze(0).to(DEVICE)

    entity_logits, needs_logits = model(x)

    entity_id = int(torch.argmax(entity_logits, dim=-1).item())
    entity = labels[entity_id]
    entity_conf = float(torch.softmax(entity_logits, dim=-1)[0, entity_id].item())

    needs_prob = float(torch.sigmoid(needs_logits)[0].item())

    fixed = apply_autocorrect(text, entity if entity_conf >= 0.45 else "NONE")

    return {
        "input": text,
        "fixed": fixed,
        "entity": entity,
        "entity_conf": entity_conf,
        "needs_correction_prob": needs_prob,
    }


def interactive(args):
    model, vocab, labels = load_runtime(args.model_dir)

    print("Loaded:", args.model_dir)
    print("Device:", DEVICE)
    print("Vocab:", len(vocab))
    print("Labels:", labels)

    if args.text:
        result = predict(args.text, model, vocab, labels)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print("Interactive mode. Type quit / exit / stop.")
    while True:
        text = input("\nInput> ").strip()
        if text.lower() in {"quit", "exit", "stop"}:
            break
        if not text:
            continue

        result = predict(text, model, vocab, labels)
        print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode")

    train_p = sub.add_parser("train")
    train_p.add_argument("--specialized_dir", default="assets/data/specialized_QA")
    train_p.add_argument("--smart_qa_path", default="tools/SmartMeatballQA.jsonl")
    train_p.add_argument("--augment_per_question", type=int, default=4)
    train_p.add_argument("--epochs", type=int, default=EPOCHS)
    train_p.add_argument("--limit", type=int, default=0)

    test_p = sub.add_parser("test")
    test_p.add_argument("--model_dir", default=str(OUT_DIR))
    test_p.add_argument("--text", default=None)

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        interactive(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
