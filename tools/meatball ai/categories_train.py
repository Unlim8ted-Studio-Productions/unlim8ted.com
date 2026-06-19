import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = Path("assets/data/categories.jsonl")
FRAGMENTS_DIR = Path("assets/data/fragments")

OUT_DIR = Path("assets/models/category_router")

MODEL_PT = OUT_DIR / "category_router.pt"
MODEL_ONNX = OUT_DIR / "category_router.onnx"
VOCAB_JSON = OUT_DIR / "vocab.json"
LABELS_JSON = OUT_DIR / "labels.json"
CONFIG_JSON = OUT_DIR / "config.json"

COMMON_LABEL = "common"

MAX_VOCAB = 8000
MIN_TOKEN_FREQ = 1

BATCH_SIZE = 64

# Main training
MAIN_EPOCHS = 35

# Extra robustness stages
MISSPELL_EPOCHS = 6
NO_PUNCT_EPOCHS = 4
SLANG_EPOCHS = 6

PATIENCE = 7

LEARNING_RATE = 1e-3
AUG_LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3

HIDDEN_SIZE = 256
DROPOUT = 0.35

TRAIN_SPLIT = 0.90

# Prediction threshold saved into config.
# Test/browser should also use top-k + margin logic, not just raw threshold.
THRESHOLD = 0.55
TOP_K = 4
MARGIN = 0.18

SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# NORMALIZATION
# ============================================================


def normalize_label(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str):
    text = normalize_text(text)
    words = re.findall(r"[a-z0-9']+", text)

    tokens = []

    for word in words:
        tokens.append(word)

    for i in range(len(words) - 1):
        tokens.append(words[i] + "_" + words[i + 1])

    for i in range(len(words) - 2):
        tokens.append(words[i] + "_" + words[i + 1] + "_" + words[i + 2])

    return tokens


# ============================================================
# AUGMENTATION
# ============================================================

SLANG_REPLACEMENTS = {
    "what is": ["what's", "whats"],
    "what are": ["what're", "what are"],
    "tell me about": ["tell me abt", "tell me bout"],
    "explain": ["explain", "explain pls", "can u explain"],
    "because": ["bc", "because"],
    "you": ["u", "you"],
    "your": ["ur", "your"],
    "please": ["pls", "please"],
    "about": ["abt", "about"],
    "and": ["and", "n"],
}


def remove_punctuation(text: str) -> str:
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def randomly_misspell_word(word: str) -> str:
    if len(word) < 4:
        return word

    operation = random.choice(["drop", "swap", "double"])

    chars = list(word)

    if operation == "drop" and len(chars) > 3:
        idx = random.randint(1, len(chars) - 2)
        del chars[idx]

    elif operation == "swap" and len(chars) > 4:
        idx = random.randint(1, len(chars) - 3)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]

    elif operation == "double":
        idx = random.randint(1, len(chars) - 2)
        chars.insert(idx, chars[idx])

    return "".join(chars)


def misspell_text(text: str, rate: float = 0.16) -> str:
    parts = text.split()

    new_parts = []

    for word in parts:
        clean = re.sub(r"[^A-Za-z0-9']", "", word)

        if clean and random.random() < rate:
            new_word = randomly_misspell_word(clean)
            word = word.replace(clean, new_word, 1)

        new_parts.append(word)

    return " ".join(new_parts)


def slangify_text(text: str) -> str:
    lower = text.lower()

    for src, replacements in SLANG_REPLACEMENTS.items():
        if src in lower and random.random() < 0.55:
            lower = lower.replace(src, random.choice(replacements))

    if random.random() < 0.30:
        lower = lower.replace("?", "")

    if random.random() < 0.25:
        lower = lower + " lol"

    if random.random() < 0.20:
        lower = lower + " pls"

    return re.sub(r"\s+", " ", lower).strip()


def augment_text(text: str, mode: str) -> str:
    if mode == "none":
        return text

    if mode == "misspell":
        return misspell_text(text)

    if mode == "no_punct":
        return remove_punctuation(text)

    if mode == "slang":
        return slangify_text(text)

    return text


# ============================================================
# DATA LOADING
# ============================================================


def get_valid_fragment_labels():
    labels = set()

    if not FRAGMENTS_DIR.exists():
        raise FileNotFoundError(f"Fragments dir not found: {FRAGMENTS_DIR}")

    for path in FRAGMENTS_DIR.glob("*.jsonl"):
        label = normalize_label(path.stem)

        if label == COMMON_LABEL:
            continue

        labels.add(label)

    return labels


def row_to_text(row: dict) -> str:
    text = row.get("input", row.get("question", ""))

    history = row.get("history", [])

    if isinstance(history, list) and history:
        history_text = " ".join(str(x) for x in history)
        text = history_text + " " + str(text)

    return str(text).strip()


def row_to_labels(row: dict):
    labels = row.get("categories", row.get("topics", []))

    if not isinstance(labels, list):
        labels = []

    clean = []

    for label in labels:
        label = normalize_label(label)

        if not label or label == COMMON_LABEL:
            continue

        if label not in clean:
            clean.append(label)

    return clean


def load_rows():
    valid_labels = get_valid_fragment_labels()

    rows = []
    removed = []
    bad_json = 0

    with DATASET_PATH.open("r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, start=1):
            raw = line.strip()

            if not raw:
                continue

            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                bad_json += 1
                removed.append(
                    {
                        "line": line_num,
                        "reason": "bad_json",
                        "raw": raw,
                    }
                )
                continue

            text = row_to_text(row)
            labels = row_to_labels(row)

            invalid = [label for label in labels if label not in valid_labels]
            labels = [label for label in labels if label in valid_labels]

            if invalid:
                removed.append(
                    {
                        "line": line_num,
                        "reason": "invalid_labels",
                        "invalid_labels": invalid,
                        "row": row,
                    }
                )

            if not text or not labels:
                removed.append(
                    {
                        "line": line_num,
                        "reason": "empty_text_or_no_valid_labels",
                        "row": row,
                    }
                )
                continue

            rows.append(
                {
                    "text": text,
                    "labels": labels,
                }
            )

    print(f"Loaded usable rows: {len(rows)}")
    print(f"Removed/problem rows: {len(removed)}")
    print(f"Bad JSON lines: {bad_json}")
    print(f"Valid fragment labels: {len(valid_labels)}")

    return rows, removed, valid_labels


# ============================================================
# VECTORIZATION
# ============================================================


def build_vocab(rows):
    counts = Counter()

    for row in rows:
        counts.update(tokenize(row["text"]))

    vocab_tokens = [
        token
        for token, count in counts.most_common(MAX_VOCAB)
        if count >= MIN_TOKEN_FREQ
    ]

    return {token: idx for idx, token in enumerate(vocab_tokens)}


def build_labels(rows):
    return sorted({label for row in rows for label in row["labels"]})


def vectorize_text(text: str, vocab: dict):
    x = np.zeros(len(vocab), dtype=np.float32)

    for token in tokenize(text):
        idx = vocab.get(token)

        if idx is not None:
            x[idx] += 1.0

    return np.log1p(x)


def vectorize_labels(labels, label_to_idx):
    y = np.zeros(len(label_to_idx), dtype=np.float32)

    for label in labels:
        idx = label_to_idx.get(label)

        if idx is not None:
            y[idx] = 1.0

    return y


# ============================================================
# DATASET
# ============================================================


class RouterDataset(Dataset):
    def __init__(self, rows, vocab, label_to_idx, augment_mode="none"):
        self.rows = rows
        self.vocab = vocab
        self.label_to_idx = label_to_idx
        self.augment_mode = augment_mode

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        text = augment_text(row["text"], self.augment_mode)

        x = vectorize_text(text, self.vocab)
        y = vectorize_labels(row["labels"], self.label_to_idx)

        return torch.tensor(x), torch.tensor(y)


# ============================================================
# MODEL
# ============================================================


class CategoryRouter(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


def calculate_pos_weight(rows, label_to_idx):
    counts = np.zeros(len(label_to_idx), dtype=np.float32)

    for row in rows:
        for label in row["labels"]:
            counts[label_to_idx[label]] += 1.0

    total = len(rows)
    weights = []

    for c in counts:
        negative = total - c
        positive = max(c, 1.0)
        weights.append(min(negative / positive, 12.0))

    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, loss_fn):
    model.eval()

    total_loss = 0.0
    total = 0

    exact = 0
    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = loss_fn(logits, y)

            probs = torch.sigmoid(logits)
            pred = probs >= THRESHOLD

            y_bool = y >= 0.5

            exact += (pred == y_bool).all(dim=1).sum().item()

            micro_tp += (pred & y_bool).sum().item()
            micro_fp += (pred & ~y_bool).sum().item()
            micro_fn += (~pred & y_bool).sum().item()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

    precision = micro_tp / max(micro_tp + micro_fp, 1)
    recall = micro_tp / max(micro_tp + micro_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "loss": total_loss / max(total, 1),
        "exact": exact / max(total, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================
# TRAINING
# ============================================================


def train_stage(
    model,
    train_rows,
    val_loader,
    vocab,
    label_to_idx,
    loss_fn,
    stage_name,
    augment_mode,
    epochs,
    lr,
    patience=None,
):
    train_ds = RouterDataset(
        train_rows,
        vocab,
        label_to_idx,
        augment_mode=augment_mode,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=WEIGHT_DECAY,
    )

    best_f1 = -1.0
    best_state = None
    stale = 0

    print()
    print(f"==================== {stage_name} ====================")

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            logits = model(x)
            loss = loss_fn(logits, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total += x.size(0)

        train_loss = total_loss / max(total, 1)
        val = evaluate(model, val_loader, loss_fn)

        print(
            f"{stage_name} epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val['loss']:.4f} | "
            f"exact={val['exact']:.3f} | "
            f"precision={val['precision']:.3f} | "
            f"recall={val['recall']:.3f} | "
            f"f1={val['f1']:.3f}"
        )

        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            best_state = {
                "model": {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                },
                "epoch": epoch,
                "f1": best_f1,
            }
            stale = 0
        else:
            stale += 1

        if patience is not None and stale >= patience:
            print(
                f"Early stopping {stage_name} at epoch {epoch}. "
                f"Best epoch={best_state['epoch']} f1={best_f1:.3f}"
            )
            break

    if best_state is not None:
        model.load_state_dict(best_state["model"])

    return best_f1


# ============================================================
# SAVE / EXPORT
# ============================================================


def save_model(model, vocab, labels):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "input_size": len(vocab),
        "hidden_size": HIDDEN_SIZE,
        "output_size": len(labels),
        "dropout": DROPOUT,
        "threshold": THRESHOLD,
        "top_k": TOP_K,
        "margin": MARGIN,
        "tokenizer": "lowercase words + bigrams + trigrams, log1p bag of words",
    }

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vocab,
            "labels": labels,
            "config": config,
        },
        MODEL_PT,
    )

    with VOCAB_JSON.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    with LABELS_JSON.open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    with CONFIG_JSON.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    model.eval()

    dummy = torch.zeros(1, len(vocab), dtype=torch.float32).to(DEVICE)

    torch.onnx.export(
        model,
        dummy,
        MODEL_ONNX,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )

    print()
    print("Saved:")
    print(MODEL_PT)
    print(MODEL_ONNX)
    print(VOCAB_JSON)
    print(LABELS_JSON)
    print(CONFIG_JSON)


# ============================================================
# MAIN
# ============================================================


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    rows, removed, valid_labels = load_rows()

    random.shuffle(rows)

    vocab = build_vocab(rows)
    labels = build_labels(rows)
    label_to_idx = {label: i for i, label in enumerate(labels)}

    print(f"Vocab size: {len(vocab)}")
    print(f"Labels used by model: {len(labels)}")

    split = int(len(rows) * TRAIN_SPLIT)

    train_rows = rows[:split]
    val_rows = rows[split:]

    val_ds = RouterDataset(
        val_rows,
        vocab,
        label_to_idx,
        augment_mode="none",
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = CategoryRouter(
        input_size=len(vocab),
        hidden_size=HIDDEN_SIZE,
        output_size=len(labels),
        dropout=DROPOUT,
    ).to(DEVICE)

    pos_weight = calculate_pos_weight(train_rows, label_to_idx).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_stage(
        model=model,
        train_rows=train_rows,
        val_loader=val_loader,
        vocab=vocab,
        label_to_idx=label_to_idx,
        loss_fn=loss_fn,
        stage_name="MAIN",
        augment_mode="none",
        epochs=MAIN_EPOCHS,
        lr=LEARNING_RATE,
        patience=PATIENCE,
    )

    train_stage(
        model=model,
        train_rows=train_rows,
        val_loader=val_loader,
        vocab=vocab,
        label_to_idx=label_to_idx,
        loss_fn=loss_fn,
        stage_name="MISSPELLING",
        augment_mode="misspell",
        epochs=MISSPELL_EPOCHS,
        lr=AUG_LEARNING_RATE,
        patience=None,
    )

    train_stage(
        model=model,
        train_rows=train_rows,
        val_loader=val_loader,
        vocab=vocab,
        label_to_idx=label_to_idx,
        loss_fn=loss_fn,
        stage_name="PUNCTUATION_REMOVAL",
        augment_mode="no_punct",
        epochs=NO_PUNCT_EPOCHS,
        lr=AUG_LEARNING_RATE,
        patience=None,
    )

    train_stage(
        model=model,
        train_rows=train_rows,
        val_loader=val_loader,
        vocab=vocab,
        label_to_idx=label_to_idx,
        loss_fn=loss_fn,
        stage_name="SLANG",
        augment_mode="slang",
        epochs=SLANG_EPOCHS,
        lr=AUG_LEARNING_RATE,
        patience=None,
    )

    save_model(model, vocab, labels)


if __name__ == "__main__":
    main()
