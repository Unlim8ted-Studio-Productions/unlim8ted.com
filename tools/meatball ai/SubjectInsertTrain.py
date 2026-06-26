import json
import re
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ============================================================
# CONFIG
# ============================================================

DATA_FILE = Path("assets/data/SubjectInserter.jsonl")
OUT_DIR = Path("assets/models/subject_inserter")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 1337
MAX_FEATURES = 18000
MAX_NGRAM = 3

BATCH_SIZE = 128
EPOCHS = 10
LR = 2e-3
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.12

HIDDEN = 256
DROPOUT = 0.20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================
# TEXT UTILS
# ============================================================


def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9_'?!.:,/ -]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text):
    return re.findall(r"[a-z0-9_']+|[?!.:,/]", normalize(text))


def ngrams(tokens):
    out = []
    for n in range(1, MAX_NGRAM + 1):
        for i in range(len(tokens) - n + 1):
            out.append("_".join(tokens[i : i + n]))
    return out


def featurize_text(message, subject):
    text = f"message: {message} subject: {subject}"
    return ngrams(tokenize(text))


# ============================================================
# LOAD DATA
# ============================================================

rows = []

with DATA_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        obj = json.loads(line)

        message = normalize(obj.get("message", ""))
        subject = normalize(obj.get("subject", ""))
        op = normalize(obj.get("op", ""))

        if not message or not op:
            continue

        rows.append(
            {
                "message": message,
                "subject": subject,
                "op": op,
                "target": normalize(obj.get("target", message)),
            }
        )

print("Loaded rows:", len(rows))

label_counts = Counter(r["op"] for r in rows)
print("Labels:")
for k, v in label_counts.most_common():
    print(k, v)

labels = sorted(label_counts.keys())
label_to_id = {label: i for i, label in enumerate(labels)}
id_to_label = {i: label for label, i in label_to_id.items()}

# ============================================================
# BUILD VOCAB
# ============================================================

feat_counts = Counter()

for r in rows:
    feat_counts.update(featurize_text(r["message"], r["subject"]))

most_common = feat_counts.most_common(MAX_FEATURES)
vocab = {feat: i for i, (feat, _) in enumerate(most_common)}

print("Vocab size:", len(vocab))
print("Num labels:", len(labels))

# ============================================================
# DATASET
# ============================================================


class InserterDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def encode(self, message, subject):
        x = torch.zeros(len(vocab), dtype=torch.float32)
        for feat in featurize_text(message, subject):
            idx = vocab.get(feat)
            if idx is not None:
                x[idx] = 1.0
        return x

    def __getitem__(self, idx):
        r = self.rows[idx]
        x = self.encode(r["message"], r["subject"])
        y = torch.tensor(label_to_id[r["op"]], dtype=torch.long)
        return x, y


dataset = InserterDataset(rows)

val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ============================================================
# MODEL
# ============================================================


class SubjectInserterNet(nn.Module):
    def __init__(self, input_dim, hidden, num_labels, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, x):
        return self.net(x)


model = SubjectInserterNet(
    input_dim=len(vocab), hidden=HIDDEN, num_labels=len(labels), dropout=DROPOUT
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# ============================================================
# TRAIN
# ============================================================


def evaluate():
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = criterion(logits, y)

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            loss_sum += loss.item() * y.size(0)

    return loss_sum / max(1, total), correct / max(1, total)


best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    total = 0

    for x, y in train_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        total_loss += loss.item() * y.size(0)
        total += y.size(0)

    val_loss, val_acc = evaluate()
    train_loss = total_loss / max(1, total)

    print(
        f"epoch {epoch:03d} | "
        f"train_loss {train_loss:.4f} | "
        f"val_loss {val_loss:.4f} | "
        f"val_acc {val_acc:.4f}"
    )

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), OUT_DIR / "subject_inserter.pt")

print("Best val acc:", best_acc)

# ============================================================
# SAVE CONFIG
# ============================================================

with (OUT_DIR / "vocab.json").open("w", encoding="utf-8") as f:
    json.dump(vocab, f, indent=2)

with (OUT_DIR / "labels.json").open("w", encoding="utf-8") as f:
    json.dump(labels, f, indent=2)

config = {
    "max_features": MAX_FEATURES,
    "max_ngram": MAX_NGRAM,
    "hidden": HIDDEN,
    "dropout": DROPOUT,
    "input_dim": len(vocab),
    "num_labels": len(labels),
    "model_type": "subject_inserter_classifier",
}

with (OUT_DIR / "config.json").open("w", encoding="utf-8") as f:
    json.dump(config, f, indent=2)

# ============================================================
# EXPORT ONNX
# ============================================================

model.load_state_dict(torch.load(OUT_DIR / "subject_inserter.pt", map_location=DEVICE))
model.eval()

dummy = torch.zeros(1, len(vocab), dtype=torch.float32).to(DEVICE)

onnx_path = OUT_DIR / "subject_inserter.onnx"

torch.onnx.export(
    model,
    dummy,
    onnx_path,
    input_names=["input"],
    output_names=["logits"],
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)

print("Saved:")
print(OUT_DIR / "subject_inserter.pt")
print(OUT_DIR / "subject_inserter.onnx")
print(OUT_DIR / "vocab.json")
print(OUT_DIR / "labels.json")
print(OUT_DIR / "config.json")

# ============================================================
# RUNTIME REWRITE FUNCTION
# ============================================================


def apply_op(message, subject, op):
    m = normalize(message)
    s = normalize(subject)

    if not s:
        return m

    if op == "already_standalone":
        return m

    if op == "no_rewrite":
        return m

    if op == "what_is_subject":
        return f"what is {s}"

    if op == "what_does_subject_do":
        return f"what does {s} do"

    if op == "who_made_subject":
        return f"who made {s}"

    if op == "where_is_subject":
        return f"where is {s}"

    if op == "when_was_subject":
        return f"when was {s}"

    if op == "does_subject_have":
        m2 = re.sub(r"\bdoes\s+it\b", f"does {s}", m)
        m2 = re.sub(r"\bdo\s+they\b", f"do {s}", m2)
        return m2

    if op == "is_subject":
        m2 = re.sub(r"\bis\s+it\b", f"is {s}", m)
        m2 = re.sub(r"\bare\s+they\b", f"are {s}", m2)
        return m2

    if op == "can_subject":
        return re.sub(r"\bcan\s+it\b|\bcan\s+they\b", f"can {s}", m)

    if op == "replace_it_with_subject":
        return re.sub(r"\bit\b", s, m)

    if op == "replace_they_with_subject":
        return re.sub(r"\bthey\b|\bthem\b", s, m)

    if op == "replace_this_with_subject":
        return re.sub(r"\bthis\b", s, m)

    if op == "replace_that_with_subject":
        return re.sub(r"\bthat\b", s, m)

    if op == "replace_he_she_with_subject":
        return re.sub(r"\bhe\b|\bshe\b|\bhim\b|\bher\b", s, m)

    if op == "append_about_subject":
        return f"{m} about {s}"

    if op == "append_for_subject":
        return f"{m} for {s}"

    return m


# ============================================================
# QUICK TEST
# ============================================================


def predict_op(message, subject):
    x = dataset.encode(message, subject).unsqueeze(0).to(DEVICE)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()

    return id_to_label[pred]


tests = [
    ("what is it", "the glitch"),
    ("tell me more", "timecat"),
    ("who made it", "life of a meatball"),
    ("does it have a shirt", "meatball"),
    ("is there a shirt", "meatball"),
    ("what is unlim8ted", "meatball"),
    ("cool", "walruses"),
]

print("\nQuick tests:")
for message, subject in tests:
    op = predict_op(message, subject)
    rewritten = apply_op(message, subject, op)
    print({"message": message, "subject": subject, "op": op, "rewritten": rewritten})
