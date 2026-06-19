import json
import random
import re
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = Path("assets/data/ContextWatcherMeatball.jsonl")

OUTPUT_DIR = Path("assets/models/context_watcher")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PT_PATH = OUTPUT_DIR / "context_watcher.pt"
MODEL_ONNX_PATH = OUTPUT_DIR / "context_watcher.onnx"

VOCAB_PATH = OUTPUT_DIR / "vocab.json"
LABELS_PATH = OUTPUT_DIR / "labels.json"
CONFIG_PATH = OUTPUT_DIR / "config.json"

SEED = 42

VAL_SPLIT = 0.12

NGRAMS = (1, 2, 3)

MAX_VOCAB_SIZE = 12000
MIN_TOKEN_FREQ = 1

BATCH_SIZE = 64
EPOCHS = 50
PATIENCE = 7
MIN_DELTA = 1e-4

LR = 8e-4
WEIGHT_DECAY = 2e-3

HIDDEN_SIZE = 192
DROPOUT = 0.35

GRAD_CLIP = 1.0

FOLLOWUP_LOSS_WEIGHT = 1.0
SUBJECT_LOSS_WEIGHT = 1.2
DOMAIN_LOSS_WEIGHT = 1.0
INTENT_LOSS_WEIGHT = 1.0
HINTS_LOSS_WEIGHT = 1.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# ALLOWED LABELS
# ============================================================

ALLOWED_SUBJECTS = [
    "",
    "unlim8ted",
    "the_glitch",
    "life_of_a_meatball",
    "timecat",
    "star_tracker",
    "meatball_ai",
    "unlim8ted_products",
    "unlim8ted_music",
    "unlim8ted_films",
    "unlim8ted_games",
    "unlim8ted_software",
    "unlim8ted_hardware",
    "general_question",
    "unknown",
]

ALLOWED_DOMAINS = [
    "",
    "identity",
    "story",
    "film",
    "game",
    "music",
    "software",
    "hardware",
    "products",
    "clothing",
    "availability",
    "creator",
    "meaning",
    "comparison",
    "recommendation",
    "smalltalk",
    "fallback",
    "general_knowledge",
]

ALLOWED_INTENTS = [
    "",
    "define",
    "explain",
    "summarize",
    "ask_availability",
    "ask_best",
    "ask_creator",
    "ask_story",
    "ask_product",
    "ask_project",
    "ask_difference",
    "ask_followup",
    "greeting",
    "reaction",
    "fallback",
    "unknown",
]

ALLOWED_TOPIC_HINTS = [
    "unlim8ted",
    "the_glitch",
    "life_of_a_meatball",
    "timecat",
    "star_tracker",
    "meatball_ai",
    "unlim8ted_products",
    "unlim8ted_clothing",
    "unlim8ted_music",
    "unlim8ted_films",
    "unlim8ted_games",
    "unlim8ted_software",
    "unlim8ted_hardware",
    "story",
    "film_story",
    "game_project",
    "app_project",
    "physical_product",
    "availability",
    "smalltalk",
    "fallback",
    "general_knowledge",
]


# ============================================================
# RANDOM
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================
# TEXT UTILS
# ============================================================


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[^a-z0-9_!?.,' -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    text = normalize_text(text)

    if not text:
        return []

    return text.split()


def make_ngrams(tokens, ngrams=NGRAMS):
    feats = []

    for n in ngrams:
        if len(tokens) < n:
            continue

        for i in range(len(tokens) - n + 1):
            feats.append("_".join(tokens[i : i + n]))

    return feats


def history_to_text(history):
    if history is None:
        return ""

    if isinstance(history, list):
        return " ".join(str(x) for x in history)

    return str(history)


def row_to_input_text(row):
    message = row.get("message", "")
    history = history_to_text(row.get("history", []))

    return f"message: {message} history: {history}"


# ============================================================
# DATA LOADING
# ============================================================


def load_jsonl(path: Path):
    rows = []

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[skip] bad JSON line {line_num}: {e}")
                continue

            if "message" not in row:
                print(f"[skip] missing message line {line_num}")
                continue

            if "context" not in row or not isinstance(row["context"], dict):
                print(f"[skip] missing context line {line_num}")
                continue

            rows.append(row)

    return rows


# ============================================================
# VALIDATION / CLEANING
# ============================================================


def clean_context(ctx):
    is_followup = bool(ctx.get("is_followup", False))

    subject = str(ctx.get("current_subject", "")).strip()
    domain = str(ctx.get("current_domain", "")).strip()
    intent = str(ctx.get("current_intent", "")).strip()

    if subject not in ALLOWED_SUBJECTS:
        subject = "unknown"

    if domain not in ALLOWED_DOMAINS:
        domain = "fallback"

    if intent not in ALLOWED_INTENTS:
        intent = "unknown"

    hints = ctx.get("topic_hints", [])

    if not isinstance(hints, list):
        hints = []

    clean_hints = []

    for h in hints:
        h = str(h).strip()
        if h in ALLOWED_TOPIC_HINTS and h not in clean_hints:
            clean_hints.append(h)

    clean_hints = clean_hints[:5]

    return {
        "is_followup": is_followup,
        "current_subject": subject,
        "current_domain": domain,
        "current_intent": intent,
        "topic_hints": clean_hints,
    }


def clean_rows(rows):
    cleaned = []

    for row in rows:
        new_row = dict(row)
        new_row["context"] = clean_context(row.get("context", {}))
        cleaned.append(new_row)

    return cleaned


# ============================================================
# VOCAB
# ============================================================


def build_vocab(rows):
    counter = Counter()

    for row in rows:
        text = row_to_input_text(row)
        tokens = tokenize(text)
        feats = make_ngrams(tokens)
        counter.update(feats)

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for token, count in counter.most_common(MAX_VOCAB_SIZE - len(vocab)):
        if count < MIN_TOKEN_FREQ:
            continue

        if token not in vocab:
            vocab[token] = len(vocab)

    return vocab


def vectorize_input(row, vocab):
    text = row_to_input_text(row)
    tokens = tokenize(text)
    feats = make_ngrams(tokens)

    x = torch.zeros(len(vocab), dtype=torch.float32)

    counts = Counter(feats)
    unk_id = vocab.get("<UNK>", 1)

    for feat, count in counts.items():
        idx = vocab.get(feat, unk_id)
        x[idx] = min(float(count), 5.0)

    return x


# ============================================================
# LABEL MAPS
# ============================================================


def build_label_maps():
    labels = {
        "subjects": ALLOWED_SUBJECTS,
        "domains": ALLOWED_DOMAINS,
        "intents": ALLOWED_INTENTS,
        "topic_hints": ALLOWED_TOPIC_HINTS,
    }

    maps = {
        "subject_to_id": {x: i for i, x in enumerate(ALLOWED_SUBJECTS)},
        "domain_to_id": {x: i for i, x in enumerate(ALLOWED_DOMAINS)},
        "intent_to_id": {x: i for i, x in enumerate(ALLOWED_INTENTS)},
        "hint_to_id": {x: i for i, x in enumerate(ALLOWED_TOPIC_HINTS)},
    }

    return labels, maps


# ============================================================
# DATASET
# ============================================================


class ContextWatcherDataset(Dataset):
    def __init__(self, rows, vocab, maps):
        self.rows = rows
        self.vocab = vocab
        self.maps = maps

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        ctx = row["context"]

        x = vectorize_input(row, self.vocab)

        y_followup = torch.tensor(
            [1.0 if ctx["is_followup"] else 0.0],
            dtype=torch.float32,
        )

        y_subject = torch.tensor(
            self.maps["subject_to_id"].get(
                ctx["current_subject"], self.maps["subject_to_id"]["unknown"]
            ),
            dtype=torch.long,
        )

        y_domain = torch.tensor(
            self.maps["domain_to_id"].get(
                ctx["current_domain"], self.maps["domain_to_id"]["fallback"]
            ),
            dtype=torch.long,
        )

        y_intent = torch.tensor(
            self.maps["intent_to_id"].get(
                ctx["current_intent"], self.maps["intent_to_id"]["unknown"]
            ),
            dtype=torch.long,
        )

        y_hints = torch.zeros(len(ALLOWED_TOPIC_HINTS), dtype=torch.float32)

        for hint in ctx["topic_hints"]:
            if hint in self.maps["hint_to_id"]:
                y_hints[self.maps["hint_to_id"][hint]] = 1.0

        return x, y_followup, y_subject, y_domain, y_intent, y_hints


# ============================================================
# MODEL
# ============================================================


class ContextWatcherNet(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_subjects,
        num_domains,
        num_intents,
        num_hints,
        dropout,
    ):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.followup_head = nn.Linear(hidden_size, 1)
        self.subject_head = nn.Linear(hidden_size, num_subjects)
        self.domain_head = nn.Linear(hidden_size, num_domains)
        self.intent_head = nn.Linear(hidden_size, num_intents)
        self.hints_head = nn.Linear(hidden_size, num_hints)

    def forward(self, x):
        h = self.backbone(x)

        return {
            "followup": self.followup_head(h),
            "subject": self.subject_head(h),
            "domain": self.domain_head(h),
            "intent": self.intent_head(h),
            "hints": self.hints_head(h),
        }


class OnnxWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)

        return (
            out["followup"],
            out["subject"],
            out["domain"],
            out["intent"],
            out["hints"],
        )


# ============================================================
# METRICS
# ============================================================


@torch.no_grad()
def evaluate(model, loader, losses):
    model.eval()

    total_loss = 0.0
    batches = 0

    followup_correct = 0
    followup_total = 0

    subject_correct = 0
    domain_correct = 0
    intent_correct = 0
    class_total = 0

    hint_tp = 0
    hint_fp = 0
    hint_fn = 0

    for batch in loader:
        x, y_followup, y_subject, y_domain, y_intent, y_hints = batch

        x = x.to(DEVICE)
        y_followup = y_followup.to(DEVICE)
        y_subject = y_subject.to(DEVICE)
        y_domain = y_domain.to(DEVICE)
        y_intent = y_intent.to(DEVICE)
        y_hints = y_hints.to(DEVICE)

        out = model(x)

        loss_followup = losses["bce"](out["followup"], y_followup)
        loss_subject = losses["ce"](out["subject"], y_subject)
        loss_domain = losses["ce"](out["domain"], y_domain)
        loss_intent = losses["ce"](out["intent"], y_intent)
        loss_hints = losses["bce"](out["hints"], y_hints)

        loss = (
            FOLLOWUP_LOSS_WEIGHT * loss_followup
            + SUBJECT_LOSS_WEIGHT * loss_subject
            + DOMAIN_LOSS_WEIGHT * loss_domain
            + INTENT_LOSS_WEIGHT * loss_intent
            + HINTS_LOSS_WEIGHT * loss_hints
        )

        total_loss += float(loss.item())
        batches += 1

        pred_followup = (torch.sigmoid(out["followup"]) >= 0.5).float()
        followup_correct += int((pred_followup == y_followup).sum().item())
        followup_total += int(y_followup.numel())

        pred_subject = torch.argmax(out["subject"], dim=-1)
        pred_domain = torch.argmax(out["domain"], dim=-1)
        pred_intent = torch.argmax(out["intent"], dim=-1)

        subject_correct += int((pred_subject == y_subject).sum().item())
        domain_correct += int((pred_domain == y_domain).sum().item())
        intent_correct += int((pred_intent == y_intent).sum().item())
        class_total += int(y_subject.numel())

        pred_hints = (torch.sigmoid(out["hints"]) >= 0.5).float()

        hint_tp += int(((pred_hints == 1) & (y_hints == 1)).sum().item())
        hint_fp += int(((pred_hints == 1) & (y_hints == 0)).sum().item())
        hint_fn += int(((pred_hints == 0) & (y_hints == 1)).sum().item())

    hint_precision = hint_tp / max(hint_tp + hint_fp, 1)
    hint_recall = hint_tp / max(hint_tp + hint_fn, 1)
    hint_f1 = 2 * hint_precision * hint_recall / max(hint_precision + hint_recall, 1e-8)

    return {
        "loss": total_loss / max(batches, 1),
        "followup_acc": followup_correct / max(followup_total, 1),
        "subject_acc": subject_correct / max(class_total, 1),
        "domain_acc": domain_correct / max(class_total, 1),
        "intent_acc": intent_correct / max(class_total, 1),
        "hint_precision": hint_precision,
        "hint_recall": hint_recall,
        "hint_f1": hint_f1,
    }


# ============================================================
# PREDICTION DEBUG
# ============================================================


@torch.no_grad()
def predict_context(model, row, vocab, labels):
    model.eval()

    x = vectorize_input(row, vocab).unsqueeze(0).to(DEVICE)

    out = model(x)

    is_followup = bool(torch.sigmoid(out["followup"])[0, 0].item() >= 0.5)

    subject_id = int(torch.argmax(out["subject"], dim=-1).item())
    domain_id = int(torch.argmax(out["domain"], dim=-1).item())
    intent_id = int(torch.argmax(out["intent"], dim=-1).item())

    hint_probs = torch.sigmoid(out["hints"])[0].cpu().tolist()

    hints = []

    for idx, prob in enumerate(hint_probs):
        if prob >= 0.5:
            hints.append((labels["topic_hints"][idx], prob))

    hints.sort(key=lambda x: x[1], reverse=True)
    hints = hints[:5]

    return {
        "is_followup": is_followup,
        "current_subject": labels["subjects"][subject_id],
        "current_domain": labels["domains"][domain_id],
        "current_intent": labels["intents"][intent_id],
        "topic_hints": [h for h, _ in hints],
        "hint_scores": hints,
    }


def print_samples(model, rows, vocab, labels, count=10):
    print()
    print("================ SAMPLE PREDICTIONS ================")

    sample_rows = random.sample(rows, min(count, len(rows)))

    for row in sample_rows:
        expected = row["context"]
        predicted = predict_context(model, row, vocab, labels)

        print()
        print("MESSAGE:")
        print(row.get("message", ""))

        print()
        print("HISTORY:")
        for h in row.get("history", []):
            print(" ", h)

        print()
        print("EXPECTED:")
        print(json.dumps(expected, indent=2))

        print()
        print("PREDICTED:")
        clean_pred = dict(predicted)
        clean_pred.pop("hint_scores", None)
        print(json.dumps(clean_pred, indent=2))

        print()
        print("HINT SCORES:")
        for h, s in predicted["hint_scores"]:
            print(f"  {s:.3f}  {h}")

        print()
        print("----------------------------------------------------")

    print("====================================================")
    print()


# ============================================================
# SAVE / EXPORT
# ============================================================


def save_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def export_onnx(model, input_size):
    model.eval()

    wrapper = OnnxWrapper(model).to(DEVICE)

    dummy = torch.zeros(1, input_size, dtype=torch.float32, device=DEVICE)

    torch.onnx.export(
        wrapper,
        dummy,
        MODEL_ONNX_PATH,
        input_names=["input"],
        output_names=[
            "followup_logits",
            "subject_logits",
            "domain_logits",
            "intent_logits",
            "hints_logits",
        ],
        dynamic_axes={
            "input": {0: "batch"},
            "followup_logits": {0: "batch"},
            "subject_logits": {0: "batch"},
            "domain_logits": {0: "batch"},
            "intent_logits": {0: "batch"},
            "hints_logits": {0: "batch"},
        },
        opset_version=17,
    )

    print(f"[saved] {MODEL_ONNX_PATH}")


# ============================================================
# STATS
# ============================================================


def print_dataset_stats(rows):
    followup_counts = Counter()
    subjects = Counter()
    domains = Counter()
    intents = Counter()
    hints = Counter()

    for row in rows:
        ctx = row["context"]

        followup_counts[str(ctx["is_followup"])] += 1
        subjects[ctx["current_subject"]] += 1
        domains[ctx["current_domain"]] += 1
        intents[ctx["current_intent"]] += 1

        for h in ctx["topic_hints"]:
            hints[h] += 1

    print()
    print("================ DATASET STATS ================")
    print(f"rows: {len(rows)}")
    print()
    print("is_followup:")
    for k, v in followup_counts.most_common():
        print(f"  {k:5s} {v:5d}")

    print()
    print("subjects:")
    for k, v in subjects.most_common():
        print(f"  {k or '<blank>':30s} {v:5d}")

    print()
    print("domains:")
    for k, v in domains.most_common():
        print(f"  {k or '<blank>':30s} {v:5d}")

    print()
    print("intents:")
    for k, v in intents.most_common():
        print(f"  {k or '<blank>':30s} {v:5d}")

    print()
    print("topic hints:")
    for k, v in hints.most_common():
        print(f"  {k:30s} {v:5d}")

    print("================================================")
    print()


# ============================================================
# MAIN
# ============================================================


def main():
    print(f"device: {DEVICE}")

    rows = load_jsonl(DATASET_PATH)
    rows = clean_rows(rows)

    if not rows:
        raise RuntimeError("No usable rows found.")

    random.shuffle(rows)

    print_dataset_stats(rows)

    vocab = build_vocab(rows)
    labels, maps = build_label_maps()

    print(f"input vocab size: {len(vocab)}")
    print(f"subjects:         {len(labels['subjects'])}")
    print(f"domains:          {len(labels['domains'])}")
    print(f"intents:          {len(labels['intents'])}")
    print(f"topic hints:      {len(labels['topic_hints'])}")

    split_idx = int(len(rows) * (1.0 - VAL_SPLIT))

    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]

    print(f"train rows: {len(train_rows)}")
    print(f"val rows:   {len(val_rows)}")

    train_ds = ContextWatcherDataset(train_rows, vocab, maps)
    val_ds = ContextWatcherDataset(val_rows, vocab, maps)

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

    model = ContextWatcherNet(
        input_size=len(vocab),
        hidden_size=HIDDEN_SIZE,
        num_subjects=len(labels["subjects"]),
        num_domains=len(labels["domains"]),
        num_intents=len(labels["intents"]),
        num_hints=len(labels["topic_hints"]),
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    losses = {
        "bce": nn.BCEWithLogitsLoss(),
        "ce": nn.CrossEntropyLoss(),
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_train_loss = 0.0
        batches = 0

        for batch in train_loader:
            x, y_followup, y_subject, y_domain, y_intent, y_hints = batch

            x = x.to(DEVICE)
            y_followup = y_followup.to(DEVICE)
            y_subject = y_subject.to(DEVICE)
            y_domain = y_domain.to(DEVICE)
            y_intent = y_intent.to(DEVICE)
            y_hints = y_hints.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            out = model(x)

            loss_followup = losses["bce"](out["followup"], y_followup)
            loss_subject = losses["ce"](out["subject"], y_subject)
            loss_domain = losses["ce"](out["domain"], y_domain)
            loss_intent = losses["ce"](out["intent"], y_intent)
            loss_hints = losses["bce"](out["hints"], y_hints)

            loss = (
                FOLLOWUP_LOSS_WEIGHT * loss_followup
                + SUBJECT_LOSS_WEIGHT * loss_subject
                + DOMAIN_LOSS_WEIGHT * loss_domain
                + INTENT_LOSS_WEIGHT * loss_intent
                + HINTS_LOSS_WEIGHT * loss_hints
            )

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            total_train_loss += float(loss.item())
            batches += 1

        train_loss = total_train_loss / max(batches, 1)

        metrics = evaluate(model, val_loader, losses)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {metrics['loss']:.4f} | "
            f"followup {metrics['followup_acc']:.3f} | "
            f"subject {metrics['subject_acc']:.3f} | "
            f"domain {metrics['domain_acc']:.3f} | "
            f"intent {metrics['intent_acc']:.3f} | "
            f"hints_f1 {metrics['hint_f1']:.3f}"
        )

        if metrics["loss"] < best_val_loss - MIN_DELTA:
            best_val_loss = metrics["loss"]
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "input_size": len(vocab),
                "hidden_size": HIDDEN_SIZE,
                "dropout": DROPOUT,
                "best_val_loss": best_val_loss,
                "metrics": metrics,
            }

            torch.save(checkpoint, MODEL_PT_PATH)

            print(f"[saved best] {MODEL_PT_PATH}")

        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                print(f"[early stop] no val improvement for {PATIENCE} epochs")
                break

    print()
    print("[reload best checkpoint]")

    checkpoint = torch.load(MODEL_PT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    save_json(VOCAB_PATH, vocab)
    save_json(LABELS_PATH, labels)

    config = {
        "model_type": "context_watcher_ffn",
        "dataset_path": str(DATASET_PATH).replace("\\", "/"),
        "input_type": "bag_of_words_ngrams",
        "ngrams": list(NGRAMS),
        "max_vocab_size": MAX_VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "dropout": DROPOUT,
        "outputs": {
            "followup": "binary sigmoid",
            "subject": "single-label softmax",
            "domain": "single-label softmax",
            "intent": "single-label softmax",
            "topic_hints": "multi-label sigmoid",
        },
        "files": {
            "model_pt": str(MODEL_PT_PATH).replace("\\", "/"),
            "model_onnx": str(MODEL_ONNX_PATH).replace("\\", "/"),
            "vocab": str(VOCAB_PATH).replace("\\", "/"),
            "labels": str(LABELS_PATH).replace("\\", "/"),
        },
    }

    save_json(CONFIG_PATH, config)

    print(f"[saved] {VOCAB_PATH}")
    print(f"[saved] {LABELS_PATH}")
    print(f"[saved] {CONFIG_PATH}")
    print(f"[saved] {MODEL_PT_PATH}")

    export_onnx(model, len(vocab))

    print_samples(model, val_rows, vocab, labels, count=12)

    print("done.")


if __name__ == "__main__":
    main()
