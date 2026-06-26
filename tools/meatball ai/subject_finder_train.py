import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = Path("assets/data/subject_QA/SubjectFinder.jsonl")
OUTPUT_DIR = Path("assets/models/subject_finder")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PT_PATH = OUTPUT_DIR / "subject_finder.pt"
VOCAB_PATH = OUTPUT_DIR / "vocab.json"
CONFIG_PATH = OUTPUT_DIR / "config.json"
CACHE_PATH = Path("assets/data/subject_finder_local_augmented.jsonl")
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

SEED = 42
VAL_SPLIT = 0.14

MAX_LEN = 96
MAX_VOCAB_SIZE = 12000
MIN_TOKEN_FREQ = 1

BATCH_SIZE = 64
EPOCHS = 60
PATIENCE = 8
MIN_DELTA = 1e-4

LR = 8e-4
WEIGHT_DECAY = 2e-3
GRAD_CLIP = 1.0

EMBED_SIZE = 128
HIDDEN_SIZE = 192
DROPOUT = 0.25

HAS_SUBJECT_LOSS_WEIGHT = 1.0
START_LOSS_WEIGHT = 1.2
END_LOSS_WEIGHT = 1.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PAD = "<PAD>"
UNK = "<UNK>"
PAD_ID = 0
UNK_ID = 1

FOLLOWUP_MESSAGES = [
    "what is it",
    "what is it about",
    "what does it do",
    "who made it",
    "who created it",
    "can i buy it",
    "where can i get it",
    "is it real",
    "is it a product",
    "is it a project",
    "tell me more about it",
    "explain it",
    "summarize it",
]

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ============================================================
# JSON / NORMALIZATION
# ============================================================


def load_jsonl(path):
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
            if "message" not in row or "target_subject" not in row:
                print(f"[skip] missing message/target_subject line {line_num}")
                continue
            rows.append(normalize_row(row))
    return rows


def save_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def cache_rows(rows, path):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def history_to_list(history):
    if history is None:
        return []
    if isinstance(history, list):
        return [str(x).strip() for x in history if str(x).strip()]
    text = str(history).strip()
    return [text] if text else []


def normalize_row(row):
    return {
        "message": str(row.get("message", "")).strip(),
        "history": history_to_list(row.get("history", [])),
        "target_subject": str(row.get("target_subject", "")).strip(),
        "source_dataset": str(row.get("source_dataset", "SubjectFinder.jsonl")).strip()
        or "SubjectFinder.jsonl",
        "source_split": str(row.get("source_split", "local")).strip() or "local",
        "source_kind": str(row.get("source_kind", "base")).strip() or "base",
    }


def row_key(row):
    return (
        row.get("message", "").strip().lower(),
        tuple(x.strip().lower() for x in row.get("history", [])),
        row.get("target_subject", "").strip().lower(),
    )


def normalize_subject_text(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"']+|[\"'?.!,;:]+$", "", text).strip()
    return text


# ============================================================
# DATA IMPROVEMENT
# ============================================================


def dedupe_rows(rows):
    seen = set()
    out = []
    for row in rows:
        key = row_key(row)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def collect_subjects(rows):
    subjects = []
    seen = set()
    for row in rows:
        subject = normalize_subject_text(row.get("target_subject", ""))
        if not subject:
            continue
        norm = subject.lower()
        if norm in seen:
            continue
        seen.add(norm)
        subjects.append(subject)
    return subjects


def build_augmented_followups(subjects):
    rows = []
    for subject in subjects:
        history = [subject]
        for message in FOLLOWUP_MESSAGES:
            rows.append(
                {
                    "message": message,
                    "history": history,
                    "target_subject": subject,
                    "source_dataset": "SubjectFinder.jsonl",
                    "source_split": "local",
                    "source_kind": "synthetic_followup",
                }
            )
    return rows


def build_no_subject_hard_negatives():
    rows = []
    for message in FOLLOWUP_MESSAGES:
        rows.append(
            {
                "message": message,
                "history": [],
                "target_subject": "",
                "source_dataset": "SubjectFinder.jsonl",
                "source_split": "local",
                "source_kind": "synthetic_negative",
            }
        )
    return rows


def split_rows(rows, val_split=VAL_SPLIT):
    positives = []
    negatives = []
    for row in rows:
        if normalize_subject_text(row.get("target_subject", "")):
            positives.append(row)
        else:
            negatives.append(row)

    rng = random.Random(SEED)
    rng.shuffle(positives)
    rng.shuffle(negatives)

    pos_cut = int(len(positives) * (1.0 - val_split))
    neg_cut = int(len(negatives) * (1.0 - val_split))

    if len(positives) > 1:
        pos_cut = max(1, min(pos_cut, len(positives) - 1))
    if len(negatives) > 1:
        neg_cut = max(1, min(neg_cut, len(negatives) - 1))

    train_rows = positives[:pos_cut] + negatives[:neg_cut]
    val_rows = positives[pos_cut:] + negatives[neg_cut:]
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def add_training_augmentation(train_rows):
    subjects = collect_subjects(train_rows)
    augmented = dedupe_rows(
        train_rows
        + build_augmented_followups(subjects)
        + build_no_subject_hard_negatives()
    )
    random.Random(SEED).shuffle(augmented)
    return augmented


# ============================================================
# TOKENIZATION / SPANS
# ============================================================


def normalize_token(tok):
    return str(tok).lower()


def tokenize_with_spans(text):
    text = str(text)
    pattern = r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)*|[^\w\s]"
    return [
        {
            "text": m.group(0),
            "norm": normalize_token(m.group(0)),
            "start": m.start(),
            "end": m.end(),
        }
        for m in re.finditer(pattern, text)
    ]


def history_to_text(history):
    if not history:
        return ""
    if isinstance(history, list):
        return " ".join(str(x) for x in history)
    return str(history)


def build_input_text(row):
    message = str(row.get("message", "")).strip()
    history = history_to_text(row.get("history", []))
    return f"message: {message} history: {history}"


def find_subject_span(input_text, target_subject):
    tokens = tokenize_with_spans(input_text)
    if not tokens:
        return 0, 0, 0, tokens

    subject = normalize_subject_text(target_subject)
    if not subject:
        return 0, 0, 0, tokens

    subject_norms = [t["norm"] for t in tokenize_with_spans(subject)]
    if not subject_norms:
        return 0, 0, 0, tokens

    input_norms = [t["norm"] for t in tokens]
    matches = []
    n = len(subject_norms)
    for i in range(0, len(input_norms) - n + 1):
        if input_norms[i : i + n] == subject_norms:
            matches.append((i, i + n - 1))

    if not matches:
        return None, None, None, tokens

    start_idx, end_idx = matches[-1]
    if start_idx >= MAX_LEN or end_idx >= MAX_LEN:
        return None, None, None, tokens
    return 1, start_idx, end_idx, tokens


# ============================================================
# VOCAB / DATASET
# ============================================================


def build_vocab(rows):
    counter = Counter()
    kept = skipped = 0
    for row in rows:
        has_subject, _, _, tokens = find_subject_span(
            build_input_text(row), row.get("target_subject", "")
        )
        if has_subject is None:
            skipped += 1
            continue
        kept += 1
        for tok in tokens[:MAX_LEN]:
            counter[tok["norm"]] += 1

    vocab = {PAD: PAD_ID, UNK: UNK_ID}
    for tok, count in counter.most_common(MAX_VOCAB_SIZE - len(vocab)):
        if count >= MIN_TOKEN_FREQ and tok not in vocab:
            vocab[tok] = len(vocab)

    print(f"span-usable rows: {kept}")
    print(f"span-skipped rows: {skipped}")
    return vocab


class SubjectFinderDataset(Dataset):
    def __init__(self, rows, vocab):
        self.rows = []
        self.vocab = vocab
        skipped = 0

        for row in rows:
            input_text = build_input_text(row)
            has_subject, start_idx, end_idx, tokens = find_subject_span(
                input_text, row.get("target_subject", "")
            )
            if has_subject is None:
                skipped += 1
                continue
            self.rows.append(
                {
                    "row": row,
                    "input_text": input_text,
                    "tokens": tokens,
                    "has_subject": has_subject,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                }
            )

        if skipped:
            print(f"[dataset] skipped rows with missing span: {skipped}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]
        ids = [self.vocab.get(tok["norm"], UNK_ID) for tok in item["tokens"][:MAX_LEN]]
        attention_mask = [1] * len(ids)
        while len(ids) < MAX_LEN:
            ids.append(PAD_ID)
            attention_mask.append(0)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.float32),
            torch.tensor([float(item["has_subject"])], dtype=torch.float32),
            torch.tensor(item["start_idx"], dtype=torch.long),
            torch.tensor(item["end_idx"], dtype=torch.long),
        )


# ============================================================
# MODEL
# ============================================================


class SubjectFinderNet(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_SIZE, padding_idx=PAD_ID)
        self.encoder = nn.Sequential(
            nn.Linear(EMBED_SIZE, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.LayerNorm(HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )
        self.has_subject_head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, 1),
        )
        self.start_head = nn.Linear(HIDDEN_SIZE, 1)
        self.end_head = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x, attention_mask):
        emb = self.embedding(x)
        h = self.encoder(emb)
        mask = attention_mask.unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        start_logits = (
            self.start_head(h).squeeze(-1).masked_fill(attention_mask == 0, -1e9)
        )
        end_logits = self.end_head(h).squeeze(-1).masked_fill(attention_mask == 0, -1e9)
        return {
            "has_subject": self.has_subject_head(pooled),
            "start": start_logits,
            "end": end_logits,
        }


# ============================================================
# DECODE / EVAL
# ============================================================


def copy_span_text(input_text, start_idx, end_idx):
    tokens = tokenize_with_spans(input_text)
    if (
        not tokens
        or start_idx < 0
        or end_idx < start_idx
        or start_idx >= len(tokens)
        or end_idx >= len(tokens)
    ):
        return ""
    return input_text[tokens[start_idx]["start"] : tokens[end_idx]["end"]].strip()


@torch.no_grad()
def predict_subject(model, row, vocab, threshold=0.5):
    model.eval()
    input_text = build_input_text(row)
    tokens = tokenize_with_spans(input_text)
    ids = [vocab.get(tok["norm"], UNK_ID) for tok in tokens[:MAX_LEN]]
    mask = [1] * len(ids)
    while len(ids) < MAX_LEN:
        ids.append(PAD_ID)
        mask.append(0)
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    attention_mask = torch.tensor([mask], dtype=torch.float32, device=DEVICE)
    out = model(x, attention_mask)
    has_prob = torch.sigmoid(out["has_subject"])[0, 0].item()
    if has_prob < threshold:
        return {
            "has_subject": False,
            "subject": "",
            "has_prob": has_prob,
            "start_idx": None,
            "end_idx": None,
        }
    start_idx = int(torch.argmax(out["start"], dim=-1).item())
    end_idx = int(torch.argmax(out["end"], dim=-1).item())
    if end_idx < start_idx:
        end_idx = start_idx
    return {
        "has_subject": True,
        "subject": copy_span_text(input_text, start_idx, end_idx),
        "has_prob": has_prob,
        "start_idx": start_idx,
        "end_idx": end_idx,
    }


@torch.no_grad()
def evaluate(model, loader, losses):
    model.eval()
    total_loss = batches = has_correct = has_total = start_correct = end_correct = (
        span_exact
    ) = span_total = 0

    for batch in loader:
        x, mask, y_has, y_start, y_end = batch
        x, mask, y_has, y_start, y_end = (
            x.to(DEVICE),
            mask.to(DEVICE),
            y_has.to(DEVICE),
            y_start.to(DEVICE),
            y_end.to(DEVICE),
        )
        out = model(x, mask)
        loss_has = losses["bce"](out["has_subject"], y_has)
        subject_mask = y_has.squeeze(-1) > 0.5
        if subject_mask.any():
            loss_start = losses["ce"](out["start"][subject_mask], y_start[subject_mask])
            loss_end = losses["ce"](out["end"][subject_mask], y_end[subject_mask])
        else:
            loss_start = torch.tensor(0.0, device=DEVICE)
            loss_end = torch.tensor(0.0, device=DEVICE)
        loss = (
            HAS_SUBJECT_LOSS_WEIGHT * loss_has
            + START_LOSS_WEIGHT * loss_start
            + END_LOSS_WEIGHT * loss_end
        )
        total_loss += float(loss.item())
        batches += 1
        pred_has = (torch.sigmoid(out["has_subject"]) >= 0.5).float()
        has_correct += int((pred_has == y_has).sum().item())
        has_total += int(y_has.numel())
        pred_start = torch.argmax(out["start"], dim=-1)
        pred_end = torch.argmax(out["end"], dim=-1)
        if subject_mask.any():
            start_correct += int(
                (pred_start[subject_mask] == y_start[subject_mask]).sum().item()
            )
            end_correct += int(
                (pred_end[subject_mask] == y_end[subject_mask]).sum().item()
            )
            span_exact += int(
                (
                    (pred_start[subject_mask] == y_start[subject_mask])
                    & (pred_end[subject_mask] == y_end[subject_mask])
                )
                .sum()
                .item()
            )
            span_total += int(subject_mask.sum().item())
    return {
        "loss": total_loss / max(batches, 1),
        "has_acc": has_correct / max(has_total, 1),
        "start_acc": start_correct / max(span_total, 1),
        "end_acc": end_correct / max(span_total, 1),
        "span_exact": span_exact / max(span_total, 1),
    }


# ============================================================
# MAIN
# ============================================================


def main():
    print(f"device: {DEVICE}")

    base_rows = dedupe_rows(load_jsonl(DATASET_PATH))
    train_base_rows, val_rows = split_rows(base_rows)
    train_rows = add_training_augmentation(train_base_rows)
    cache_rows(train_rows + val_rows, CACHE_PATH)

    print(f"base rows: {len(base_rows)}")
    print(f"base train rows: {len(train_base_rows)}")
    print(f"augmented train rows: {len(train_rows)}")
    print(f"val rows: {len(val_rows)}")
    print(f"cached rows: {CACHE_PATH}")

    vocab = build_vocab(train_rows)
    train_ds = SubjectFinderDataset(train_rows, vocab)
    val_ds = SubjectFinderDataset(val_rows, vocab)

    if len(train_ds) == 0 or len(val_ds) == 0:
        raise RuntimeError(
            "No usable rows. Make sure target_subject appears in message/history text."
        )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"usable train rows: {len(train_ds)}")
    print(f"usable val rows: {len(val_ds)}")
    print(f"vocab size: {len(vocab)}")
    print(f"max len: {MAX_LEN}")

    model = SubjectFinderNet(vocab_size=len(vocab)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    losses = {"bce": nn.BCEWithLogitsLoss(), "ce": nn.CrossEntropyLoss()}

    best_val = float("inf")
    bad_epochs = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = batches = 0
        for x, mask, y_has, y_start, y_end in train_loader:
            x, mask, y_has, y_start, y_end = (
                x.to(DEVICE),
                mask.to(DEVICE),
                y_has.to(DEVICE),
                y_start.to(DEVICE),
                y_end.to(DEVICE),
            )
            optimizer.zero_grad(set_to_none=True)
            out = model(x, mask)
            loss_has = losses["bce"](out["has_subject"], y_has)
            subject_mask = y_has.squeeze(-1) > 0.5
            if subject_mask.any():
                loss_start = losses["ce"](
                    out["start"][subject_mask], y_start[subject_mask]
                )
                loss_end = losses["ce"](out["end"][subject_mask], y_end[subject_mask])
            else:
                loss_start = torch.tensor(0.0, device=DEVICE)
                loss_end = torch.tensor(0.0, device=DEVICE)
            loss = (
                HAS_SUBJECT_LOSS_WEIGHT * loss_has
                + START_LOSS_WEIGHT * loss_start
                + END_LOSS_WEIGHT * loss_end
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1

        train_loss = total_loss / max(batches, 1)
        metrics = evaluate(model, val_loader, losses)
        print(
            f"epoch {epoch:03d} | train_loss {train_loss:.4f} | val_loss {metrics['loss']:.4f} | has {metrics['has_acc']:.3f} | start {metrics['start_acc']:.3f} | end {metrics['end_acc']:.3f} | span {metrics['span_exact']:.3f}"
        )

        if metrics["loss"] < best_val - MIN_DELTA:
            best_val = metrics["loss"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "vocab_size": len(vocab),
                    "max_len": MAX_LEN,
                    "embed_size": EMBED_SIZE,
                    "hidden_size": HIDDEN_SIZE,
                    "dropout": DROPOUT,
                    "best_val_loss": best_val,
                    "metrics": metrics,
                },
                MODEL_PT_PATH,
            )
            print(f"[saved best] {MODEL_PT_PATH}")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("[early stop]")
                break

    checkpoint = torch.load(MODEL_PT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    save_json(VOCAB_PATH, vocab)
    save_json(
        CONFIG_PATH,
        {
            "model_type": "subject_span_finder",
            "dataset_path": str(DATASET_PATH).replace("\\", "/"),
            "cached_dataset_path": str(CACHE_PATH).replace("\\", "/"),
            "val_split": VAL_SPLIT,
            "synthetic_followup_messages": FOLLOWUP_MESSAGES,
            "max_len": MAX_LEN,
            "max_vocab_size": MAX_VOCAB_SIZE,
            "embed_size": EMBED_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "dropout": DROPOUT,
            "pad_id": PAD_ID,
            "unk_id": UNK_ID,
            "model_pt": str(MODEL_PT_PATH).replace("\\", "/"),
            "vocab": str(VOCAB_PATH).replace("\\", "/"),
        },
    )

    print(f"[saved] {VOCAB_PATH}")
    print(f"[saved] {CONFIG_PATH}")
    print(f"[saved] {MODEL_PT_PATH}")
    print("\nSAMPLES:")
    for row in random.sample(val_rows, min(20, len(val_rows))):
        pred = predict_subject(model, row, vocab)
        print("\nMESSAGE:", row.get("message", ""))
        print("HISTORY:", row.get("history", []))
        print("TRUE:", row.get("target_subject", ""))
        print("PRED:", pred["subject"])
        print("HAS:", f"{pred['has_prob']:.3f}", pred["has_subject"])
        print("SPAN:", pred["start_idx"], pred["end_idx"])

    print("done.")


if __name__ == "__main__":
    main()
