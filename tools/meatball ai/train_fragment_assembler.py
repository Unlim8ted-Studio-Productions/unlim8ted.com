import json
import math
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

DATASET_PATH = Path("assets/data/fragments-training.jsonl")

OUTPUT_DIR = Path("assets/models/ordered_fragment_assembler")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PT_PATH = OUTPUT_DIR / "ordered_fragment_assembler.pt"
ENCODER_ONNX_PATH = OUTPUT_DIR / "ordered_fragment_encoder.onnx"
DECODER_ONNX_PATH = OUTPUT_DIR / "ordered_fragment_decoder_step.onnx"

VOCAB_PATH = OUTPUT_DIR / "vocab.json"
FRAGMENT_IDS_PATH = OUTPUT_DIR / "fragment_ids.json"
CONFIG_PATH = OUTPUT_DIR / "config.json"

SEED = 42

VAL_SPLIT = 0.12

MAX_VOCAB_SIZE = 20000
MIN_TOKEN_FREQ = 1

NGRAMS = (1, 2, 3)

MAX_TARGET_LEN = 10
# This means max 10 real fragments, plus EOS internally.

BATCH_SIZE = 64
EPOCHS = 60
LR = 2e-3
WEIGHT_DECAY = 1e-4

HIDDEN_SIZE = 384
DROPOUT = 0.25
EMBED_SIZE = 192

GRAD_CLIP = 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# SPECIAL TOKENS
# ============================================================

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK_FRAG = "<UNK_FRAG>"

SPECIAL_FRAGMENT_TOKENS = [PAD, BOS, EOS, UNK_FRAG]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_FRAG_ID = 3


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


def make_ngrams(tokens, ngram_sizes=NGRAMS):
    feats = []

    for n in ngram_sizes:
        if len(tokens) < n:
            continue

        for i in range(len(tokens) - n + 1):
            gram = "_".join(tokens[i : i + n])
            feats.append(gram)

    return feats


def row_to_input_text(row):
    question = row.get("question", "")
    history = row.get("history", [])
    topics = row.get("topics", [])

    if isinstance(history, list):
        history_text = " ".join(str(x) for x in history)
    else:
        history_text = str(history)

    if isinstance(topics, list):
        topics_text = " ".join(str(x) for x in topics)
    else:
        topics_text = str(topics)

    combined = (
        f"question: {question} " f"history: {history_text} " f"topics: {topics_text}"
    )

    return combined


def extract_answer_ids(row):
    answer = row.get("answer", [])

    if isinstance(answer, str):
        answer = [answer]

    if not isinstance(answer, list):
        answer = ["i_cant_i_dont_know"]

    clean = []

    for item in answer:
        item = str(item).strip()
        if item:
            clean.append(item)

    if not clean:
        clean = ["i_cant_i_dont_know"]

    return clean[:MAX_TARGET_LEN]


# ============================================================
# LOAD DATA
# ============================================================


def load_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"[skip] bad JSON line {line_num}")
                continue

            if "question" not in row:
                print(f"[skip] missing question line {line_num}")
                continue

            if "answer" not in row:
                print(f"[skip] missing answer line {line_num}")
                continue

            rows.append(row)

    return rows


# ============================================================
# BUILD VOCAB
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


def build_fragment_ids(rows):
    seen = []

    for row in rows:
        answer_ids = extract_answer_ids(row)
        for frag_id in answer_ids:
            if frag_id not in seen:
                seen.append(frag_id)

    fragment_ids = SPECIAL_FRAGMENT_TOKENS + seen

    return fragment_ids


# ============================================================
# VECTORIZATION
# ============================================================


def vectorize_input(row, vocab):
    text = row_to_input_text(row)
    tokens = tokenize(text)
    feats = make_ngrams(tokens)

    x = torch.zeros(len(vocab), dtype=torch.float32)

    counts = Counter(feats)

    for feat, count in counts.items():
        idx = vocab.get(feat, vocab["<UNK>"])
        x[idx] = min(float(count), 5.0)

    return x


def encode_target(answer_ids, fragment_to_id):
    ids = []

    for frag_id in answer_ids:
        ids.append(fragment_to_id.get(frag_id, UNK_FRAG_ID))

    ids = ids[:MAX_TARGET_LEN]

    # Decoder target includes EOS.
    ids.append(EOS_ID)

    # Pad to fixed length.
    while len(ids) < MAX_TARGET_LEN + 1:
        ids.append(PAD_ID)

    return torch.tensor(ids, dtype=torch.long)


# ============================================================
# DATASET
# ============================================================


class OrderedFragmentDataset(Dataset):
    def __init__(self, rows, vocab, fragment_to_id):
        self.rows = rows
        self.vocab = vocab
        self.fragment_to_id = fragment_to_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        x = vectorize_input(row, self.vocab)

        answer_ids = extract_answer_ids(row)
        y = encode_target(answer_ids, self.fragment_to_id)

        return x, y


# ============================================================
# MODEL
# ============================================================


class OrderedFragmentAssembler(nn.Module):
    """
    Ordered sequence model.

    Encoder:
        bag/ngram vector -> hidden state

    Decoder:
        previous fragment token -> GRU -> next fragment token

    Training:
        teacher forcing with BOS + expected previous fragments

    Inference:
        start with BOS, repeatedly predict next fragment until EOS
    """

    def __init__(
        self,
        input_size,
        num_fragments,
        hidden_size=HIDDEN_SIZE,
        embed_size=EMBED_SIZE,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_fragments = num_fragments
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
        )

        self.embedding = nn.Embedding(num_fragments, embed_size)

        self.decoder_cell = nn.GRUCell(embed_size, hidden_size)

        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_fragments),
        )

    def encode(self, x):
        return self.encoder(x)

    def decoder_step(self, prev_token, hidden):
        emb = self.embedding(prev_token)
        hidden = self.decoder_cell(emb, hidden)
        logits = self.output(hidden)
        return logits, hidden

    def forward(self, x, target=None, max_len=MAX_TARGET_LEN + 1, teacher_forcing=True):
        batch_size = x.size(0)

        hidden = self.encode(x)

        prev_token = torch.full(
            (batch_size,),
            BOS_ID,
            dtype=torch.long,
            device=x.device,
        )

        logits_list = []

        for t in range(max_len):
            logits, hidden = self.decoder_step(prev_token, hidden)
            logits_list.append(logits.unsqueeze(1))

            if teacher_forcing and target is not None:
                prev_token = target[:, t]
            else:
                prev_token = torch.argmax(logits, dim=-1)

        logits = torch.cat(logits_list, dim=1)

        return logits


class EncoderExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, x):
        return self.encoder(x)


class DecoderStepExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.embedding = model.embedding
        self.decoder_cell = model.decoder_cell
        self.output = model.output

    def forward(self, prev_token, hidden):
        emb = self.embedding(prev_token)
        next_hidden = self.decoder_cell(emb, hidden)
        logits = self.output(next_hidden)
        return logits, next_hidden


# ============================================================
# METRICS
# ============================================================


def sequence_from_ids(ids):
    out = []

    for x in ids:
        x = int(x)

        if x == PAD_ID:
            continue

        if x == EOS_ID:
            break

        if x in (BOS_ID, PAD_ID):
            continue

        out.append(x)

    return out


@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0

    exact = 0
    total_rows = 0

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, reduction="sum")

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(
            x, target=None, max_len=MAX_TARGET_LEN + 1, teacher_forcing=False
        )

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        total_loss += float(loss.item())

        preds = torch.argmax(logits, dim=-1)

        mask = y != PAD_ID

        correct_tokens += int(((preds == y) & mask).sum().item())
        total_tokens += int(mask.sum().item())

        for pred_row, y_row in zip(preds.cpu().tolist(), y.cpu().tolist()):
            pred_seq = sequence_from_ids(pred_row)
            true_seq = sequence_from_ids(y_row)

            if pred_seq == true_seq:
                exact += 1

            total_rows += 1

    avg_loss = total_loss / max(total_tokens, 1)
    token_acc = correct_tokens / max(total_tokens, 1)
    exact_acc = exact / max(total_rows, 1)

    return {
        "val_loss_per_token": avg_loss,
        "val_token_accuracy": token_acc,
        "val_exact_sequence_accuracy": exact_acc,
    }


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================


@torch.no_grad()
def predict_ordered_fragments(model, row, vocab, fragment_ids):
    model.eval()

    x = vectorize_input(row, vocab).unsqueeze(0).to(DEVICE)

    hidden = model.encode(x)

    prev_token = torch.tensor([BOS_ID], dtype=torch.long, device=DEVICE)

    predicted = []
    scores = []

    for _ in range(MAX_TARGET_LEN + 1):
        logits, hidden = model.decoder_step(prev_token, hidden)

        probs = torch.softmax(logits, dim=-1)
        score, token = torch.max(probs, dim=-1)

        token_id = int(token.item())
        score_value = float(score.item())

        if token_id == EOS_ID:
            break

        if token_id in (PAD_ID, BOS_ID):
            break

        predicted.append(fragment_ids[token_id])
        scores.append(score_value)

        prev_token = token

    return predicted, scores


def print_samples(model, rows, vocab, fragment_ids, count=8):
    sample_rows = random.sample(rows, min(count, len(rows)))

    print("\n================ SAMPLE PREDICTIONS ================\n")

    for row in sample_rows:
        question = row.get("question", "")
        history = row.get("history", [])
        topics = row.get("topics", [])
        expected = extract_answer_ids(row)

        predicted, scores = predict_ordered_fragments(model, row, vocab, fragment_ids)

        print("QUESTION:")
        print(question)
        print()

        print("HISTORY:")
        print(history)
        print()

        print("TOPICS:")
        print(topics)
        print()

        print("EXPECTED ORDER:")
        print(expected)
        print()

        print("PREDICTED ORDER:")
        for frag_id, score in zip(predicted, scores):
            print(f"{score:.3f}  {frag_id}")

        print("\n----------------------------------------------------\n")


# ============================================================
# EXPORT
# ============================================================


def export_onnx(model, input_size, num_fragments):
    model.eval()

    encoder = EncoderExportWrapper(model).to(DEVICE)
    decoder = DecoderStepExportWrapper(model).to(DEVICE)

    dummy_x = torch.zeros(1, input_size, dtype=torch.float32, device=DEVICE)

    torch.onnx.export(
        encoder,
        dummy_x,
        ENCODER_ONNX_PATH,
        input_names=["input"],
        output_names=["hidden"],
        dynamic_axes={
            "input": {0: "batch"},
            "hidden": {0: "batch"},
        },
        opset_version=17,
    )

    dummy_prev_token = torch.tensor([BOS_ID], dtype=torch.long, device=DEVICE)
    dummy_hidden = torch.zeros(1, HIDDEN_SIZE, dtype=torch.float32, device=DEVICE)

    torch.onnx.export(
        decoder,
        (dummy_prev_token, dummy_hidden),
        DECODER_ONNX_PATH,
        input_names=["prev_token", "hidden"],
        output_names=["logits", "next_hidden"],
        dynamic_axes={
            "prev_token": {0: "batch"},
            "hidden": {0: "batch"},
            "logits": {0: "batch"},
            "next_hidden": {0: "batch"},
        },
        opset_version=17,
    )

    print(f"[saved] {ENCODER_ONNX_PATH}")
    print(f"[saved] {DECODER_ONNX_PATH}")


# ============================================================
# MAIN
# ============================================================


def main():
    print(f"Using device: {DEVICE}")

    rows = load_jsonl(DATASET_PATH)

    if not rows:
        raise RuntimeError(f"No rows found in {DATASET_PATH}")

    random.shuffle(rows)

    print(f"Loaded rows: {len(rows)}")

    vocab = build_vocab(rows)
    fragment_ids = build_fragment_ids(rows)
    fragment_to_id = {frag_id: i for i, frag_id in enumerate(fragment_ids)}

    print(f"Input vocab size: {len(vocab)}")
    print(f"Fragment output size: {len(fragment_ids)}")
    print(f"Max target fragments: {MAX_TARGET_LEN}")

    split_idx = int(len(rows) * (1.0 - VAL_SPLIT))

    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]

    print(f"Train rows: {len(train_rows)}")
    print(f"Val rows: {len(val_rows)}")

    train_ds = OrderedFragmentDataset(train_rows, vocab, fragment_to_id)
    val_ds = OrderedFragmentDataset(val_rows, vocab, fragment_to_id)

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

    model = OrderedFragmentAssembler(
        input_size=len(vocab),
        num_fragments=len(fragment_ids),
        hidden_size=HIDDEN_SIZE,
        embed_size=EMBED_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    best_exact = -1.0
    best_loss = math.inf

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        batches = 0

        # Teacher forcing starts high and slowly lowers.
        teacher_forcing_ratio = max(0.55, 1.0 - (epoch / EPOCHS) * 0.45)

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            use_teacher_forcing = random.random() < teacher_forcing_ratio

            logits = model(
                x,
                target=y,
                max_len=MAX_TARGET_LEN + 1,
                teacher_forcing=use_teacher_forcing,
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            optimizer.step()

            total_loss += float(loss.item())
            batches += 1

        train_loss = total_loss / max(batches, 1)

        metrics = evaluate(model, val_loader)

        print(
            f"epoch {epoch:03d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss/token {metrics['val_loss_per_token']:.4f} | "
            f"val_token_acc {metrics['val_token_accuracy']:.4f} | "
            f"val_exact_seq {metrics['val_exact_sequence_accuracy']:.4f} | "
            f"tf {teacher_forcing_ratio:.2f}"
        )

        improved = False

        if metrics["val_exact_sequence_accuracy"] > best_exact:
            improved = True

        elif (
            metrics["val_exact_sequence_accuracy"] == best_exact
            and metrics["val_loss_per_token"] < best_loss
        ):
            improved = True

        if improved:
            best_exact = metrics["val_exact_sequence_accuracy"]
            best_loss = metrics["val_loss_per_token"]

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "vocab_size": len(vocab),
                "num_fragments": len(fragment_ids),
                "hidden_size": HIDDEN_SIZE,
                "embed_size": EMBED_SIZE,
                "dropout": DROPOUT,
                "max_target_len": MAX_TARGET_LEN,
                "special_tokens": {
                    "PAD": PAD,
                    "BOS": BOS,
                    "EOS": EOS,
                    "UNK_FRAG": UNK_FRAG,
                    "PAD_ID": PAD_ID,
                    "BOS_ID": BOS_ID,
                    "EOS_ID": EOS_ID,
                    "UNK_FRAG_ID": UNK_FRAG_ID,
                },
                "metrics": metrics,
            }

            torch.save(checkpoint, MODEL_PT_PATH)

            print(f"[saved best] {MODEL_PT_PATH}")

    # Reload best before export.
    checkpoint = torch.load(MODEL_PT_PATH, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with VOCAB_PATH.open("w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    with FRAGMENT_IDS_PATH.open("w", encoding="utf-8") as f:
        json.dump(fragment_ids, f, indent=2)

    config = {
        "model_type": "ordered_fragment_assembler_gru",
        "input_type": "bag_of_ngrams",
        "ngrams": list(NGRAMS),
        "max_vocab_size": MAX_VOCAB_SIZE,
        "max_target_len": MAX_TARGET_LEN,
        "hidden_size": HIDDEN_SIZE,
        "embed_size": EMBED_SIZE,
        "dropout": DROPOUT,
        "pad_id": PAD_ID,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "unk_frag_id": UNK_FRAG_ID,
        "encoder_onnx": str(ENCODER_ONNX_PATH).replace("\\", "/"),
        "decoder_step_onnx": str(DECODER_ONNX_PATH).replace("\\", "/"),
        "vocab": str(VOCAB_PATH).replace("\\", "/"),
        "fragment_ids": str(FRAGMENT_IDS_PATH).replace("\\", "/"),
    }

    with CONFIG_PATH.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"[saved] {VOCAB_PATH}")
    print(f"[saved] {FRAGMENT_IDS_PATH}")
    print(f"[saved] {CONFIG_PATH}")

    export_onnx(
        model=model,
        input_size=len(vocab),
        num_fragments=len(fragment_ids),
    )

    print_samples(model, val_rows, vocab, fragment_ids, count=10)

    print("\nDone.")


if __name__ == "__main__":
    main()
