# train_alternating_latent_chunk_qa_cross_attention.py

import json
import re
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path(r"assets/data/specialized_QA")
OTHER_FILE = Path(r"tools/Smart-Meatball-Data.jsonl")
OUT_DIR = Path(r"tools\meatball ai\alternating_latent_chunk_qa_cross_attention_out")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
EPOCHS = 30
BATCH_SIZE = 64

PRED_LR = 1e-3
LATENT_LR = 3e-4

WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

TRAIN_SPLIT = 0.9
LIMIT = 0

MAX_QUESTION_CHUNKS = 64
MAX_CHUNK_CHARS = 32

MAX_ANSWER_LEN = 80
MAX_GENERATE_LEN = 80

MAX_CHARS = 8000
MAX_ANSWER_TOKENS = 12000

CHAR_EMBED_SIZE = 48
CHUNK_HIDDEN = 96
LATENT_SIZE = 128

LATENT_INTERNAL_STEPS = 4

ANSWER_EMBED_SIZE = 128
PRED_HIDDEN = 256

DROPOUT = 0.15

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


# ============================================================
# TEXT
# ============================================================


def normalize(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text):
    return re.findall(r"[\w']+|[^\w\s]", normalize(text), re.UNICODE)


def word_ngrams(tokens, ns=(1, 2, 3)):
    tokens = [t.lower() for t in tokens]
    out = []

    for n in ns:
        for i in range(len(tokens) - n + 1):
            out.append(" ".join(tokens[i : i + n]))

    return out


def char_ngrams(text, ns=(3, 4, 5)):
    text = normalize(text).lower().replace(" ", "_")
    out = []

    for n in ns:
        for i in range(len(text) - n + 1):
            out.append(text[i : i + n])

    return out


def question_chunks(text):
    toks = tokenize(text)

    chunks = []
    chunks.extend(word_ngrams(toks, ns=(1, 2, 3)))
    chunks.extend(char_ngrams(text, ns=(3, 4, 5)))

    seen = set()
    unique = []

    for c in chunks:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique[:MAX_QUESTION_CHUNKS]


# ============================================================
# DATA
# ============================================================


def load_jsonl_file(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    return rows


def extract_qa(row):
    q = (
        row.get("question")
        or row.get("input")
        or row.get("prompt")
        or row.get("q")
        or ""
    )

    a = (
        row.get("answer")
        or row.get("output")
        or row.get("response")
        or row.get("completion")
        or row.get("a")
        or ""
    )

    return normalize(q), normalize(a)


def load_dataset(data_dir):
    rows = []

    paths = list(data_dir.rglob("*.jsonl"))
    print(f"JSONL files found in DATA_DIR: {len(paths)}", flush=True)

    for path in paths:
        file_rows = load_jsonl_file(path)
        print(f"Loading {path} | rows: {len(file_rows)}", flush=True)

        for raw in file_rows:
            q, a = extract_qa(raw)

            if q and a:
                rows.append(
                    {
                        "question": q,
                        "answer": a,
                        "source": str(path),
                    }
                )

    if OTHER_FILE.exists():
        other_rows = load_jsonl_file(OTHER_FILE)
        print(f"Loading OTHER_FILE {OTHER_FILE} | rows: {len(other_rows)}", flush=True)

        for raw in other_rows:
            q, a = extract_qa(raw)

            if q and a:
                rows.append(
                    {
                        "question": q,
                        "answer": a,
                        "source": str(OTHER_FILE),
                    }
                )
    else:
        print(f"OTHER_FILE not found: {OTHER_FILE}", flush=True)

    cleaned = []
    seen = set()

    for row in rows:
        key = row["question"].lower()

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(row)

    print(f"Total usable rows before dedupe: {len(rows)}", flush=True)
    print(f"Total usable rows after dedupe: {len(cleaned)}", flush=True)

    return cleaned


# ============================================================
# VOCABS
# ============================================================


def build_char_vocab(rows):
    counts = Counter()

    for row in rows:
        for chunk in question_chunks(row["question"]):
            counts.update(list(chunk))

    vocab = {
        PAD: PAD_ID,
        BOS: BOS_ID,
        EOS: EOS_ID,
        UNK: UNK_ID,
    }

    for ch, _ in counts.most_common(MAX_CHARS):
        if ch not in vocab:
            vocab[ch] = len(vocab)

    id_to_char = {v: k for k, v in vocab.items()}
    return vocab, id_to_char


def build_answer_vocab(rows):
    counts = Counter()

    for row in rows:
        counts.update(tokenize(row["answer"]))

    vocab = {
        PAD: PAD_ID,
        BOS: BOS_ID,
        EOS: EOS_ID,
        UNK: UNK_ID,
    }

    for tok, _ in counts.most_common(MAX_ANSWER_TOKENS):
        if tok not in vocab:
            vocab[tok] = len(vocab)

    id_to_token = {v: k for k, v in vocab.items()}
    return vocab, id_to_token


def encode_question_chunks(text, char_vocab):
    chunks = question_chunks(text)
    encoded = []

    for chunk in chunks:
        ids = []

        for ch in chunk[:MAX_CHUNK_CHARS]:
            ids.append(char_vocab.get(ch, UNK_ID))

        while len(ids) < MAX_CHUNK_CHARS:
            ids.append(PAD_ID)

        encoded.append(ids)

    while len(encoded) < MAX_QUESTION_CHUNKS:
        encoded.append([PAD_ID] * MAX_CHUNK_CHARS)

    return torch.tensor(encoded, dtype=torch.long)


def encode_answer(text, answer_vocab):
    ids = [BOS_ID]

    for tok in tokenize(text):
        ids.append(answer_vocab.get(tok, UNK_ID))

    ids.append(EOS_ID)

    ids = ids[:MAX_ANSWER_LEN]

    if ids[-1] != EOS_ID and len(ids) == MAX_ANSWER_LEN:
        ids[-1] = EOS_ID

    while len(ids) < MAX_ANSWER_LEN:
        ids.append(PAD_ID)

    return torch.tensor(ids, dtype=torch.long)


def decode_answer(ids, id_to_token):
    toks = []

    for idx in ids:
        idx = int(idx)

        if idx == EOS_ID:
            break

        if idx in (PAD_ID, BOS_ID):
            continue

        toks.append(id_to_token.get(idx, UNK))

    text = " ".join(toks)

    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = text.replace(" ’ ", "’")
    text = text.replace(" 's", "'s")
    text = text.replace(" n't", "n't")
    text = text.replace(" 'm", "'m")
    text = text.replace(" 're", "'re")
    text = text.replace(" 've", "'ve")
    text = text.replace(" 'll", "'ll")
    text = text.replace(" 'd", "'d")

    return text.strip()


# ============================================================
# DATASET
# ============================================================


class ChunkQADataset(Dataset):
    def __init__(self, rows, char_vocab, answer_vocab):
        self.rows = rows
        self.char_vocab = char_vocab
        self.answer_vocab = answer_vocab

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        q_chunks = encode_question_chunks(row["question"], self.char_vocab)
        answer_ids = encode_answer(row["answer"], self.answer_vocab)

        return q_chunks, answer_ids


# ============================================================
# LATENT MODEL
# ============================================================


class ChunkToLatentModel(nn.Module):
    """
    Encodes question-derived chunks.

    Returns:
        global_z:        [B, LATENT_SIZE]
        chunk_vecs:      [B, G, LATENT_SIZE]
        chunk_mask:      [B, G]
    """

    def __init__(self, char_vocab_size):
        super().__init__()

        self.char_embedding = nn.Embedding(
            char_vocab_size,
            CHAR_EMBED_SIZE,
            padding_idx=PAD_ID,
        )

        self.chunk_gru = nn.GRU(
            input_size=CHAR_EMBED_SIZE,
            hidden_size=CHUNK_HIDDEN,
            batch_first=True,
            bidirectional=True,
        )

        self.chunk_to_latent = nn.Sequential(
            nn.Linear(CHUNK_HIDDEN * 2, LATENT_SIZE),
            nn.LayerNorm(LATENT_SIZE),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        self.pool_attention = nn.Sequential(
            nn.Linear(LATENT_SIZE, LATENT_SIZE),
            nn.Tanh(),
            nn.Linear(LATENT_SIZE, 1),
        )

        self.latent_refine = nn.GRUCell(LATENT_SIZE, LATENT_SIZE)
        self.latent_norm = nn.LayerNorm(LATENT_SIZE)

        self.final = nn.Sequential(
            nn.Linear(LATENT_SIZE, LATENT_SIZE),
            nn.LayerNorm(LATENT_SIZE),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

    def encode_chunks(self, chunk_ids):
        B, G, C = chunk_ids.shape

        flat = chunk_ids.reshape(B * G, C)

        emb = self.char_embedding(flat)
        out, _ = self.chunk_gru(emb)

        char_mask = (flat != PAD_ID).float().unsqueeze(-1)
        pooled = (out * char_mask).sum(dim=1) / char_mask.sum(dim=1).clamp(min=1.0)

        chunk_vecs = self.chunk_to_latent(pooled)
        chunk_vecs = chunk_vecs.reshape(B, G, LATENT_SIZE)

        chunk_mask = (chunk_ids != PAD_ID).any(dim=-1)

        return chunk_vecs, chunk_mask

    def forward(self, chunk_ids):
        chunk_vecs, chunk_mask = self.encode_chunks(chunk_ids)

        scores = self.pool_attention(chunk_vecs).squeeze(-1)
        scores = scores.masked_fill(~chunk_mask, -1e9)

        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        z = (chunk_vecs * weights).sum(dim=1)

        h = z
        x = z

        for _ in range(LATENT_INTERNAL_STEPS):
            h = self.latent_refine(x, h)
            h = self.latent_norm(h)
            x = h

        global_z = self.final(h)

        return global_z, chunk_vecs, chunk_mask


# ============================================================
# PREDICTION MODEL WITH CROSS ATTENTION
# ============================================================


class PredictionModel(nn.Module):
    """
    Decoder with cross-attention.

    At each answer token:
        decoder hidden state queries the question chunk vectors.
    """

    def __init__(self, answer_vocab_size):
        super().__init__()

        self.answer_embedding = nn.Embedding(
            answer_vocab_size,
            ANSWER_EMBED_SIZE,
            padding_idx=PAD_ID,
        )

        self.hidden_init = nn.Linear(LATENT_SIZE, PRED_HIDDEN)

        self.gru = nn.GRUCell(
            input_size=ANSWER_EMBED_SIZE + LATENT_SIZE + LATENT_SIZE,
            hidden_size=PRED_HIDDEN,
        )

        self.query_proj = nn.Linear(PRED_HIDDEN, LATENT_SIZE)
        self.key_proj = nn.Linear(LATENT_SIZE, LATENT_SIZE)
        self.value_proj = nn.Linear(LATENT_SIZE, LATENT_SIZE)

        self.attn_norm = nn.LayerNorm(LATENT_SIZE)

        self.output = nn.Sequential(
            nn.Linear(PRED_HIDDEN + LATENT_SIZE + LATENT_SIZE, PRED_HIDDEN),
            nn.LayerNorm(PRED_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(PRED_HIDDEN, answer_vocab_size),
        )

    def cross_attention(self, h, chunk_vecs, chunk_mask):
        """
        h:          [B, PRED_HIDDEN]
        chunk_vecs: [B, G, LATENT_SIZE]
        chunk_mask: [B, G]

        returns:
            context: [B, LATENT_SIZE]
        """

        q = self.query_proj(h).unsqueeze(1)
        k = self.key_proj(chunk_vecs)
        v = self.value_proj(chunk_vecs)

        scores = (q * k).sum(dim=-1) / (LATENT_SIZE**0.5)
        scores = scores.masked_fill(~chunk_mask, -1e9)

        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        context = (v * weights).sum(dim=1)

        context = self.attn_norm(context)

        return context

    def forward(self, global_z, chunk_vecs, chunk_mask, answer_ids):
        B, T = answer_ids.shape

        h = torch.tanh(self.hidden_init(global_z))

        prev_ids = answer_ids[:, 0]
        logits = []

        context = self.cross_attention(h, chunk_vecs, chunk_mask)

        for t in range(1, T):
            emb = self.answer_embedding(prev_ids)

            inp = torch.cat([emb, global_z, context], dim=-1)
            h = self.gru(inp, h)

            context = self.cross_attention(h, chunk_vecs, chunk_mask)

            step_logits = self.output(torch.cat([h, global_z, context], dim=-1))
            logits.append(step_logits)

            prev_ids = answer_ids[:, t]

        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def generate(self, global_z, chunk_vecs, chunk_mask, max_len=MAX_GENERATE_LEN):
        B = global_z.shape[0]

        h = torch.tanh(self.hidden_init(global_z))

        prev_ids = torch.full(
            (B,),
            BOS_ID,
            dtype=torch.long,
            device=global_z.device,
        )

        outputs = []

        context = self.cross_attention(h, chunk_vecs, chunk_mask)

        for _ in range(max_len):
            emb = self.answer_embedding(prev_ids)

            inp = torch.cat([emb, global_z, context], dim=-1)
            h = self.gru(inp, h)

            context = self.cross_attention(h, chunk_vecs, chunk_mask)

            logits = self.output(torch.cat([h, global_z, context], dim=-1))
            next_ids = torch.argmax(logits, dim=-1)

            outputs.append(next_ids)

            prev_ids = next_ids

        return torch.stack(outputs, dim=1)


# ============================================================
# HELPERS
# ============================================================


def set_requires_grad(module, value):
    for p in module.parameters():
        p.requires_grad = value


def forward_loss(latent_model, prediction_model, q_chunks, answer_ids, criterion):
    global_z, chunk_vecs, chunk_mask = latent_model(q_chunks)

    logits = prediction_model(
        global_z,
        chunk_vecs,
        chunk_mask,
        answer_ids,
    )

    target = answer_ids[:, 1:]

    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
    )

    return loss, logits


def token_accuracy(logits, target):
    pred = torch.argmax(logits, dim=-1)

    mask = target != PAD_ID
    correct = (pred == target) & mask

    return correct.sum().item(), mask.sum().item()


def exact_accuracy(logits, target):
    pred = torch.argmax(logits, dim=-1)

    mask = target != PAD_ID
    matches = ((pred == target) | ~mask).all(dim=1)

    return matches.sum().item(), target.size(0)


@torch.no_grad()
def evaluate(latent_model, prediction_model, loader, criterion):
    latent_model.eval()
    prediction_model.eval()

    total_loss = 0.0
    batches = 0

    tok_correct = 0
    tok_total = 0

    exact_correct = 0
    exact_total = 0

    for q_chunks, answer_ids in loader:
        q_chunks = q_chunks.to(DEVICE)
        answer_ids = answer_ids.to(DEVICE)

        loss, logits = forward_loss(
            latent_model,
            prediction_model,
            q_chunks,
            answer_ids,
            criterion,
        )

        target = answer_ids[:, 1:]

        c, t = token_accuracy(logits, target)
        ec, et = exact_accuracy(logits, target)

        total_loss += loss.item()
        batches += 1

        tok_correct += c
        tok_total += t

        exact_correct += ec
        exact_total += et

    return {
        "loss": total_loss / max(1, batches),
        "token_acc": tok_correct / max(1, tok_total),
        "exact_acc": exact_correct / max(1, exact_total),
    }


@torch.no_grad()
def sample_predictions(
    latent_model, prediction_model, rows, char_vocab, id_to_token, count=3
):
    if not rows:
        return

    latent_model.eval()
    prediction_model.eval()

    samples = random.sample(rows, min(count, len(rows)))

    print("\nSAMPLES")

    for row in samples:
        q_chunks = encode_question_chunks(row["question"], char_vocab)
        q_chunks = q_chunks.unsqueeze(0).to(DEVICE)

        global_z, chunk_vecs, chunk_mask = latent_model(q_chunks)

        pred_ids = (
            prediction_model.generate(
                global_z,
                chunk_vecs,
                chunk_mask,
                max_len=MAX_GENERATE_LEN,
            )[0]
            .cpu()
            .tolist()
        )

        print("Q:", row["question"])
        print("T:", row["answer"])
        print("P:", decode_answer(pred_ids, id_to_token))
        print("-" * 60)


# ============================================================
# TRAINING
# ============================================================


def train():
    random.seed(SEED)
    torch.manual_seed(SEED)

    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(SEED)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_dataset(DATA_DIR)

    if LIMIT > 0:
        random.shuffle(rows)
        rows = rows[:LIMIT]

    random.shuffle(rows)

    split = int(len(rows) * TRAIN_SPLIT)

    train_rows = rows[:split]
    val_rows = rows[split:]

    char_vocab, id_to_char = build_char_vocab(train_rows)
    answer_vocab, id_to_token = build_answer_vocab(train_rows)

    train_ds = ChunkQADataset(train_rows, char_vocab, answer_vocab)
    val_ds = ChunkQADataset(val_rows, char_vocab, answer_vocab)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    latent_model = ChunkToLatentModel(
        char_vocab_size=len(char_vocab),
    ).to(DEVICE)

    prediction_model = PredictionModel(
        answer_vocab_size=len(answer_vocab),
    ).to(DEVICE)

    latent_optimizer = torch.optim.AdamW(
        latent_model.parameters(),
        lr=LATENT_LR,
        weight_decay=WEIGHT_DECAY,
    )

    pred_optimizer = torch.optim.AdamW(
        prediction_model.parameters(),
        lr=PRED_LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    best_val = float("inf")

    print("Device:", DEVICE)
    print("Rows:", len(rows))
    print("Train:", len(train_rows))
    print("Val:", len(val_rows))
    print("Char vocab:", len(char_vocab))
    print("Answer vocab:", len(answer_vocab))
    print("Latent LR:", LATENT_LR)
    print("Prediction LR:", PRED_LR)
    print("Training schedule: prediction update x1, latent update x2")
    print("Decoder: GRUCell with cross-attention over chunk vectors")

    for epoch in range(1, EPOCHS + 1):
        latent_model.train()
        prediction_model.train()

        total_pred_loss = 0.0
        total_latent_loss = 0.0
        batches = 0

        for q_chunks, answer_ids in train_loader:
            q_chunks = q_chunks.to(DEVICE)
            answer_ids = answer_ids.to(DEVICE)

            target = answer_ids[:, 1:]

            # ==================================================
            # STEP 1:
            # Prediction model update only
            # latent model frozen
            # ==================================================

            set_requires_grad(latent_model, False)
            set_requires_grad(prediction_model, True)

            latent_model.eval()
            prediction_model.train()

            pred_optimizer.zero_grad()

            with torch.no_grad():
                global_z, chunk_vecs, chunk_mask = latent_model(q_chunks)

            logits = prediction_model(
                global_z,
                chunk_vecs,
                chunk_mask,
                answer_ids,
            )

            pred_loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
            )

            pred_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                prediction_model.parameters(),
                GRAD_CLIP,
            )

            pred_optimizer.step()

            # ==================================================
            # STEP 2:
            # Latent model update only #1
            # prediction model frozen
            # ==================================================

            set_requires_grad(latent_model, True)
            set_requires_grad(prediction_model, False)

            latent_model.train()
            prediction_model.eval()

            latent_optimizer.zero_grad()

            global_z, chunk_vecs, chunk_mask = latent_model(q_chunks)

            logits = prediction_model(
                global_z,
                chunk_vecs,
                chunk_mask,
                answer_ids,
            )

            latent_loss_1 = criterion(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
            )

            latent_loss_1.backward()

            torch.nn.utils.clip_grad_norm_(
                latent_model.parameters(),
                GRAD_CLIP,
            )

            latent_optimizer.step()

            # ==================================================
            # STEP 3:
            # Latent model update only #2
            # prediction model still frozen
            # ==================================================

            latent_optimizer.zero_grad()

            global_z, chunk_vecs, chunk_mask = latent_model(q_chunks)

            logits = prediction_model(
                global_z,
                chunk_vecs,
                chunk_mask,
                answer_ids,
            )

            latent_loss_2 = criterion(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
            )

            latent_loss_2.backward()

            torch.nn.utils.clip_grad_norm_(
                latent_model.parameters(),
                GRAD_CLIP,
            )

            latent_optimizer.step()

            total_pred_loss += pred_loss.item()
            total_latent_loss += (latent_loss_1.item() + latent_loss_2.item()) / 2.0
            batches += 1

        set_requires_grad(latent_model, True)
        set_requires_grad(prediction_model, True)

        val = evaluate(
            latent_model,
            prediction_model,
            val_loader,
            criterion,
        )

        avg_pred_loss = total_pred_loss / max(1, batches)
        avg_latent_loss = total_latent_loss / max(1, batches)

        print(
            f"Epoch {epoch:03d} | "
            f"pred_train {avg_pred_loss:.4f} | "
            f"latent_train {avg_latent_loss:.4f} | "
            f"val {val['loss']:.4f} | "
            f"token_acc {val['token_acc'] * 100:.2f}% | "
            f"exact_acc {val['exact_acc'] * 100:.2f}%"
        )

        if val["loss"] < best_val:
            best_val = val["loss"]

            checkpoint = {
                "latent_model_state": latent_model.state_dict(),
                "prediction_model_state": prediction_model.state_dict(),
                "char_vocab": char_vocab,
                "id_to_char": id_to_char,
                "answer_vocab": answer_vocab,
                "id_to_token": id_to_token,
                "config": {
                    "model_type": "alternating_latent_chunk_qa_cross_attention",
                    "max_question_chunks": MAX_QUESTION_CHUNKS,
                    "max_chunk_chars": MAX_CHUNK_CHARS,
                    "max_answer_len": MAX_ANSWER_LEN,
                    "max_generate_len": MAX_GENERATE_LEN,
                    "char_embed_size": CHAR_EMBED_SIZE,
                    "chunk_hidden": CHUNK_HIDDEN,
                    "latent_size": LATENT_SIZE,
                    "latent_internal_steps": LATENT_INTERNAL_STEPS,
                    "answer_embed_size": ANSWER_EMBED_SIZE,
                    "pred_hidden": PRED_HIDDEN,
                    "dropout": DROPOUT,
                    "cross_attention": True,
                },
            }

            torch.save(
                checkpoint, OUT_DIR / "alternating_latent_chunk_qa_cross_attention.pt"
            )
            print("Saved best model.")

        if epoch % 5 == 0:
            sample_predictions(
                latent_model,
                prediction_model,
                val_rows if val_rows else train_rows,
                char_vocab,
                id_to_token,
                count=3,
            )

    print("Done.")
    print(
        "Best checkpoint:", OUT_DIR / "alternating_latent_chunk_qa_cross_attention.pt"
    )


if __name__ == "__main__":
    train()
