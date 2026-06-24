# train_alternating_latent_chunk_qa_public_encoder_retrieval.py

import json
import re
import random
import math
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

DATA_DIR = Path(r"assets/data/specialized_QA")
OUT_DIR = Path(r"tools\meatball ai\alternating_latent_chunk_qa_out")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42
EPOCHS = 30
BATCH_SIZE = 48

PUBLIC_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PRED_LR = 1e-3
LATENT_LR = 3e-4

WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

TRAIN_SPLIT = 0.9
LIMIT = 0

MAX_ANSWER_LEN = 90
MAX_GENERATE_LEN = 90
MAX_ANSWER_TOKENS = 12000

RETRIEVAL_TOP_K = 5
RETRIEVAL_TEXT_MAX_TOKENS = 80

LATENT_SIZE = 192
LATENT_INTERNAL_STEPS = 4

ANSWER_EMBED_SIZE = 160
PRED_HIDDEN = 320

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


def detokenize_tokens(toks):
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
    text = text.replace("( ", "(")
    text = text.replace(" )", ")")
    return text.strip()


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
    paths = list(data_dir.rglob("*.jsonl"))

    rows = []

    for path in paths:
        for raw in load_jsonl_file(path):
            q, a = extract_qa(raw)

            if q and a:
                rows.append({
                    "question": q,
                    "answer": a,
                    "source": str(path),
                })

    cleaned = []
    seen = set()

    for row in rows:
        key = row["question"].lower().strip()

        if key in seen:
            continue

        seen.add(key)
        cleaned.append(row)

    return cleaned


# ============================================================
# VOCAB
# ============================================================

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


def encode_answer(text, answer_vocab, max_len=MAX_ANSWER_LEN):
    ids = [BOS_ID]

    for tok in tokenize(text):
        ids.append(answer_vocab.get(tok, UNK_ID))

    ids.append(EOS_ID)
    ids = ids[:max_len]

    if ids[-1] != EOS_ID and len(ids) == max_len:
        ids[-1] = EOS_ID

    while len(ids) < max_len:
        ids.append(PAD_ID)

    return torch.tensor(ids, dtype=torch.long)


def encode_plain_tokens(text, answer_vocab, max_len):
    ids = []

    for tok in tokenize(text)[:max_len]:
        ids.append(answer_vocab.get(tok, UNK_ID))

    while len(ids) < max_len:
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

    return detokenize_tokens(toks)


# ============================================================
# PUBLIC ENCODER
# ============================================================

@torch.no_grad()
def encode_questions(public_encoder, questions):
    return public_encoder.encode(
        questions,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        device=DEVICE,
    )


@torch.no_grad()
def build_embedding_matrix(public_encoder, rows, cache_path):
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")

        if payload.get("encoder_name") == PUBLIC_ENCODER_NAME and payload.get("count") == len(rows):
            return payload["embeddings"].to(DEVICE)

    questions = [row["question"] for row in rows]
    embeddings = []

    batch_size = 256

    for i in range(0, len(questions), batch_size):
        batch = questions[i:i + batch_size]
        emb = encode_questions(public_encoder, batch)
        embeddings.append(emb.detach().cpu())

        print(f"Encoded retrieval embeddings: {min(i + batch_size, len(questions))}/{len(questions)}")

    matrix = torch.cat(embeddings, dim=0)

    payload = {
        "encoder_name": PUBLIC_ENCODER_NAME,
        "count": len(rows),
        "embeddings": matrix,
    }

    torch.save(payload, cache_path)

    return matrix.to(DEVICE)


@torch.no_grad()
def retrieve_topk(query_embeddings, bank_embeddings, top_k):
    sims = query_embeddings @ bank_embeddings.T
    values, indices = torch.topk(sims, k=min(top_k, bank_embeddings.size(0)), dim=-1)
    return indices, values


# ============================================================
# DATASET
# ============================================================

class PublicEncoderRetrievalQADataset(Dataset):
    def __init__(self, rows, answer_vocab):
        self.rows = rows
        self.answer_vocab = answer_vocab

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        answer_ids = encode_answer(row["answer"], self.answer_vocab)
        return row["question"], answer_ids


def collate_batch(batch):
    questions = [item[0] for item in batch]
    answer_ids = torch.stack([item[1] for item in batch], dim=0)
    return questions, answer_ids


def build_retrieved_answer_tensor(retrieved_indices, rows, answer_vocab):
    batch_items = []

    for row_indices in retrieved_indices.detach().cpu().tolist():
        per_query = []

        for idx in row_indices:
            answer_text = rows[idx]["answer"]
            encoded = encode_plain_tokens(
                answer_text,
                answer_vocab,
                max_len=RETRIEVAL_TEXT_MAX_TOKENS,
            )
            per_query.append(encoded)

        batch_items.append(torch.stack(per_query, dim=0))

    return torch.stack(batch_items, dim=0).to(DEVICE)


# ============================================================
# LATENT MODEL
# ============================================================

class PublicEmbeddingToLatentModel(nn.Module):
    def __init__(self, public_embed_size):
        super().__init__()

        self.project = nn.Sequential(
            nn.Linear(public_embed_size, LATENT_SIZE),
            nn.LayerNorm(LATENT_SIZE),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        self.latent_refine = nn.GRUCell(LATENT_SIZE, LATENT_SIZE)
        self.latent_norm = nn.LayerNorm(LATENT_SIZE)

        self.final = nn.Sequential(
            nn.Linear(LATENT_SIZE, LATENT_SIZE),
            nn.LayerNorm(LATENT_SIZE),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

    def forward(self, public_embeddings):
        z = self.project(public_embeddings)

        h = z
        x = z

        for _ in range(LATENT_INTERNAL_STEPS):
            h = self.latent_refine(x, h)
            h = self.latent_norm(h)
            x = h

        return self.final(h)


# ============================================================
# RETRIEVAL MEMORY ENCODER
# ============================================================

class RetrievalMemoryEncoder(nn.Module):
    def __init__(self, answer_vocab_size):
        super().__init__()

        self.token_embedding = nn.Embedding(
            answer_vocab_size,
            ANSWER_EMBED_SIZE,
            padding_idx=PAD_ID,
        )

        self.answer_gru = nn.GRU(
            input_size=ANSWER_EMBED_SIZE,
            hidden_size=LATENT_SIZE // 2,
            batch_first=True,
            bidirectional=True,
        )

        self.memory_projection = nn.Sequential(
            nn.Linear(LATENT_SIZE, LATENT_SIZE),
            nn.LayerNorm(LATENT_SIZE),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

    def forward(self, retrieved_answer_ids):
        B, K, T = retrieved_answer_ids.shape

        flat = retrieved_answer_ids.reshape(B * K, T)

        emb = self.token_embedding(flat)
        out, _ = self.answer_gru(emb)

        mask = (flat != PAD_ID).float().unsqueeze(-1)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        memory = self.memory_projection(pooled)
        memory = memory.reshape(B, K, LATENT_SIZE)

        valid = (retrieved_answer_ids != PAD_ID).any(dim=-1)

        return memory, valid


# ============================================================
# PREDICTION MODEL WITH CROSS-ATTENTION
# ============================================================

class AttentivePredictionModel(nn.Module):
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

        self.query_projection = nn.Linear(PRED_HIDDEN, LATENT_SIZE)

        self.output = nn.Sequential(
            nn.Linear(PRED_HIDDEN + LATENT_SIZE + LATENT_SIZE, PRED_HIDDEN),
            nn.LayerNorm(PRED_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(PRED_HIDDEN, answer_vocab_size),
        )

        self.gate = nn.Sequential(
            nn.Linear(PRED_HIDDEN + LATENT_SIZE + LATENT_SIZE, LATENT_SIZE),
            nn.Sigmoid(),
        )

    def attend(self, h, memory, memory_valid):
        query = self.query_projection(h).unsqueeze(-1)

        scores = torch.bmm(memory, query).squeeze(-1)
        scores = scores / math.sqrt(LATENT_SIZE)
        scores = scores.masked_fill(~memory_valid, -1e9)

        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        context = (memory * weights).sum(dim=1)

        return context

    def forward(self, z, memory, memory_valid, answer_ids):
        B, T = answer_ids.shape

        h = torch.tanh(self.hidden_init(z))

        prev_ids = answer_ids[:, 0]
        logits = []

        context = torch.zeros_like(z)

        for t in range(1, T):
            emb = self.answer_embedding(prev_ids)

            inp = torch.cat([emb, z, context], dim=-1)
            h = self.gru(inp, h)

            context = self.attend(h, memory, memory_valid)

            gate = self.gate(torch.cat([h, z, context], dim=-1))
            mixed_context = gate * context + (1.0 - gate) * z

            step_logits = self.output(torch.cat([h, z, mixed_context], dim=-1))
            logits.append(step_logits)

            prev_ids = answer_ids[:, t]

        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def generate(self, z, memory, memory_valid, max_len=MAX_GENERATE_LEN, temperature=0.9):
        B = z.shape[0]

        h = torch.tanh(self.hidden_init(z))

        prev_ids = torch.full(
            (B,),
            BOS_ID,
            dtype=torch.long,
            device=z.device,
        )

        outputs = []
        context = torch.zeros_like(z)

        for _ in range(max_len):
            emb = self.answer_embedding(prev_ids)

            inp = torch.cat([emb, z, context], dim=-1)
            h = self.gru(inp, h)

            context = self.attend(h, memory, memory_valid)

            gate = self.gate(torch.cat([h, z, context], dim=-1))
            mixed_context = gate * context + (1.0 - gate) * z

            logits = self.output(torch.cat([h, z, mixed_context], dim=-1))

            if temperature <= 0:
                next_ids = torch.argmax(logits, dim=-1)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)

            outputs.append(next_ids)
            prev_ids = next_ids

        return torch.stack(outputs, dim=1)


# ============================================================
# HELPERS
# ============================================================

def set_requires_grad(module, value):
    for p in module.parameters():
        p.requires_grad = value


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


def forward_loss(
    latent_model,
    memory_encoder,
    prediction_model,
    embeddings,
    retrieved_answer_ids,
    answer_ids,
    criterion,
):
    z = latent_model(embeddings)
    memory, memory_valid = memory_encoder(retrieved_answer_ids)

    logits = prediction_model(
        z=z,
        memory=memory,
        memory_valid=memory_valid,
        answer_ids=answer_ids,
    )

    target = answer_ids[:, 1:]

    loss = criterion(
        logits.reshape(-1, logits.size(-1)),
        target.reshape(-1),
    )

    return loss, logits


@torch.no_grad()
def evaluate(
    public_encoder,
    retrieval_rows,
    retrieval_bank,
    latent_model,
    memory_encoder,
    prediction_model,
    loader,
    answer_vocab,
    criterion,
):
    latent_model.eval()
    memory_encoder.eval()
    prediction_model.eval()

    total_loss = 0.0
    batches = 0

    tok_correct = 0
    tok_total = 0

    exact_correct = 0
    exact_total = 0

    for questions, answer_ids in loader:
        answer_ids = answer_ids.to(DEVICE)

        embeddings = encode_questions(public_encoder, questions)
        retrieved_indices, _ = retrieve_topk(embeddings, retrieval_bank, RETRIEVAL_TOP_K)

        retrieved_answer_ids = build_retrieved_answer_tensor(
            retrieved_indices,
            retrieval_rows,
            answer_vocab,
        )

        loss, logits = forward_loss(
            latent_model,
            memory_encoder,
            prediction_model,
            embeddings,
            retrieved_answer_ids,
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
    public_encoder,
    retrieval_rows,
    retrieval_bank,
    latent_model,
    memory_encoder,
    prediction_model,
    rows,
    answer_vocab,
    id_to_token,
    count=5,
):
    if not rows:
        return

    latent_model.eval()
    memory_encoder.eval()
    prediction_model.eval()

    samples = random.sample(rows, min(count, len(rows)))

    print("\nSAMPLES")

    for row in samples:
        question = row["question"]

        embeddings = encode_questions(public_encoder, [question])
        retrieved_indices, retrieved_scores = retrieve_topk(
            embeddings,
            retrieval_bank,
            RETRIEVAL_TOP_K,
        )

        retrieved_answer_ids = build_retrieved_answer_tensor(
            retrieved_indices,
            retrieval_rows,
            answer_vocab,
        )

        z = latent_model(embeddings)
        memory, memory_valid = memory_encoder(retrieved_answer_ids)

        pred_ids = prediction_model.generate(
            z=z,
            memory=memory,
            memory_valid=memory_valid,
            max_len=MAX_GENERATE_LEN,
            temperature=0,
        )[0].cpu().tolist()

        print("Q:", question)
        print("T:", row["answer"])
        print("P:", decode_answer(pred_ids, id_to_token))
        print("Retrieved:")

        for idx, score in zip(
            retrieved_indices[0].detach().cpu().tolist(),
            retrieved_scores[0].detach().cpu().tolist(),
        ):
            print(f"  {score:.3f} | {retrieval_rows[idx]['question']}")

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

    if not rows:
        raise RuntimeError(f"No rows loaded from {DATA_DIR}")

    if LIMIT > 0:
        random.shuffle(rows)
        rows = rows[:LIMIT]

    random.shuffle(rows)

    split = int(len(rows) * TRAIN_SPLIT)

    train_rows = rows[:split]
    val_rows = rows[split:]

    answer_vocab, id_to_token = build_answer_vocab(train_rows)

    train_ds = PublicEncoderRetrievalQADataset(train_rows, answer_vocab)
    val_ds = PublicEncoderRetrievalQADataset(val_rows, answer_vocab)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )

    public_encoder = SentenceTransformer(PUBLIC_ENCODER_NAME, device=DEVICE)
    public_encoder.eval()

    for p in public_encoder.parameters():
        p.requires_grad = False

    public_embed_size = public_encoder.get_sentence_embedding_dimension()

    retrieval_rows = train_rows

    retrieval_bank = build_embedding_matrix(
        public_encoder=public_encoder,
        rows=retrieval_rows,
        cache_path=OUT_DIR / "retrieval_question_embeddings.pt",
    )

    latent_model = PublicEmbeddingToLatentModel(
        public_embed_size=public_embed_size,
    ).to(DEVICE)

    memory_encoder = RetrievalMemoryEncoder(
        answer_vocab_size=len(answer_vocab),
    ).to(DEVICE)

    prediction_model = AttentivePredictionModel(
        answer_vocab_size=len(answer_vocab),
    ).to(DEVICE)

    latent_params = list(latent_model.parameters()) + list(memory_encoder.parameters())

    latent_optimizer = torch.optim.AdamW(
        latent_params,
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
    print("Public encoder:", PUBLIC_ENCODER_NAME)
    print("Public embed size:", public_embed_size)
    print("Answer vocab:", len(answer_vocab))
    print("Retrieval top K:", RETRIEVAL_TOP_K)
    print("Latent LR:", LATENT_LR)
    print("Prediction LR:", PRED_LR)
    print("Training schedule: prediction update x1, latent/memory update x2")

    for epoch in range(1, EPOCHS + 1):
        latent_model.train()
        memory_encoder.train()
        prediction_model.train()

        total_pred_loss = 0.0
        total_latent_loss = 0.0
        batches = 0

        for questions, answer_ids in train_loader:
            answer_ids = answer_ids.to(DEVICE)

            with torch.no_grad():
                embeddings = encode_questions(public_encoder, questions)
                retrieved_indices, _ = retrieve_topk(
                    embeddings,
                    retrieval_bank,
                    RETRIEVAL_TOP_K,
                )

                retrieved_answer_ids = build_retrieved_answer_tensor(
                    retrieved_indices,
                    retrieval_rows,
                    answer_vocab,
                )

            target = answer_ids[:, 1:]

            # ==================================================
            # STEP 1: prediction model update only
            # ==================================================

            set_requires_grad(latent_model, False)
            set_requires_grad(memory_encoder, False)
            set_requires_grad(prediction_model, True)

            latent_model.eval()
            memory_encoder.eval()
            prediction_model.train()

            pred_optimizer.zero_grad()

            with torch.no_grad():
                z = latent_model(embeddings)
                memory, memory_valid = memory_encoder(retrieved_answer_ids)

            logits = prediction_model(
                z=z,
                memory=memory,
                memory_valid=memory_valid,
                answer_ids=answer_ids,
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
            # STEP 2: latent + memory update only #1
            # ==================================================

            set_requires_grad(latent_model, True)
            set_requires_grad(memory_encoder, True)
            set_requires_grad(prediction_model, False)

            latent_model.train()
            memory_encoder.train()
            prediction_model.eval()

            latent_optimizer.zero_grad()

            latent_loss_1, _ = forward_loss(
                latent_model,
                memory_encoder,
                prediction_model,
                embeddings,
                retrieved_answer_ids,
                answer_ids,
                criterion,
            )

            latent_loss_1.backward()

            torch.nn.utils.clip_grad_norm_(
                latent_params,
                GRAD_CLIP,
            )

            latent_optimizer.step()

            # ==================================================
            # STEP 3: latent + memory update only #2
            # ==================================================

            latent_optimizer.zero_grad()

            latent_loss_2, _ = forward_loss(
                latent_model,
                memory_encoder,
                prediction_model,
                embeddings,
                retrieved_answer_ids,
                answer_ids,
                criterion,
            )

            latent_loss_2.backward()

            torch.nn.utils.clip_grad_norm_(
                latent_params,
                GRAD_CLIP,
            )

            latent_optimizer.step()

            total_pred_loss += pred_loss.item()
            total_latent_loss += (latent_loss_1.item() + latent_loss_2.item()) / 2.0
            batches += 1

        set_requires_grad(latent_model, True)
        set_requires_grad(memory_encoder, True)
        set_requires_grad(prediction_model, True)

        val = evaluate(
            public_encoder=public_encoder,
            retrieval_rows=retrieval_rows,
            retrieval_bank=retrieval_bank,
            latent_model=latent_model,
            memory_encoder=memory_encoder,
            prediction_model=prediction_model,
            loader=val_loader,
            answer_vocab=answer_vocab,
            criterion=criterion,
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
                "memory_encoder_state": memory_encoder.state_dict(),
                "prediction_model_state": prediction_model.state_dict(),
                "answer_vocab": answer_vocab,
                "id_to_token": id_to_token,
                "public_encoder_name": PUBLIC_ENCODER_NAME,
                "public_embed_size": public_embed_size,
                "retrieval_rows": retrieval_rows,
                "config": {
                    "max_answer_len": MAX_ANSWER_LEN,
                    "max_generate_len": MAX_GENERATE_LEN,
                    "max_answer_tokens": MAX_ANSWER_TOKENS,
                    "retrieval_top_k": RETRIEVAL_TOP_K,
                    "retrieval_text_max_tokens": RETRIEVAL_TEXT_MAX_TOKENS,
                    "latent_size": LATENT_SIZE,
                    "latent_internal_steps": LATENT_INTERNAL_STEPS,
                    "answer_embed_size": ANSWER_EMBED_SIZE,
                    "pred_hidden": PRED_HIDDEN,
                    "dropout": DROPOUT,
                },
            }

            torch.save(
                checkpoint,
                OUT_DIR / "alternating_latent_chunk_qa_public_encoder_retrieval.pt",
            )

            print("Saved best retrieval-attention model.")

        if epoch % 5 == 0:
            sample_predictions(
                public_encoder=public_encoder,
                retrieval_rows=retrieval_rows,
                retrieval_bank=retrieval_bank,
                latent_model=latent_model,
                memory_encoder=memory_encoder,
                prediction_model=prediction_model,
                rows=val_rows if val_rows else train_rows,
                answer_vocab=answer_vocab,
                id_to_token=id_to_token,
                count=5,
            )

    print("Done.")
    print(
        "Best checkpoint:",
        OUT_DIR / "alternating_latent_chunk_qa_public_encoder_retrieval.pt",
    )


if __name__ == "__main__":
    train()