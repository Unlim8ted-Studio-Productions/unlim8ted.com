# test_alternating_latent_chunk_qa.py

import re
import torch
import torch.nn as nn

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = r"tools\meatball ai\alternating_latent_chunk_qa_out\alternating_latent_chunk_qa.pt"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"

# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

char_vocab = checkpoint["char_vocab"]
id_to_token = checkpoint["id_to_token"]
answer_vocab = checkpoint["answer_vocab"]

cfg = checkpoint["config"]

MAX_QUESTION_CHUNKS = cfg["max_question_chunks"]
MAX_CHUNK_CHARS = cfg["max_chunk_chars"]
MAX_GENERATE_LEN = cfg["max_generate_len"]

CHAR_EMBED_SIZE = cfg["char_embed_size"]
CHUNK_HIDDEN = cfg["chunk_hidden"]
LATENT_SIZE = cfg["latent_size"]
LATENT_INTERNAL_STEPS = cfg["latent_internal_steps"]
ANSWER_EMBED_SIZE = cfg["answer_embed_size"]
PRED_HIDDEN = cfg["pred_hidden"]
DROPOUT = cfg["dropout"]

# ============================================================
# TEXT HELPERS
# ============================================================

def normalize(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text

def tokenize(text):
    return re.findall(r"[\w']+|[^\w\s]", normalize(text))

def word_ngrams(tokens, ns=(1, 2, 3)):
    tokens = [t.lower() for t in tokens]
    out = []

    for n in ns:
        for i in range(len(tokens) - n + 1):
            out.append(" ".join(tokens[i:i+n]))

    return out

def char_ngrams(text, ns=(3, 4, 5)):
    text = normalize(text).lower().replace(" ", "_")
    out = []

    for n in ns:
        for i in range(len(text) - n + 1):
            out.append(text[i:i+n])

    return out

def question_chunks(text):
    toks = tokenize(text)

    chunks = []
    chunks.extend(word_ngrams(toks))
    chunks.extend(char_ngrams(text))

    seen = set()
    unique = []

    for c in chunks:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique[:MAX_QUESTION_CHUNKS]

def encode_question_chunks(text):
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

def decode_answer(ids):
    tokens = []

    for idx in ids:

        if idx == EOS_ID:
            break

        if idx in (PAD_ID, BOS_ID):
            continue

        tokens.append(id_to_token.get(int(idx), UNK))

    text = " ".join(tokens)

    text = re.sub(r"\s+([.,!?;:])", r"\1", text)

    return text.strip()

# ============================================================
# MODELS
# ============================================================

class ChunkToLatentModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.char_embedding = nn.Embedding(
            len(char_vocab),
            CHAR_EMBED_SIZE,
            padding_idx=PAD_ID
        )

        self.chunk_gru = nn.GRU(
            CHAR_EMBED_SIZE,
            CHUNK_HIDDEN,
            batch_first=True,
            bidirectional=True
        )

        self.chunk_to_latent = nn.Sequential(
            nn.Linear(CHUNK_HIDDEN * 2, LATENT_SIZE),
            nn.LayerNorm(LATENT_SIZE),
            nn.GELU(),
            nn.Dropout(DROPOUT)
        )

        self.chunk_attention = nn.Sequential(
            nn.Linear(LATENT_SIZE, LATENT_SIZE),
            nn.Tanh(),
            nn.Linear(LATENT_SIZE, 1)
        )

        self.latent_refine = nn.GRUCell(
            LATENT_SIZE,
            LATENT_SIZE
        )

        self.latent_norm = nn.LayerNorm(LATENT_SIZE)

        self.final = nn.Sequential(
            nn.Linear(LATENT_SIZE, LATENT_SIZE),
            nn.LayerNorm(LATENT_SIZE),
            nn.GELU(),
            nn.Dropout(DROPOUT)
        )

    def encode_chunks(self, chunk_ids):

        B, G, C = chunk_ids.shape

        flat = chunk_ids.reshape(B * G, C)

        emb = self.char_embedding(flat)

        out, _ = self.chunk_gru(emb)

        mask = (flat != PAD_ID).float().unsqueeze(-1)

        pooled = (
            (out * mask).sum(dim=1)
            / mask.sum(dim=1).clamp(min=1.0)
        )

        chunk_vecs = self.chunk_to_latent(pooled)

        return chunk_vecs.reshape(B, G, LATENT_SIZE)

    def forward(self, chunk_ids):

        chunk_vecs = self.encode_chunks(chunk_ids)

        valid_mask = (chunk_ids != PAD_ID).any(dim=-1)

        scores = self.chunk_attention(chunk_vecs).squeeze(-1)
        scores = scores.masked_fill(~valid_mask, -1e9)

        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)

        z = (chunk_vecs * weights).sum(dim=1)

        h = z
        x = z

        for _ in range(LATENT_INTERNAL_STEPS):
            h = self.latent_refine(x, h)
            h = self.latent_norm(h)
            x = h

        return self.final(h)

class PredictionModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.answer_embedding = nn.Embedding(
            len(answer_vocab),
            ANSWER_EMBED_SIZE,
            padding_idx=PAD_ID
        )

        self.hidden_init = nn.Linear(
            LATENT_SIZE,
            PRED_HIDDEN
        )

        self.gru = nn.GRUCell(
            ANSWER_EMBED_SIZE + LATENT_SIZE,
            PRED_HIDDEN
        )

        self.output = nn.Sequential(
            nn.Linear(PRED_HIDDEN + LATENT_SIZE, PRED_HIDDEN),
            nn.LayerNorm(PRED_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(PRED_HIDDEN, len(answer_vocab))
        )

    @torch.no_grad()
    def generate(self, z, max_len=MAX_GENERATE_LEN):

        B = z.shape[0]

        h = torch.tanh(self.hidden_init(z))

        prev_ids = torch.full(
            (B,),
            BOS_ID,
            dtype=torch.long,
            device=z.device
        )

        outputs = []

        for _ in range(max_len):

            emb = self.answer_embedding(prev_ids)

            inp = torch.cat([emb, z], dim=-1)

            h = self.gru(inp, h)

            logits = self.output(torch.cat([h, z], dim=-1))

            next_ids = torch.argmax(logits, dim=-1)

            outputs.append(next_ids)

            prev_ids = next_ids

        return torch.stack(outputs, dim=1)

# ============================================================
# LOAD WEIGHTS
# ============================================================

latent_model = ChunkToLatentModel().to(DEVICE)
prediction_model = PredictionModel().to(DEVICE)

latent_model.load_state_dict(
    checkpoint["latent_model_state"]
)

prediction_model.load_state_dict(
    checkpoint["prediction_model_state"]
)

latent_model.eval()
prediction_model.eval()

print("Model loaded successfully.")

# ============================================================
# INTERACTIVE CHAT
# ============================================================

while True:

    question = input("\nQuestion: ").strip()

    if question.lower() in ["quit", "exit"]:
        break

    x = encode_question_chunks(question)
    x = x.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        z = latent_model(x)
        pred = prediction_model.generate(z)[0]

    answer = decode_answer(pred.cpu().tolist())

    print("Answer:", answer)