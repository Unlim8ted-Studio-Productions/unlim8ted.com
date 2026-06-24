# train_fast_latent_cover_chunk_qa.py

import argparse
import json
import math
import random
import re
import time
import heapq
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================

SEED = 42
TRAIN_SPLIT = 0.90

DATA_DIR = Path("assets/data/specialized_QA")
SMART_QA_PATH = Path("tools/SmartMeatballQA.jsonl")
OUT_DIR = Path("tools/meatball ai/fast_latent_cover_chunk_qa_out")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 30
BATCH_SIZE = 64

PRED_LR = 1e-3
LATENT_LR = 3e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0

MAX_QUESTION_CHUNKS = 64
MAX_CHUNK_CHARS = 32

MAX_ANSWER_CHUNKS = 48
MAX_GENERATE_LEN = 48

MAX_CHARS = 8000
MAX_OUTPUT_CHUNKS = 24000

MAX_CANDIDATES = 350000
MAX_SELECTED_PHRASE_CHUNKS = 12000
MAX_CHUNK_WORDS = 8
MIN_CHUNK_OCCURRENCES = 2
MIN_CHUNK_GAIN = 4

PROGRESS_EVERY_SELECTED = 100
SAVE_EVERY_SELECTED = 1000

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

SPECIAL_OUTPUT_CHUNKS = [PAD, BOS, EOS, UNK]


# ============================================================
# RANDOM
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================
# FILE UTILS
# ============================================================


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_jsonl_file(path: Path):
    rows = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception as e:
                print(f"[skip bad json] {path} line {line_num}: {e}")

    return rows


# ============================================================
# TEXT
# ============================================================


def normalize(text):
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def tokenize(text):
    return re.findall(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)*|[^\w\s]", normalize(text))


def canonical_token(tok):
    if re.match(r"[A-Za-z0-9_]+(?:[-'][A-Za-z0-9_]+)*$", tok):
        return tok.lower()
    return tok


def canonical_tokens(tokens):
    return [canonical_token(t) for t in tokens]


def phrase_key(tokens):
    return " ".join(canonical_tokens(tokens))


def phrase_text(tokens):
    out = ""

    for tok in tokens:
        if tok in [".", ",", "!", "?", ":", ";", "%", ")", "]", "}"]:
            out = out.rstrip() + tok
        elif tok in ["(", "[", "{"]:
            out += tok
        elif tok.startswith(("'", "’")):
            out = out.rstrip() + tok
        else:
            if out and not out.endswith((" ", "(", "[", "{", "'", "’")):
                out += " "
            out += tok

    return out.strip()


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
        if c in seen:
            continue
        seen.add(c)
        unique.append(c)

    return unique[:MAX_QUESTION_CHUNKS]


def decode_chunk_texts(chunks):
    out = ""

    for chunk in chunks:
        chunk = str(chunk).strip()
        if not chunk:
            continue

        if chunk in [".", ",", "!", "?", ":", ";", "%", ")", "]", "}"]:
            out = out.rstrip() + chunk
        elif chunk in ["(", "[", "{"]:
            if out and not out.endswith(" "):
                out += " "
            out += chunk
        elif chunk.startswith(("'", "’")):
            out = out.rstrip() + chunk
        else:
            if out and not out.endswith((" ", "(", "[", "{", "'", "’")):
                out += " "
            out += chunk

    return re.sub(r"\s+", " ", out).strip()


# ============================================================
# DATA
# ============================================================


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


def load_dataset(data_dir: Path, smart_qa_path: Path):
    rows = []

    paths = sorted(data_dir.rglob("*.jsonl"))

    print()
    print("Loading specialized QA files...")
    for path in paths:
        before = len(rows)

        for raw in load_jsonl_file(path):
            q, a = extract_qa(raw)
            if q and a:
                rows.append({"question": q, "answer": a, "source": str(path)})

        print(f"{path.name}: {len(rows) - before} rows")

    if smart_qa_path.exists():
        before = len(rows)

        for raw in load_jsonl_file(smart_qa_path):
            q, a = extract_qa(raw)
            if q and a:
                rows.append({"question": q, "answer": a, "source": str(smart_qa_path)})

        print(f"{smart_qa_path.name}: {len(rows) - before} rows")

    cleaned = []
    seen = set()

    for row in rows:
        key = row["question"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(row)

    print(f"combined rows before dedupe: {len(rows)}")
    print(f"combined rows after dedupe:  {len(cleaned)}")

    return cleaned


# ============================================================
# QUESTION CHAR VOCAB
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


# ============================================================
# FAST COVER CHUNK MINING
# ============================================================


def build_candidate_occurrences_fast(rows, max_chunk_words):
    t0 = time.time()

    tokenized_answers = [tokenize(row["answer"]) for row in rows]

    counts = Counter()
    readable = {}

    print()
    print("[1] counting candidate phrases...")

    for answer_idx, tokens in enumerate(tokenized_answers, start=1):
        max_n = min(max_chunk_words, len(tokens))

        for n in range(2, max_n + 1):
            for start in range(len(tokens) - n + 1):
                span = tokens[start : start + n]

                if not any(re.match(r"[A-Za-z0-9_]+", t) for t in span):
                    continue

                key = phrase_key(span)
                counts[key] += 1

                if key not in readable:
                    readable[key] = phrase_text(span)

        if answer_idx % 25000 == 0:
            print(
                f"  counted answers: {answer_idx:,}/{len(rows):,} | unique={len(counts):,}"
            )

    print(f"  raw unique candidates: {len(counts):,}")

    keep_keys = []

    for key, count in counts.items():
        if count < MIN_CHUNK_OCCURRENCES:
            continue

        length = len(key.split(" "))
        score = count * (length**1.35)
        keep_keys.append((score, key))

    keep_keys.sort(reverse=True)
    keep_keys = keep_keys[:MAX_CANDIDATES]
    keep_set = {key for _, key in keep_keys}

    print(f"  kept candidates: {len(keep_set):,}")

    occurrences = defaultdict(list)

    print("[2] building kept occurrence lists...")

    for answer_idx, tokens in enumerate(tokenized_answers):
        max_n = min(max_chunk_words, len(tokens))

        for n in range(2, max_n + 1):
            for start in range(len(tokens) - n + 1):
                span = tokens[start : start + n]
                key = phrase_key(span)

                if key in keep_set:
                    occurrences[key].append((answer_idx, start, start + n))

        if (answer_idx + 1) % 25000 == 0:
            occ_total = sum(len(v) for v in occurrences.values())
            print(
                f"  occurrence answers: {answer_idx + 1:,}/{len(rows):,} | occ={occ_total:,}"
            )

    print(f"candidate prep time: {(time.time() - t0) / 60:.1f}m")

    return tokenized_answers, occurrences, readable, counts


def count_uncovered_gain_fast(occ_list, covered_spans):
    gain = 0

    for answer_idx, start, end in occ_list:
        covered = covered_spans[answer_idx]

        for pos in range(start, end):
            if pos not in covered:
                gain += 1

    return gain


def mark_covered_fast(occ_list, covered_spans):
    for answer_idx, start, end in occ_list:
        covered_spans[answer_idx].update(range(start, end))


def build_fast_cover_chunks(rows, out_dir: Path, max_chunk_words=MAX_CHUNK_WORDS):
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    tokenized_answers, occurrences, readable, phrase_counts = (
        build_candidate_occurrences_fast(
            rows,
            max_chunk_words,
        )
    )

    covered_spans = [set() for _ in tokenized_answers]
    selected_phrase_keys = []
    selected_set = set()

    heap = []

    print()
    print("[3] initializing heap...")

    for key, occ_list in occurrences.items():
        if len(occ_list) < MIN_CHUNK_OCCURRENCES:
            continue

        length = len(key.split(" "))
        approx_gain = len(occ_list) * length
        heapq.heappush(heap, (-approx_gain, -length, -len(occ_list), key))

    print(f"heap candidates: {len(heap):,}")

    print()
    print("[4] greedy lazy cover selection...")

    last_save = time.time()

    while heap and len(selected_phrase_keys) < MAX_SELECTED_PHRASE_CHUNKS:
        _, _, _, key = heapq.heappop(heap)

        if key in selected_set:
            continue

        occ_list = occurrences[key]
        length = len(key.split(" "))

        real_gain = count_uncovered_gain_fast(occ_list, covered_spans)

        if real_gain < MIN_CHUNK_GAIN:
            continue

        # Lazy greedy check:
        # Reinsert with true gain unless it still looks like the best available.
        if heap:
            next_best_estimate = -heap[0][0]
            if real_gain < next_best_estimate:
                heapq.heappush(heap, (-real_gain, -length, -len(occ_list), key))
                continue

        selected_set.add(key)
        selected_phrase_keys.append(key)
        mark_covered_fast(occ_list, covered_spans)

        selected_count = len(selected_phrase_keys)

        if selected_count % PROGRESS_EVERY_SELECTED == 0:
            elapsed = time.time() - t0
            print(
                f"  selected={selected_count:,} | "
                f"gain={real_gain:,} | "
                f"len={length} | "
                f"heap={len(heap):,} | "
                f"elapsed={elapsed / 3600:.2f}h"
            )

        if selected_count % SAVE_EVERY_SELECTED == 0 or time.time() - last_save > 1800:
            save_json(
                out_dir / "partial_selected_phrase_keys.json",
                selected_phrase_keys,
            )
            last_save = time.time()
            print(f"  [saved partial] selected={selected_count:,}")

    print()
    print("[5] building single-token fallback chunks...")

    single_counter = Counter()
    single_text = {}

    for answer_idx, tokens in enumerate(tokenized_answers):
        covered = covered_spans[answer_idx]

        for pos, token in enumerate(tokens):
            if pos in covered:
                continue

            key = phrase_key([token])
            single_counter[key] += 1
            single_text[key] = token

    output_chunks = []

    for special in SPECIAL_OUTPUT_CHUNKS:
        output_chunks.append(
            {
                "id": len(output_chunks),
                "key": special,
                "text": special,
                "special": True,
                "count": 0,
                "length": 1,
            }
        )

    selected_phrase_keys.sort(
        key=lambda k: (len(k.split(" ")), phrase_counts[k], len(readable[k])),
        reverse=True,
    )

    for key in selected_phrase_keys:
        if len(output_chunks) >= MAX_OUTPUT_CHUNKS:
            break

        output_chunks.append(
            {
                "id": len(output_chunks),
                "key": key,
                "text": readable[key],
                "special": False,
                "count": phrase_counts[key],
                "length": len(key.split(" ")),
            }
        )

    for key, count in single_counter.most_common():
        if len(output_chunks) >= MAX_OUTPUT_CHUNKS:
            break

        output_chunks.append(
            {
                "id": len(output_chunks),
                "key": key,
                "text": single_text[key],
                "special": False,
                "count": count,
                "length": 1,
            }
        )

    save_json(out_dir / "output_chunks.json", output_chunks)
    save_json(out_dir / "selected_phrase_keys.json", selected_phrase_keys)

    stats = {
        "rows": len(rows),
        "output_chunks": len(output_chunks),
        "selected_phrase_chunks": len(selected_phrase_keys),
        "single_fallback_chunks": len(
            [c for c in output_chunks if not c.get("special") and c["length"] == 1]
        ),
        "elapsed_seconds": time.time() - t0,
        "max_candidates": MAX_CANDIDATES,
        "max_selected_phrase_chunks": MAX_SELECTED_PHRASE_CHUNKS,
        "max_output_chunks": MAX_OUTPUT_CHUNKS,
    }

    save_json(out_dir / "chunk_mining_stats.json", stats)

    print()
    print("[saved] output_chunks.json")
    print("[saved] selected_phrase_keys.json")
    print("[saved] chunk_mining_stats.json")
    print(f"chunk mining elapsed: {stats['elapsed_seconds'] / 3600:.2f}h")

    return output_chunks


def build_chunk_lookup(output_chunks):
    key_to_id = {}
    max_len = 1

    for ch in output_chunks:
        key_to_id[ch["key"]] = ch["id"]
        max_len = max(max_len, int(ch["length"]))

    return key_to_id, max_len


def encode_answer_to_chunks(answer, key_to_id, max_chunk_words):
    tokens = tokenize(answer)
    canon = canonical_tokens(tokens)

    ids = [BOS_ID]
    i = 0

    while i < len(canon):
        matched = False
        max_len = min(max_chunk_words, len(canon) - i)

        for n in range(max_len, 0, -1):
            key = " ".join(canon[i : i + n])

            if key in key_to_id:
                ids.append(key_to_id[key])
                i += n
                matched = True
                break

        if not matched:
            ids.append(UNK_ID)
            i += 1

        if len(ids) >= MAX_ANSWER_CHUNKS - 1:
            break

    ids.append(EOS_ID)

    while len(ids) < MAX_ANSWER_CHUNKS:
        ids.append(PAD_ID)

    return torch.tensor(ids[:MAX_ANSWER_CHUNKS], dtype=torch.long)


def decode_answer_chunk_ids(ids, output_chunks):
    texts = []

    for idx in ids:
        idx = int(idx)

        if idx == EOS_ID:
            break

        if idx in (PAD_ID, BOS_ID):
            continue

        if idx == UNK_ID:
            continue

        if 0 <= idx < len(output_chunks):
            texts.append(output_chunks[idx]["text"])

    return decode_chunk_texts(texts)


# ============================================================
# DATASET
# ============================================================


class LatentCoverChunkDataset(Dataset):
    def __init__(self, rows, char_vocab, key_to_id, max_chunk_words):
        self.rows = rows
        self.char_vocab = char_vocab
        self.key_to_id = key_to_id
        self.max_chunk_words = max_chunk_words

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        q_chunks = encode_question_chunks(row["question"], self.char_vocab)
        answer_ids = encode_answer_to_chunks(
            row["answer"],
            self.key_to_id,
            self.max_chunk_words,
        )

        return q_chunks, answer_ids


# ============================================================
# MODELS
# ============================================================


class ChunkToLatentModel(nn.Module):
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

        self.chunk_attention = nn.Sequential(
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

        mask = (flat != PAD_ID).float().unsqueeze(-1)
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        chunk_vecs = self.chunk_to_latent(pooled)
        chunk_vecs = chunk_vecs.reshape(B, G, LATENT_SIZE)

        return chunk_vecs

    def forward(self, chunk_ids):
        chunk_vecs = self.encode_chunks(chunk_ids)

        valid_chunk_mask = (chunk_ids != PAD_ID).any(dim=-1)

        scores = self.chunk_attention(chunk_vecs).squeeze(-1)
        scores = scores.masked_fill(~valid_chunk_mask, -1e9)

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
    def __init__(self, output_vocab_size):
        super().__init__()

        self.answer_embedding = nn.Embedding(
            output_vocab_size,
            ANSWER_EMBED_SIZE,
            padding_idx=PAD_ID,
        )

        self.hidden_init = nn.Linear(LATENT_SIZE, PRED_HIDDEN)

        self.gru = nn.GRUCell(
            input_size=ANSWER_EMBED_SIZE + LATENT_SIZE,
            hidden_size=PRED_HIDDEN,
        )

        self.output = nn.Sequential(
            nn.Linear(PRED_HIDDEN + LATENT_SIZE, PRED_HIDDEN),
            nn.LayerNorm(PRED_HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(PRED_HIDDEN, output_vocab_size),
        )

    def forward(self, z, answer_ids):
        B, T = answer_ids.shape

        h = torch.tanh(self.hidden_init(z))
        prev_ids = answer_ids[:, 0]

        logits = []

        for t in range(1, T):
            emb = self.answer_embedding(prev_ids)
            inp = torch.cat([emb, z], dim=-1)
            h = self.gru(inp, h)

            step_logits = self.output(torch.cat([h, z], dim=-1))
            logits.append(step_logits)

            prev_ids = answer_ids[:, t]

        return torch.stack(logits, dim=1)

    @torch.no_grad()
    def generate(self, z, max_len=MAX_GENERATE_LEN):
        B = z.shape[0]

        h = torch.tanh(self.hidden_init(z))

        prev_ids = torch.full(
            (B,),
            BOS_ID,
            dtype=torch.long,
            device=z.device,
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
# TRAIN HELPERS
# ============================================================


def set_requires_grad(module, value):
    for p in module.parameters():
        p.requires_grad = value


def forward_loss(latent_model, prediction_model, q_chunks, answer_ids, criterion):
    z = latent_model(q_chunks)
    logits = prediction_model(z, answer_ids)
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
    latent_model,
    prediction_model,
    rows,
    char_vocab,
    output_chunks,
    count=3,
):
    if not rows:
        return

    latent_model.eval()
    prediction_model.eval()

    samples = random.sample(rows, min(count, len(rows)))

    print()
    print("SAMPLES")

    for row in samples:
        q_chunks = encode_question_chunks(row["question"], char_vocab)
        q_chunks = q_chunks.unsqueeze(0).to(DEVICE)

        z = latent_model(q_chunks)
        pred_ids = (
            prediction_model.generate(z, max_len=MAX_GENERATE_LEN)[0].cpu().tolist()
        )

        print("Q:", row["question"])
        print("T:", row["answer"])
        print("P:", decode_answer_chunk_ids(pred_ids, output_chunks))
        print("-" * 60)


# ============================================================
# MAIN TRAIN
# ============================================================


def train(args):
    OUT_DIR_FINAL = Path(args.out_dir)
    OUT_DIR_FINAL.mkdir(parents=True, exist_ok=True)

    rows = load_dataset(Path(args.data_dir), Path(args.smart_qa_path))

    if args.limit > 0:
        random.shuffle(rows)
        rows = rows[: args.limit]

    if len(rows) < 10:
        raise RuntimeError(f"Need at least 10 rows, got {len(rows)}")

    random.shuffle(rows)

    split = int(len(rows) * TRAIN_SPLIT)

    train_rows = rows[:split]
    val_rows = rows[split:] or rows[:]

    print()
    print("Device:", DEVICE)
    print("Rows:", len(rows))
    print("Train:", len(train_rows))
    print("Val:", len(val_rows))

    char_vocab, id_to_char = build_char_vocab(train_rows)
    save_json(OUT_DIR_FINAL / "char_vocab.json", char_vocab)
    save_json(OUT_DIR_FINAL / "id_to_char.json", id_to_char)

    if args.reuse_chunks and (OUT_DIR_FINAL / "output_chunks.json").exists():
        print("[reuse] loading existing output_chunks.json")
        with (OUT_DIR_FINAL / "output_chunks.json").open("r", encoding="utf-8") as f:
            output_chunks = json.load(f)
    else:
        output_chunks = build_fast_cover_chunks(
            train_rows,
            OUT_DIR_FINAL,
            max_chunk_words=args.max_chunk_words,
        )

    key_to_id, max_chunk_words_used = build_chunk_lookup(output_chunks)

    encoded_lengths = []
    unk_count = 0

    for row in rows:
        ids = encode_answer_to_chunks(
            row["answer"],
            key_to_id,
            max_chunk_words_used,
        ).tolist()

        stripped = [x for x in ids if x not in (PAD_ID, BOS_ID, EOS_ID)]
        encoded_lengths.append(len(stripped))
        unk_count += sum(1 for x in stripped if x == UNK_ID)

    print()
    print("Chunk stats:")
    print(f"output chunk vocab size:    {len(output_chunks):,}")
    print(
        f"avg chunks per answer:      {sum(encoded_lengths) / max(1, len(encoded_lengths)):.2f}"
    )
    print(f"max chunks per answer:      {max(encoded_lengths)}")
    print(f"encoded UNK uses:           {unk_count:,}")

    save_json(
        OUT_DIR_FINAL / "encoding_stats.json",
        {
            "rows": len(rows),
            "output_chunk_vocab_size": len(output_chunks),
            "avg_chunks_per_answer": sum(encoded_lengths)
            / max(1, len(encoded_lengths)),
            "max_chunks_per_answer": max(encoded_lengths),
            "encoded_unk_uses": unk_count,
        },
    )

    train_ds = LatentCoverChunkDataset(
        train_rows,
        char_vocab,
        key_to_id,
        max_chunk_words_used,
    )

    val_ds = LatentCoverChunkDataset(
        val_rows,
        char_vocab,
        key_to_id,
        max_chunk_words_used,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    latent_model = ChunkToLatentModel(char_vocab_size=len(char_vocab)).to(DEVICE)
    prediction_model = PredictionModel(output_vocab_size=len(output_chunks)).to(DEVICE)

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

    print()
    print("Char vocab:", len(char_vocab))
    print("Output chunk vocab:", len(output_chunks))
    print("Latent LR:", LATENT_LR)
    print("Prediction LR:", PRED_LR)
    print("Training schedule: prediction update x1, latent update x2")

    for epoch in range(1, args.epochs + 1):
        latent_model.train()
        prediction_model.train()

        total_pred_loss = 0.0
        total_latent_loss = 0.0
        batches = 0

        t_epoch = time.time()

        for q_chunks, answer_ids in train_loader:
            q_chunks = q_chunks.to(DEVICE)
            answer_ids = answer_ids.to(DEVICE)

            target = answer_ids[:, 1:]

            # Prediction model update
            set_requires_grad(latent_model, False)
            set_requires_grad(prediction_model, True)

            latent_model.eval()
            prediction_model.train()

            pred_optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                z = latent_model(q_chunks)

            logits = prediction_model(z, answer_ids)

            pred_loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
            )

            pred_loss.backward()
            torch.nn.utils.clip_grad_norm_(prediction_model.parameters(), GRAD_CLIP)
            pred_optimizer.step()

            # Latent model update #1
            set_requires_grad(latent_model, True)
            set_requires_grad(prediction_model, False)

            latent_model.train()
            prediction_model.eval()

            latent_optimizer.zero_grad(set_to_none=True)

            z = latent_model(q_chunks)
            logits = prediction_model(z, answer_ids)

            latent_loss_1 = criterion(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
            )

            latent_loss_1.backward()
            torch.nn.utils.clip_grad_norm_(latent_model.parameters(), GRAD_CLIP)
            latent_optimizer.step()

            # Latent model update #2
            latent_optimizer.zero_grad(set_to_none=True)

            z = latent_model(q_chunks)
            logits = prediction_model(z, answer_ids)

            latent_loss_2 = criterion(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
            )

            latent_loss_2.backward()
            torch.nn.utils.clip_grad_norm_(latent_model.parameters(), GRAD_CLIP)
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
            f"exact_acc {val['exact_acc'] * 100:.2f}% | "
            f"time {(time.time() - t_epoch) / 60:.1f}m"
        )

        if val["loss"] < best_val:
            best_val = val["loss"]

            checkpoint = {
                "latent_model_state": latent_model.state_dict(),
                "prediction_model_state": prediction_model.state_dict(),
                "char_vocab": char_vocab,
                "id_to_char": id_to_char,
                "output_chunks": output_chunks,
                "config": {
                    "max_question_chunks": MAX_QUESTION_CHUNKS,
                    "max_chunk_chars": MAX_CHUNK_CHARS,
                    "max_answer_chunks": MAX_ANSWER_CHUNKS,
                    "max_generate_len": MAX_GENERATE_LEN,
                    "char_embed_size": CHAR_EMBED_SIZE,
                    "chunk_hidden": CHUNK_HIDDEN,
                    "latent_size": LATENT_SIZE,
                    "latent_internal_steps": LATENT_INTERNAL_STEPS,
                    "answer_embed_size": ANSWER_EMBED_SIZE,
                    "pred_hidden": PRED_HIDDEN,
                    "dropout": DROPOUT,
                    "chunk_strategy": "fast_lazy_greedy_cover_chunks",
                    "max_chunk_words_used": max_chunk_words_used,
                },
            }

            torch.save(checkpoint, OUT_DIR_FINAL / "fast_latent_cover_chunk_qa.pt")
            print("Saved best model.")

        if epoch % 5 == 0:
            sample_predictions(
                latent_model,
                prediction_model,
                val_rows,
                char_vocab,
                output_chunks,
                count=3,
            )

    print()
    print("Done.")
    print("Best checkpoint:", OUT_DIR_FINAL / "fast_latent_cover_chunk_qa.pt")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--smart_qa_path", default=str(SMART_QA_PATH))
    parser.add_argument("--out_dir", default=str(OUT_DIR))
    parser.add_argument("--limit", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)

    parser.add_argument("--max_chunk_words", type=int, default=MAX_CHUNK_WORDS)

    parser.add_argument(
        "--reuse_chunks",
        action="store_true",
        help="Reuse existing output_chunks.json in out_dir instead of mining again.",
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
