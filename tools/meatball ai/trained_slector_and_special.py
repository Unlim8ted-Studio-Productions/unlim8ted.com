import json
import math
import random
import re
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================

SEED = 42
VAL_SPLIT = 0.12

INPUT_NGRAMS = (1, 2, 3)

MAX_INPUT_VOCAB_SIZE = 12000
MIN_INPUT_TOKEN_FREQ = 1

MAX_OUTPUT_VOCAB_SIZE = 12000
MAX_SINGLE_CHUNKS = 5000
MAX_PHRASE_CHUNKS = 7000

MIN_CHUNK_FREQ = 3
MIN_CHUNK_WORDS = 2
MAX_CHUNK_WORDS = 6

MAX_OUTPUT_CHUNKS = 40

BATCH_SIZE = 64

SELECTOR_EPOCHS = 50
TOPIC_EPOCHS = 80

PATIENCE = 15
MIN_DELTA = 1e-4

SELECTOR_LR = 1e-3
TOPIC_LR = 8e-4

WEIGHT_DECAY = 2e-3
LABEL_SMOOTHING = 0.06
GRAD_CLIP = 1.0

HIDDEN_SIZE = 192
EMBED_SIZE = 128
DROPOUT = 0.35

EXPORT_ONNX = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# SPECIAL OUTPUT TOKENS
# ============================================================

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
# BASIC FILE UTILS
# ============================================================


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_jsonl(path: Path):
    rows = []

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    with path.open("r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, start=1):
            raw = line.rstrip("\n")
            line = raw.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print()
                print("BAD JSONL")
                print("file:", path)
                print("line:", line_num)
                print("error:", e)
                print("repr:", repr(line[:500]))
                print("around error:", repr(line[max(0, e.pos - 100) : e.pos + 100]))
                raise

            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()

            if not question or not answer:
                print(f"[skip] missing question/answer {path.name} line {line_num}")
                continue

            rows.append(
                {
                    "question": question,
                    "answer": answer,
                    "history": row.get("history", []),
                }
            )

    return rows


# ============================================================
# TOKENIZATION — SAME LOGIC AS OLD CHUNK SCRIPT
# ============================================================


def normalize_spaces(text: str) -> str:
    text = str(text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def answer_tokenize(text: str):
    text = normalize_spaces(text)
    return re.findall(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*|[^\w\s]", text)


def canonical_token(token: str) -> str:
    if re.match(r"[A-Za-z0-9_]+$", token):
        return token.lower()
    return token


def canonical_tokens(tokens):
    return [canonical_token(t) for t in tokens]


def input_normalize(text: str) -> str:
    text = str(text).lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[^a-z0-9_!?.,' -]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def input_tokenize(text: str):
    text = input_normalize(text)

    if not text:
        return []

    return text.split()


def make_input_ngrams(tokens, ngrams=INPUT_NGRAMS):
    feats = []

    for n in ngrams:
        if len(tokens) < n:
            continue

        for i in range(len(tokens) - n + 1):
            feats.append("_".join(tokens[i : i + n]))

    return feats


def row_to_input_text(row):
    question = row.get("question", "")

    history = row.get("history", [])
    if isinstance(history, list):
        history_text = " ".join(str(x) for x in history)
    else:
        history_text = str(history)

    return f"question: {question} history: {history_text}"


# ============================================================
# OUTPUT CHUNK MINING — SAME LOGIC AS OLD SCRIPT
# ============================================================


def phrase_key(tokens):
    return " ".join(canonical_tokens(tokens))


def phrase_readable_text(tokens):
    out = ""

    for tok in tokens:
        if tok in [".", ",", "!", "?", ":", ";", "%", ")", "]", "}"]:
            out = out.rstrip() + tok
        elif tok in ["(", "[", "{"]:
            out += tok
        elif tok in ["'", "’"]:
            out = out.rstrip() + tok
        else:
            if out and not out.endswith((" ", "(", "[", "{", "'", "’")):
                out += " "
            out += tok

    return out.strip()


def mine_common_chunks(rows):
    phrase_counts = Counter()
    readable_variants = defaultdict(Counter)

    for row in rows:
        answer = row.get("answer", "")
        tokens = answer_tokenize(answer)

        for n in range(MIN_CHUNK_WORDS, MAX_CHUNK_WORDS + 1):
            if len(tokens) < n:
                continue

            for i in range(len(tokens) - n + 1):
                span = tokens[i : i + n]

                if not any(re.match(r"[A-Za-z0-9_]+$", t) for t in span):
                    continue

                key = phrase_key(span)
                readable = phrase_readable_text(span)

                phrase_counts[key] += 1
                readable_variants[key][readable] += 1

    candidates = []

    for key, count in phrase_counts.items():
        if count < MIN_CHUNK_FREQ:
            continue

        toks = key.split(" ")
        length = len(toks)
        score = count * (length**1.35)

        readable = readable_variants[key].most_common(1)[0][0]

        candidates.append(
            {
                "key": key,
                "text": readable,
                "count": count,
                "length": length,
                "score": score,
            }
        )

    candidates.sort(key=lambda x: x["score"], reverse=True)

    max_phrase_chunks = max(0, MAX_OUTPUT_VOCAB_SIZE - len(SPECIAL_OUTPUT_CHUNKS))
    candidates = candidates[:max_phrase_chunks]

    return candidates


def collect_single_token_chunks(rows):
    readable_variants = defaultdict(Counter)

    for row in rows:
        tokens = answer_tokenize(row.get("answer", ""))

        for tok in tokens:
            key = phrase_key([tok])
            readable_variants[key][tok] += 1

    singles = []

    for key, variants in readable_variants.items():
        readable = variants.most_common(1)[0][0]

        singles.append(
            {
                "key": key,
                "text": readable,
                "count": sum(variants.values()),
                "length": 1,
                "score": sum(variants.values()),
            }
        )

    singles.sort(key=lambda x: x["count"], reverse=True)

    return singles


def build_output_chunks(rows):
    common_chunks = mine_common_chunks(rows)
    single_chunks = collect_single_token_chunks(rows)

    output_chunks = []

    for tok in SPECIAL_OUTPUT_CHUNKS:
        output_chunks.append(
            {
                "id": len(output_chunks),
                "key": tok,
                "text": tok,
                "special": True,
                "count": 0,
                "length": 1,
            }
        )

    used_keys = {c["key"] for c in output_chunks}

    single_added = 0

    for ch in single_chunks:
        if len(output_chunks) >= MAX_OUTPUT_VOCAB_SIZE:
            break

        if single_added >= MAX_SINGLE_CHUNKS:
            break

        if ch["key"] in used_keys:
            continue

        used_keys.add(ch["key"])
        single_added += 1

        output_chunks.append(
            {
                "id": len(output_chunks),
                "key": ch["key"],
                "text": ch["text"],
                "special": False,
                "count": ch["count"],
                "length": ch["length"],
            }
        )

    phrase_added = 0

    for ch in common_chunks:
        if len(output_chunks) >= MAX_OUTPUT_VOCAB_SIZE:
            break

        if phrase_added >= MAX_PHRASE_CHUNKS:
            break

        if ch["key"] in used_keys:
            continue

        used_keys.add(ch["key"])
        phrase_added += 1

        output_chunks.append(
            {
                "id": len(output_chunks),
                "key": ch["key"],
                "text": ch["text"],
                "special": False,
                "count": ch["count"],
                "length": ch["length"],
            }
        )

    print(f"single chunks added: {single_added}")
    print(f"phrase chunks added: {phrase_added}")

    return output_chunks


def build_chunk_lookup(output_chunks):
    key_to_id = {}
    key_to_text = {}

    for ch in output_chunks:
        key_to_id[ch["key"]] = ch["id"]
        key_to_text[ch["key"]] = ch["text"]

    phrase_keys_by_len = defaultdict(list)

    for ch in output_chunks:
        if ch.get("special"):
            continue

        length = ch["length"]

        if length >= 1:
            phrase_keys_by_len[length].append(ch["key"])

    return key_to_id, key_to_text, phrase_keys_by_len


def encode_answer_to_chunks(answer, key_to_id):
    tokens = answer_tokenize(answer)
    canon = canonical_tokens(tokens)

    ids = []
    i = 0

    max_len = min(MAX_CHUNK_WORDS, len(canon))

    while i < len(canon):
        matched = False

        for n in range(min(max_len, len(canon) - i), 0, -1):
            key = " ".join(canon[i : i + n])

            if key in key_to_id:
                ids.append(key_to_id[key])
                i += n
                matched = True
                break

        if not matched:
            ids.append(UNK_ID)
            i += 1

        if len(ids) >= MAX_OUTPUT_CHUNKS:
            break

    ids.append(EOS_ID)

    while len(ids) < MAX_OUTPUT_CHUNKS + 1:
        ids.append(PAD_ID)

    return ids[: MAX_OUTPUT_CHUNKS + 1]


def join_chunk_texts(chunks):
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

    out = re.sub(r"\s+", " ", out).strip()
    return out


def decode_chunk_ids(ids, output_chunks):
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
            text = output_chunks[idx]["text"]
            texts.append(text)

    return join_chunk_texts(texts)


def strip_special(ids):
    out = []

    for idx in ids:
        idx = int(idx)

        if idx == EOS_ID:
            break

        if idx in (PAD_ID, BOS_ID):
            continue

        out.append(idx)

    return out


# ============================================================
# INPUT VOCAB — PER TOPIC FOR SPECIALISTS
# ============================================================


def build_input_vocab(rows):
    counter = Counter()

    for row in rows:
        text = row_to_input_text(row)
        tokens = input_tokenize(text)
        feats = make_input_ngrams(tokens)
        counter.update(feats)

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for feat, count in counter.most_common(MAX_INPUT_VOCAB_SIZE - len(vocab)):
        if count < MIN_INPUT_TOKEN_FREQ:
            continue

        vocab[feat] = len(vocab)

    return vocab


def vectorize_input(row, input_vocab):
    text = row_to_input_text(row)
    tokens = input_tokenize(text)
    feats = make_input_ngrams(tokens)

    x = torch.zeros(len(input_vocab), dtype=torch.float32)

    counts = Counter(feats)

    for feat, count in counts.items():
        idx = input_vocab.get(feat, input_vocab["<UNK>"])
        x[idx] = min(float(count), 5.0)

    return x


# ============================================================
# SELECTOR VOCAB
# ============================================================


def selector_text_from_question(question: str):
    return f"question: {question}"


def build_selector_vocab(selector_examples):
    counter = Counter()

    for question, topic in selector_examples:
        text = selector_text_from_question(question)
        tokens = input_tokenize(text)
        feats = make_input_ngrams(tokens)
        counter.update(feats)

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1,
    }

    for feat, count in counter.most_common(MAX_INPUT_VOCAB_SIZE - len(vocab)):
        if count < MIN_INPUT_TOKEN_FREQ:
            continue

        vocab[feat] = len(vocab)

    return vocab


def vectorize_selector_question(question, selector_vocab):
    text = selector_text_from_question(question)
    tokens = input_tokenize(text)
    feats = make_input_ngrams(tokens)

    x = torch.zeros(len(selector_vocab), dtype=torch.float32)

    counts = Counter(feats)

    for feat, count in counts.items():
        idx = selector_vocab.get(feat, selector_vocab["<UNK>"])
        x[idx] = min(float(count), 5.0)

    return x


# ============================================================
# DATASETS
# ============================================================


class MeatballChunkDataset(Dataset):
    def __init__(self, rows, input_vocab, key_to_id):
        self.rows = rows
        self.input_vocab = input_vocab
        self.key_to_id = key_to_id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        x = vectorize_input(row, self.input_vocab)

        y_ids = encode_answer_to_chunks(row.get("answer", ""), self.key_to_id)
        y = torch.tensor(y_ids, dtype=torch.long)

        return x, y


class SelectorDataset(Dataset):
    def __init__(self, examples, selector_vocab, topic_to_id):
        self.examples = examples
        self.selector_vocab = selector_vocab
        self.topic_to_id = topic_to_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        question, topic = self.examples[idx]

        x = vectorize_selector_question(question, self.selector_vocab)
        y = torch.tensor(self.topic_to_id[topic], dtype=torch.long)

        return x, y


# ============================================================
# MODELS
# ============================================================


class SelectorModel(nn.Module):
    def __init__(
        self, input_size, num_topics, hidden_size=HIDDEN_SIZE, dropout=DROPOUT
    ):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_topics),
        )

    def forward(self, x):
        return self.model(x)


class ChunkAnswerModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_vocab_size,
        hidden_size=HIDDEN_SIZE,
        embed_size=EMBED_SIZE,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_vocab_size = output_vocab_size
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
            nn.Dropout(dropout),
        )

        self.embedding = nn.Embedding(output_vocab_size, embed_size)

        self.decoder_cell = nn.GRUCell(embed_size, hidden_size)

        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_vocab_size),
        )

    def encode(self, x):
        return self.encoder(x)

    def decoder_step(self, prev_token, hidden):
        emb = self.embedding(prev_token)
        hidden = self.decoder_cell(emb, hidden)
        logits = self.output(hidden)
        return logits, hidden

    def forward(
        self,
        x,
        target=None,
        teacher_forcing=True,
        max_len=MAX_OUTPUT_CHUNKS + 1,
    ):
        batch_size = x.size(0)

        hidden = self.encode(x)

        prev_token = torch.full(
            (batch_size,),
            BOS_ID,
            dtype=torch.long,
            device=x.device,
        )

        logits_steps = []

        for t in range(max_len):
            logits, hidden = self.decoder_step(prev_token, hidden)
            logits_steps.append(logits.unsqueeze(1))

            if teacher_forcing and target is not None:
                prev_token = target[:, t]
            else:
                prev_token = torch.argmax(logits, dim=-1)

        return torch.cat(logits_steps, dim=1)


# ============================================================
# EVALUATION
# ============================================================


@torch.no_grad()
def evaluate_selector(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    batches = 0

    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += float(loss.item())
        batches += 1

        preds = torch.argmax(logits, dim=-1)

        correct += int((preds == y).sum().item())
        total += int(y.numel())

    return {
        "loss": total_loss / max(batches, 1),
        "acc": correct / max(total, 1),
    }


@torch.no_grad()
def evaluate_chunk_model(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    batches = 0

    token_correct = 0
    token_total = 0

    exact = 0
    rows = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(
            x,
            target=y,
            teacher_forcing=True,
            max_len=MAX_OUTPUT_CHUNKS + 1,
        )

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
        )

        total_loss += float(loss.item())
        batches += 1

        preds = torch.argmax(logits, dim=-1)
        mask = y != PAD_ID

        token_correct += int(((preds == y) & mask).sum().item())
        token_total += int(mask.sum().item())

        for pred_row, true_row in zip(preds.cpu().tolist(), y.cpu().tolist()):
            if strip_special(pred_row) == strip_special(true_row):
                exact += 1
            rows += 1

    return {
        "loss": total_loss / max(batches, 1),
        "token_acc": token_correct / max(token_total, 1),
        "exact_seq": exact / max(rows, 1),
    }


@torch.no_grad()
def predict_answer(
    model, row, input_vocab, output_chunks, max_len=MAX_OUTPUT_CHUNKS + 1
):
    model.eval()

    x = vectorize_input(row, input_vocab).unsqueeze(0).to(DEVICE)

    hidden = model.encode(x)

    prev_token = torch.tensor([BOS_ID], dtype=torch.long, device=DEVICE)

    pred_ids = []
    scores = []

    for _ in range(max_len):
        logits, hidden = model.decoder_step(prev_token, hidden)
        probs = torch.softmax(logits, dim=-1)
        score, token = torch.max(probs, dim=-1)

        token_id = int(token.item())
        score_value = float(score.item())

        if token_id == EOS_ID:
            break

        if token_id in (PAD_ID, BOS_ID):
            break

        pred_ids.append(token_id)
        scores.append(score_value)

        prev_token = token

    answer = decode_chunk_ids(pred_ids + [EOS_ID], output_chunks)

    return pred_ids, scores, answer


def print_chunk_stats(topic, output_chunks):
    real_chunks = [c for c in output_chunks if not c.get("special")]

    multi = [c for c in real_chunks if c["length"] >= 2]
    singles = [c for c in real_chunks if c["length"] == 1]

    print()
    print(f"================ {topic} OUTPUT CHUNK STATS ================")
    print(f"total output chunks: {len(output_chunks)}")
    print(f"real chunks:         {len(real_chunks)}")
    print(f"multi-token chunks:  {len(multi)}")
    print(f"single-token chunks: {len(singles)}")

    print()
    print("top reusable chunks:")
    for ch in sorted(multi, key=lambda x: x.get("count", 0), reverse=True)[:25]:
        print(f"  count={ch['count']:4d} len={ch['length']:2d}  {ch['text']}")

    print("====================================================")
    print()


def print_samples(topic, model, rows, input_vocab, output_chunks, count=5):
    print()
    print(f"================ {topic} SAMPLE PREDICTIONS ================")

    samples = random.sample(rows, min(count, len(rows)))

    for row in samples:
        pred_ids, scores, predicted_answer = predict_answer(
            model,
            row,
            input_vocab,
            output_chunks,
        )

        print()
        print("QUESTION:")
        print(row.get("question", ""))

        print()
        print("EXPECTED ANSWER:")
        print(row.get("answer", ""))

        print()
        print("PREDICTED ANSWER:")
        print(predicted_answer)

        print()
        print("PREDICTED CHUNKS:")
        for idx, score in zip(pred_ids, scores):
            text = (
                output_chunks[idx]["text"] if 0 <= idx < len(output_chunks) else "???"
            )
            print(f"  {score:.3f}  {idx:5d}  {text}")

        print()
        print("----------------------------------------------------")

    print("====================================================")
    print()


# ============================================================
# TRAIN SELECTOR
# ============================================================


def train_selector(selector_examples, out_dir: Path, topics):
    print()
    print("====================================================")
    print("TRAINING SELECTOR")
    print("====================================================")

    random.shuffle(selector_examples)

    selector_vocab = build_selector_vocab(selector_examples)

    topic_to_id = {topic: i for i, topic in enumerate(topics)}
    id_to_topic = {i: topic for topic, i in topic_to_id.items()}

    split_idx = int(len(selector_examples) * (1.0 - VAL_SPLIT))
    train_examples = selector_examples[:split_idx]
    val_examples = selector_examples[split_idx:]

    train_ds = SelectorDataset(train_examples, selector_vocab, topic_to_id)
    val_ds = SelectorDataset(val_examples, selector_vocab, topic_to_id)

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

    model = SelectorModel(
        input_size=len(selector_vocab),
        num_topics=len(topics),
        hidden_size=HIDDEN_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=SELECTOR_LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.03)

    best_val_loss = math.inf
    epochs_without_improvement = 0

    selector_pt_path = out_dir / "selector.pt"

    for epoch in range(1, SELECTOR_EPOCHS + 1):
        model.train()

        train_loss_total = 0.0
        batches = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_loss_total += float(loss.item())
            batches += 1

        train_loss = train_loss_total / max(batches, 1)
        metrics = evaluate_selector(model, val_loader, criterion)

        print(
            f"[selector] epoch {epoch:03d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {metrics['loss']:.4f} | "
            f"val_acc {metrics['acc']:.4f}"
        )

        if metrics["loss"] < best_val_loss - MIN_DELTA:
            best_val_loss = metrics["loss"]
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "input_size": len(selector_vocab),
                "num_topics": len(topics),
                "hidden_size": HIDDEN_SIZE,
                "dropout": DROPOUT,
                "best_val_loss": best_val_loss,
            }

            torch.save(checkpoint, selector_pt_path)
            print(f"[saved best selector] {selector_pt_path}")

        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                print(
                    f"[selector early stop] no val_loss improvement for {PATIENCE} epochs"
                )
                break

    save_json(out_dir / "selector_vocab.json", selector_vocab)
    save_json(out_dir / "selector_labels.json", id_to_topic)

    selector_config = {
        "model_type": "meatball_topic_selector_bow_ngrams",
        "input_type": "bag_of_words_ngrams",
        "input_ngrams": list(INPUT_NGRAMS),
        "max_input_vocab_size": MAX_INPUT_VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "dropout": DROPOUT,
        "model_pt_path": "selector.pt",
        "selector_vocab_path": "selector_vocab.json",
        "selector_labels_path": "selector_labels.json",
    }

    save_json(out_dir / "selector_config.json", selector_config)

    print("[saved] selector files")


# ============================================================
# TRAIN TOPIC SPECIALIST
# ============================================================


def train_topic_model(topic, rows, topic_dir: Path):
    print()
    print("====================================================")
    print(f"TRAINING TOPIC SPECIALIST: {topic}")
    print("====================================================")
    print(f"rows: {len(rows)}")

    topic_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("[1] mining reusable answer chunks...")
    output_chunks = build_output_chunks(rows)
    key_to_id, key_to_text, phrase_keys_by_len = build_chunk_lookup(output_chunks)

    print_chunk_stats(topic, output_chunks)

    print("[2] building topic input vocab...")
    input_vocab = build_input_vocab(rows)

    print(f"input vocab size:  {len(input_vocab)}")
    print(f"output vocab size: {len(output_chunks)}")

    random.shuffle(rows)

    split_idx = int(len(rows) * (1.0 - VAL_SPLIT))
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:]

    if len(val_rows) < 1:
        val_rows = train_rows[:]

    print(f"train rows: {len(train_rows)}")
    print(f"val rows:   {len(val_rows)}")

    train_ds = MeatballChunkDataset(train_rows, input_vocab, key_to_id)
    val_ds = MeatballChunkDataset(val_rows, input_vocab, key_to_id)

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

    print()
    print("[3] training chunk model...")

    model = ChunkAnswerModel(
        input_size=len(input_vocab),
        output_vocab_size=len(output_chunks),
        hidden_size=HIDDEN_SIZE,
        embed_size=EMBED_SIZE,
        dropout=DROPOUT,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TOPIC_LR,
        weight_decay=WEIGHT_DECAY,
    )

    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_ID,
        label_smoothing=LABEL_SMOOTHING,
    )

    best_val_loss = math.inf
    epochs_without_improvement = 0

    model_pt_path = topic_dir / "model.pt"

    for epoch in range(1, TOPIC_EPOCHS + 1):
        model.train()

        train_loss_total = 0.0
        batches = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(
                x,
                target=y,
                teacher_forcing=True,
                max_len=MAX_OUTPUT_CHUNKS + 1,
            )

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
            )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_loss_total += float(loss.item())
            batches += 1

        train_loss = train_loss_total / max(batches, 1)
        metrics = evaluate_chunk_model(model, val_loader, criterion)

        print(
            f"[{topic}] epoch {epoch:03d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {metrics['loss']:.4f} | "
            f"val_token_acc {metrics['token_acc']:.4f} | "
            f"val_exact_seq {metrics['exact_seq']:.4f}"
        )

        if metrics["loss"] < best_val_loss - MIN_DELTA:
            best_val_loss = metrics["loss"]
            epochs_without_improvement = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "input_vocab_size": len(input_vocab),
                "output_vocab_size": len(output_chunks),
                "hidden_size": HIDDEN_SIZE,
                "embed_size": EMBED_SIZE,
                "dropout": DROPOUT,
                "max_output_chunks": MAX_OUTPUT_CHUNKS,
                "best_val_loss": best_val_loss,
                "special_tokens": {
                    "PAD": PAD,
                    "BOS": BOS,
                    "EOS": EOS,
                    "UNK": UNK,
                    "PAD_ID": PAD_ID,
                    "BOS_ID": BOS_ID,
                    "EOS_ID": EOS_ID,
                    "UNK_ID": UNK_ID,
                },
            }

            torch.save(checkpoint, model_pt_path)
            print(f"[saved best {topic}] {model_pt_path}")

        else:
            epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                print(
                    f"[{topic} early stop] no val_loss improvement for {PATIENCE} epochs"
                )
                break

    print()
    print("[4] saving topic files...")

    checkpoint = torch.load(model_pt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    save_json(topic_dir / "input_vocab.json", input_vocab)
    save_json(topic_dir / "output_chunks.json", output_chunks)

    config = {
        "model_type": "question_to_answer_chunks_gru_topic_specialist",
        "topic": topic,
        "input_type": "bag_of_words_ngrams",
        "input_ngrams": list(INPUT_NGRAMS),
        "max_input_vocab_size": MAX_INPUT_VOCAB_SIZE,
        "output_type": "mined_answer_chunks",
        "min_chunk_freq": MIN_CHUNK_FREQ,
        "min_chunk_words": MIN_CHUNK_WORDS,
        "max_chunk_words": MAX_CHUNK_WORDS,
        "max_output_chunks": MAX_OUTPUT_CHUNKS,
        "max_output_vocab_size": MAX_OUTPUT_VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "embed_size": EMBED_SIZE,
        "dropout": DROPOUT,
        "pad_id": PAD_ID,
        "bos_id": BOS_ID,
        "eos_id": EOS_ID,
        "unk_id": UNK_ID,
        "model_pt_path": "model.pt",
        "input_vocab_path": "input_vocab.json",
        "output_chunks_path": "output_chunks.json",
        "note": "Topic-specialized Meatball chunk model using original mined chunk logic.",
    }

    save_json(topic_dir / "config.json", config)

    print(f"[saved] {topic_dir / 'input_vocab.json'}")
    print(f"[saved] {topic_dir / 'output_chunks.json'}")
    print(f"[saved] {topic_dir / 'config.json'}")
    print(f"[saved] {model_pt_path}")

    print_samples(topic, model, val_rows, input_vocab, output_chunks, count=5)


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default="assets/data/specialized_QA",
        help="Folder containing topic JSONL files, e.g. animals.jsonl, games.jsonl, art.jsonl",
    )

    parser.add_argument(
        "--out_dir",
        default="assets/models/specialized_meatball_chunks",
        help="Output folder for selector and topic models",
    )

    parser.add_argument(
        "--limit_per_topic",
        type=int,
        default=0,
        help="Optional row limit per topic for quick tests",
    )

    parser.add_argument(
        "--skip_selector",
        action="store_true",
        help="Skip selector training",
    )

    parser.add_argument(
        "--only_topic",
        default=None,
        help="Train only one topic, e.g. --only_topic art",
    )

    args = parser.parse_args()

    print(f"device: {DEVICE}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    topics_dir = out_dir / "topics"

    out_dir.mkdir(parents=True, exist_ok=True)
    topics_dir.mkdir(parents=True, exist_ok=True)

    topic_files = sorted(data_dir.glob("*.jsonl"))

    if not topic_files:
        raise RuntimeError(f"No .jsonl topic files found in {data_dir}")

    topic_rows = {}
    selector_examples = []

    print()
    print("Loading topic datasets...")

    for path in topic_files:
        topic = path.stem

        if args.only_topic and topic != args.only_topic:
            continue

        rows = load_jsonl(path)

        cleaned = []
        seen_questions = set()

        for row in rows:
            q = row["question"].strip()
            a = row["answer"].strip()

            if not q or not a:
                continue

            key = q.lower()

            if key in seen_questions:
                continue

            seen_questions.add(key)

            cleaned.append(
                {
                    "question": q,
                    "answer": a,
                    "history": row.get("history", []),
                }
            )

        if args.limit_per_topic and len(cleaned) > args.limit_per_topic:
            cleaned = random.sample(cleaned, args.limit_per_topic)

        if len(cleaned) < 10:
            print(f"[skip] {topic}: only {len(cleaned)} usable rows")
            continue

        topic_rows[topic] = cleaned

        for row in cleaned:
            selector_examples.append((row["question"], topic))

        print(f"{topic}: {len(cleaned)} rows")

    topics = sorted(topic_rows.keys())

    if not topics:
        raise RuntimeError("No usable topics loaded.")

    print()
    print(f"usable topics: {topics}")
    print(f"selector examples: {len(selector_examples)}")

    if not args.skip_selector and not args.only_topic:
        train_selector(selector_examples, out_dir, topics)
    elif args.only_topic:
        print("[selector skipped] only_topic mode")
    else:
        print("[selector skipped]")

    manifest = {
        "architecture": "MeatballAI selector plus topic-specialized original chunk models",
        "selector": {
            "model": "selector.pt",
            "config": "selector_config.json",
            "vocab": "selector_vocab.json",
            "labels": "selector_labels.json",
        },
        "topics": {},
        "notes": [
            "Each topic has its own input_vocab.json and output_chunks.json.",
            "Specialized models use original mined chunk logic.",
            "Do not use one shared chunk vocabulary across topics.",
        ],
    }

    for topic in topics:
        topic_dir = topics_dir / topic

        train_topic_model(topic, topic_rows[topic], topic_dir)

        manifest["topics"][topic] = {
            "dir": f"topics/{topic}",
            "model": f"topics/{topic}/model.pt",
            "config": f"topics/{topic}/config.json",
            "input_vocab": f"topics/{topic}/input_vocab.json",
            "output_chunks": f"topics/{topic}/output_chunks.json",
        }

        save_json(out_dir / "manifest.json", manifest)

    save_json(out_dir / "manifest.json", manifest)

    print()
    print("DONE")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
