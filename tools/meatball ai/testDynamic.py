import json
import re
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn

# ============================================================
# PATHS
# ============================================================

ANSWER_MODEL_DIR = Path("assets/models/meatball_chunk_answer_model")

ANSWER_MODEL_PT_PATH = ANSWER_MODEL_DIR / "meatball_chunk_answer_model.pt"
ANSWER_INPUT_VOCAB_PATH = ANSWER_MODEL_DIR / "input_vocab.json"
ANSWER_OUTPUT_CHUNKS_PATH = ANSWER_MODEL_DIR / "output_chunks.json"
ANSWER_CONFIG_PATH = ANSWER_MODEL_DIR / "config.json"

SUBJECT_FINDER_DIR = Path("assets/models/subject_finder")

SUBJECT_FINDER_PT_PATH = SUBJECT_FINDER_DIR / "subject_finder.pt"
SUBJECT_FINDER_VOCAB_PATH = SUBJECT_FINDER_DIR / "vocab.json"
SUBJECT_FINDER_CONFIG_PATH = SUBJECT_FINDER_DIR / "config.json"

SUBJECT_INSERTER_DIR = Path("assets/models/subject_inserter")

SUBJECT_INSERTER_PT_PATH = SUBJECT_INSERTER_DIR / "subject_inserter.pt"
SUBJECT_INSERTER_VOCAB_PATH = SUBJECT_INSERTER_DIR / "vocab.json"
SUBJECT_INSERTER_LABELS_PATH = SUBJECT_INSERTER_DIR / "labels.json"
SUBJECT_INSERTER_CONFIG_PATH = SUBJECT_INSERTER_DIR / "config.json"

DATASET_PATH = Path("assets/data/SmartMeatballQA.jsonl")


# ============================================================
# SPECIAL TOKENS
# ============================================================

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# SETTINGS
# ============================================================

USE_SUBJECT_FINDER = True
USE_SUBJECT_INSERTER = True

MAX_CONTEXT_HISTORY_ITEMS = 6

DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_K = 1
DEFAULT_MIN_SCORE_STOP = None

SUBJECT_THRESHOLD = 0.50


# ============================================================
# JSON UTILS
# ============================================================


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []

    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    return rows


# ============================================================
# SHARED TEXT UTILS
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


def make_ngrams(tokens, ngrams):
    feats = []

    for n in ngrams:
        if len(tokens) < n:
            continue

        for i in range(len(tokens) - n + 1):
            feats.append("_".join(tokens[i : i + n]))

    return feats


def vectorize_text(text, vocab, ngrams):
    tokens = tokenize(text)
    feats = make_ngrams(tokens, ngrams)

    x = torch.zeros(len(vocab), dtype=torch.float32)

    counts = Counter(feats)
    unk_id = vocab.get("<UNK>", 1)

    for feat, count in counts.items():
        idx = vocab.get(feat, unk_id)
        x[idx] = min(float(count), 5.0)

    return x


# ============================================================
# SUBJECT FINDER TEXT UTILS
# ============================================================


def subject_history_to_text(history):
    if not history:
        return ""

    return " ".join(str(x) for x in history[-MAX_CONTEXT_HISTORY_ITEMS:])


def subject_finder_input_text(message, history):
    history_text = subject_history_to_text(history)
    return f"message: {message} history: {history_text}"


def subject_tokenize(text):
    return tokenize(text)


# ============================================================
# SUBJECT INSERTER TEXT UTILS
# ============================================================


def inserter_featurize_text(message, subject):
    text = f"message: {message} subject: {subject}"
    tokens = tokenize(text)
    feats = []

    for n in [1, 2, 3]:
        if len(tokens) < n:
            continue

        for i in range(len(tokens) - n + 1):
            feats.append("_".join(tokens[i : i + n]))

    return feats


def vectorize_inserter_input(message, subject, vocab):
    x = torch.zeros(len(vocab), dtype=torch.float32)

    for feat in inserter_featurize_text(message, subject):
        idx = vocab.get(feat)
        if idx is not None:
            x[idx] = 1.0

    return x


# ============================================================
# ANSWER MODEL INPUT
# ============================================================


def answer_row_to_input_text(question):
    return f"question: {question} context:"


def vectorize_answer_input(question, vocab, ngrams):
    text = answer_row_to_input_text(question)
    return vectorize_text(text, vocab, ngrams)


# ============================================================
# SUBJECT FINDER MODEL
# ============================================================


class SubjectFinderNet(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, max_len, dropout):
        super().__init__()

        self.max_len = max_len

        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.encoder = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.has_subject_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

        self.start_head = nn.Linear(hidden_size, 1)
        self.end_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        emb = self.embedding(input_ids)
        h = self.encoder(emb)

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        has_subject = self.has_subject_head(pooled).squeeze(-1)

        start_logits = self.start_head(h).squeeze(-1)
        end_logits = self.end_head(h).squeeze(-1)

        start_logits = start_logits.masked_fill(attention_mask == 0, -1e9)
        end_logits = end_logits.masked_fill(attention_mask == 0, -1e9)

        return {
            "has_subject": has_subject,
            "start": start_logits,
            "end": end_logits,
        }


def encode_subject_finder_input(text, vocab, max_len):
    tokens = subject_tokenize(text)

    ids = []

    for tok in tokens[:max_len]:
        ids.append(vocab.get(tok, vocab.get("<UNK>", 1)))

    attention = [1] * len(ids)

    while len(ids) < max_len:
        ids.append(vocab.get("<PAD>", 0))
        attention.append(0)

    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(attention, dtype=torch.long),
        tokens[:max_len],
    )


def load_subject_finder():
    if not SUBJECT_FINDER_PT_PATH.exists():
        print(f"[warn] missing subject finder model: {SUBJECT_FINDER_PT_PATH}")
        return None

    if not SUBJECT_FINDER_VOCAB_PATH.exists():
        print(f"[warn] missing subject finder vocab: {SUBJECT_FINDER_VOCAB_PATH}")
        return None

    if not SUBJECT_FINDER_CONFIG_PATH.exists():
        print(f"[warn] missing subject finder config: {SUBJECT_FINDER_CONFIG_PATH}")
        return None

    vocab = load_json(SUBJECT_FINDER_VOCAB_PATH)
    config = load_json(SUBJECT_FINDER_CONFIG_PATH)

    checkpoint = torch.load(SUBJECT_FINDER_PT_PATH, map_location=DEVICE)

    max_len = int(config.get("max_len", config.get("MAX_LEN", 96)))
    embed_size = int(config.get("embed_size", config.get("EMBED_SIZE", 96)))
    hidden_size = int(config.get("hidden_size", config.get("HIDDEN", 192)))
    dropout = float(config.get("dropout", config.get("DROPOUT", 0.2)))

    model = SubjectFinderNet(
        vocab_size=len(vocab),
        embed_size=embed_size,
        hidden_size=hidden_size,
        max_len=max_len,
        dropout=dropout,
    ).to(DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("Bad subject finder checkpoint format.")

    model.eval()

    return {
        "model": model,
        "vocab": vocab,
        "config": config,
        "max_len": max_len,
    }


@torch.no_grad()
def predict_subject(subject_pack, message, history):
    if subject_pack is None:
        return {
            "subject": "",
            "has_subject": False,
            "score": 0.0,
            "start": None,
            "end": None,
        }

    model = subject_pack["model"]
    vocab = subject_pack["vocab"]
    max_len = subject_pack["max_len"]

    text = subject_finder_input_text(message, history)

    input_ids, attention_mask, tokens = encode_subject_finder_input(
        text=text,
        vocab=vocab,
        max_len=max_len,
    )

    input_ids = input_ids.unsqueeze(0).to(DEVICE)
    attention_mask = attention_mask.unsqueeze(0).to(DEVICE)

    out = model(input_ids, attention_mask)

    has_prob = float(torch.sigmoid(out["has_subject"])[0].item())

    start_id = int(torch.argmax(out["start"], dim=-1)[0].item())
    end_id = int(torch.argmax(out["end"], dim=-1)[0].item())

    if end_id < start_id:
        start_id, end_id = end_id, start_id

    if has_prob < SUBJECT_THRESHOLD:
        return {
            "subject": "",
            "has_subject": False,
            "score": has_prob,
            "start": start_id,
            "end": end_id,
        }

    if not tokens or start_id >= len(tokens) or end_id >= len(tokens):
        return {
            "subject": "",
            "has_subject": False,
            "score": has_prob,
            "start": start_id,
            "end": end_id,
        }

    subject_tokens = tokens[start_id : end_id + 1]
    subject = " ".join(subject_tokens).strip()

    subject = re.sub(r"^(message|history|user|bot)\s*:?\s*", "", subject).strip()

    return {
        "subject": subject,
        "has_subject": bool(subject),
        "score": has_prob,
        "start": start_id,
        "end": end_id,
    }


# ============================================================
# SUBJECT INSERTER MODEL
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


def load_subject_inserter():
    if not SUBJECT_INSERTER_PT_PATH.exists():
        print(f"[warn] missing subject inserter model: {SUBJECT_INSERTER_PT_PATH}")
        return None

    if not SUBJECT_INSERTER_VOCAB_PATH.exists():
        print(f"[warn] missing subject inserter vocab: {SUBJECT_INSERTER_VOCAB_PATH}")
        return None

    if not SUBJECT_INSERTER_LABELS_PATH.exists():
        print(f"[warn] missing subject inserter labels: {SUBJECT_INSERTER_LABELS_PATH}")
        return None

    if not SUBJECT_INSERTER_CONFIG_PATH.exists():
        print(f"[warn] missing subject inserter config: {SUBJECT_INSERTER_CONFIG_PATH}")
        return None

    vocab = load_json(SUBJECT_INSERTER_VOCAB_PATH)
    labels = load_json(SUBJECT_INSERTER_LABELS_PATH)
    config = load_json(SUBJECT_INSERTER_CONFIG_PATH)

    checkpoint = torch.load(SUBJECT_INSERTER_PT_PATH, map_location=DEVICE)

    hidden = int(config.get("hidden", config.get("hidden_size", 256)))
    dropout = float(config.get("dropout", 0.2))

    model = SubjectInserterNet(
        input_dim=len(vocab),
        hidden=hidden,
        num_labels=len(labels),
        dropout=dropout,
    ).to(DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise ValueError("Bad subject inserter checkpoint format.")

    model.eval()

    return {
        "model": model,
        "vocab": vocab,
        "labels": labels,
        "config": config,
    }


@torch.no_grad()
def predict_inserter_op(inserter_pack, message, subject):
    if inserter_pack is None:
        return {
            "op": "already_standalone",
            "score": 0.0,
        }

    model = inserter_pack["model"]
    vocab = inserter_pack["vocab"]
    labels = inserter_pack["labels"]

    x = (
        vectorize_inserter_input(
            message=message,
            subject=subject,
            vocab=vocab,
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    logits = model(x)
    probs = torch.softmax(logits, dim=-1)

    score, pred = torch.max(probs, dim=-1)

    op = labels[int(pred.item())]

    return {
        "op": op,
        "score": float(score.item()),
    }


def apply_inserter_op(message, subject, op):
    m = normalize_text(message)
    s = normalize_text(subject)

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
        m2 = re.sub(r"\bdoes\s+this\b", f"does {s}", m2)
        m2 = re.sub(r"\bdoes\s+that\b", f"does {s}", m2)
        return m2

    if op == "is_subject":
        m2 = re.sub(r"\bis\s+it\b", f"is {s}", m)
        m2 = re.sub(r"\bare\s+they\b", f"are {s}", m2)
        m2 = re.sub(r"\bis\s+this\b", f"is {s}", m2)
        m2 = re.sub(r"\bis\s+that\b", f"is {s}", m2)
        return m2

    if op == "can_subject":
        return re.sub(
            r"\bcan\s+it\b|\bcan\s+they\b|\bcan\s+this\b|\bcan\s+that\b",
            f"can {s}",
            m,
        )

    if op == "replace_it_with_subject":
        return re.sub(r"\bit\b", s, m)

    if op == "replace_they_with_subject":
        return re.sub(r"\bthey\b|\bthem\b|\btheir\b", s, m)

    if op == "replace_this_with_subject":
        return re.sub(r"\bthis\b", s, m)

    if op == "replace_that_with_subject":
        return re.sub(r"\bthat\b", s, m)

    if op == "replace_he_she_with_subject":
        return re.sub(r"\bhe\b|\bshe\b|\bhim\b|\bher\b|\bhis\b", s, m)

    if op == "append_about_subject":
        return f"{m} about {s}"

    if op == "append_for_subject":
        return f"{m} for {s}"

    return m


def rewrite_question(subject_pack, inserter_pack, message, history):
    subject_result = predict_subject(
        subject_pack=subject_pack,
        message=message,
        history=history,
    )

    subject = subject_result["subject"]

    if not subject:
        return {
            "standalone_question": normalize_text(message),
            "subject": "",
            "subject_score": subject_result["score"],
            "op": "already_standalone",
            "op_score": 0.0,
        }

    op_result = predict_inserter_op(
        inserter_pack=inserter_pack,
        message=message,
        subject=subject,
    )

    op = op_result["op"]

    standalone = apply_inserter_op(
        message=message,
        subject=subject,
        op=op,
    )

    return {
        "standalone_question": standalone,
        "subject": subject,
        "subject_score": subject_result["score"],
        "op": op,
        "op_score": op_result["score"],
    }


# ============================================================
# ANSWER MODEL
# ============================================================


class ChunkAnswerModel(nn.Module):
    def __init__(
        self,
        input_size,
        output_vocab_size,
        hidden_size,
        embed_size,
        dropout,
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


def load_answer_model():
    if not ANSWER_MODEL_PT_PATH.exists():
        raise FileNotFoundError(f"Missing answer model: {ANSWER_MODEL_PT_PATH}")

    if not ANSWER_INPUT_VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"Missing answer input vocab: {ANSWER_INPUT_VOCAB_PATH}"
        )

    if not ANSWER_OUTPUT_CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Missing answer output chunks: {ANSWER_OUTPUT_CHUNKS_PATH}"
        )

    if not ANSWER_CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing answer config: {ANSWER_CONFIG_PATH}")

    input_vocab = load_json(ANSWER_INPUT_VOCAB_PATH)
    output_chunks = load_json(ANSWER_OUTPUT_CHUNKS_PATH)
    config = load_json(ANSWER_CONFIG_PATH)

    checkpoint = torch.load(ANSWER_MODEL_PT_PATH, map_location=DEVICE)

    hidden_size = checkpoint.get("hidden_size", config.get("hidden_size", 192))
    embed_size = checkpoint.get("embed_size", config.get("embed_size", 128))
    dropout = checkpoint.get("dropout", config.get("dropout", 0.35))

    model = ChunkAnswerModel(
        input_size=len(input_vocab),
        output_vocab_size=len(output_chunks),
        hidden_size=hidden_size,
        embed_size=embed_size,
        dropout=dropout,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    input_ngrams = tuple(config.get("input_ngrams", [1, 2, 3]))
    max_output_chunks = int(config.get("max_output_chunks", 40))

    return {
        "model": model,
        "input_vocab": input_vocab,
        "output_chunks": output_chunks,
        "input_ngrams": input_ngrams,
        "max_output_chunks": max_output_chunks,
    }


# ============================================================
# ANSWER TEXT CLEANUP
# ============================================================

KNOWN_CAPS = {
    "unlim8ted": "Unlim8ted",
    "unlim8ted studios": "Unlim8ted Studios",
    "unlim8ted studio productions": "Unlim8ted Studio Productions",
    "timecat": "TimeCat",
    "meatball": "Meatball",
    "smarter meatball": "Smarter Meatball",
    "the glitch": "The Glitch",
    "the life of a meatball": "The Life of a Meatball",
    "ai": "AI",
    "html": "HTML",
    "css": "CSS",
    "javascript": "JavaScript",
    "js": "JS",
    "json": "JSON",
    "jsonl": "JSONL",
    "onnx": "ONNX",
    "gru": "GRU",
    "llm": "LLM",
    "browser ai": "browser AI",
}


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

    return out.strip()


def capitalize_sentence_starts(text):
    if not text:
        return text

    chars = list(text)
    should_cap = True

    for i, ch in enumerate(chars):
        if should_cap and ch.isalpha():
            chars[i] = ch.upper()
            should_cap = False

        if ch in ".!?":
            should_cap = True

    return "".join(chars)


def apply_known_caps(text):
    for low, proper in sorted(
        KNOWN_CAPS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        pattern = re.compile(r"\b" + re.escape(low) + r"\b", flags=re.IGNORECASE)
        text = pattern.sub(proper, text)

    return text


def remove_repeated_sentence_fragments(text):
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())

    seen = set()
    kept = []

    for piece in pieces:
        clean = piece.strip()

        if not clean:
            continue

        key = clean.lower().strip(".!?")

        if key in seen:
            continue

        seen.add(key)
        kept.append(clean)

    return " ".join(kept)


def remove_repeated_halves(text):
    clean = text.strip()

    m = re.match(r"^(.*?)\s+and\s+\1[.!?]?$", clean, flags=re.IGNORECASE)

    if m:
        return m.group(1).strip() + "."

    words = clean.split()
    n = len(words)

    if n >= 8 and n % 2 == 0:
        first = " ".join(words[: n // 2]).strip(" .!?")
        second = " ".join(words[n // 2 :]).strip(" .!?")

        if first.lower() == second.lower():
            return first + "."

    return clean


def postprocess_answer(text):
    text = str(text)

    text = text.replace("<UNK>", "")
    text = text.replace("<PAD>", "")
    text = text.replace("<BOS>", "")
    text = text.replace("<EOS>", "")

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    text = re.sub(r"^[,.;:\s]+", "", text).strip()

    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])([A-Za-z0-9])", r"\1 \2", text)

    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r",{2,}", ",", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)

    text = re.sub(r"\b(is|are|was|were)\s*\?\s+", r"\1 ", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+’", "’", text)
    text = re.sub(r"’\s+", "’", text)
    text = re.sub(r"\s+'", "'", text)
    text = re.sub(r"'\s+", "'", text)

    text = re.sub(r"\bi\b", "I", text)

    text = re.sub(r"\bI\s*'m\b", "I’m", text, flags=re.IGNORECASE)
    text = re.sub(r"\bI\s*’m\b", "I’m", text, flags=re.IGNORECASE)
    text = re.sub(r"\bI\s*'ll\b", "I’ll", text, flags=re.IGNORECASE)
    text = re.sub(r"\bI\s*’ll\b", "I’ll", text, flags=re.IGNORECASE)
    text = re.sub(r"\bI\s*'ve\b", "I’ve", text, flags=re.IGNORECASE)
    text = re.sub(r"\bI\s*’ve\b", "I’ve", text, flags=re.IGNORECASE)
    text = re.sub(r"\bI\s*'d\b", "I’d", text, flags=re.IGNORECASE)
    text = re.sub(r"\bI\s*’d\b", "I’d", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdon\s*'t\b", "don’t", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdon\s*’t\b", "don’t", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcan\s*'t\b", "can’t", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcan\s*’t\b", "can’t", text, flags=re.IGNORECASE)
    text = re.sub(r"\bit\s*'s\b", "it’s", text, flags=re.IGNORECASE)
    text = re.sub(r"\bit\s*’s\b", "it’s", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthat\s*'s\b", "that’s", text, flags=re.IGNORECASE)
    text = re.sub(r"\bthat\s*’s\b", "that’s", text, flags=re.IGNORECASE)
    text = re.sub(r"\byou\s*'re\b", "you’re", text, flags=re.IGNORECASE)
    text = re.sub(r"\byou\s*’re\b", "you’re", text, flags=re.IGNORECASE)

    text = remove_repeated_sentence_fragments(text)
    text = remove_repeated_halves(text)

    text = capitalize_sentence_starts(text)
    text = apply_known_caps(text)

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"([,.;:!?])([A-Za-z0-9])", r"\1 \2", text)

    if text and text[-1] not in ".!?":
        text += "."

    return text


def decode_chunk_ids(ids, output_chunks):
    texts = []

    for idx in ids:
        idx = int(idx)

        if idx == EOS_ID:
            break

        if idx in (PAD_ID, BOS_ID, UNK_ID):
            continue

        if 0 <= idx < len(output_chunks):
            item = output_chunks[idx]

            if isinstance(item, dict):
                texts.append(item["text"])
            else:
                texts.append(str(item))

    raw = join_chunk_texts(texts)
    return postprocess_answer(raw)


# ============================================================
# ANSWER PREDICTION
# ============================================================


def mask_bad_tokens(logits):
    logits[:, PAD_ID] = -1e9
    logits[:, BOS_ID] = -1e9
    logits[:, UNK_ID] = -1e9
    return logits


@torch.no_grad()
def predict_answer(
    answer_pack,
    question,
    temperature=1.0,
    top_k=1,
    min_score_stop=None,
):
    model = answer_pack["model"]
    input_vocab = answer_pack["input_vocab"]
    output_chunks = answer_pack["output_chunks"]
    input_ngrams = answer_pack["input_ngrams"]
    max_output_chunks = answer_pack["max_output_chunks"]

    model.eval()

    x = (
        vectorize_answer_input(
            question=question,
            vocab=input_vocab,
            ngrams=input_ngrams,
        )
        .unsqueeze(0)
        .to(DEVICE)
    )

    hidden = model.encode(x)

    prev_token = torch.tensor([BOS_ID], dtype=torch.long, device=DEVICE)

    pred_ids = []
    pred_scores = []

    for _ in range(max_output_chunks + 1):
        logits, hidden = model.decoder_step(prev_token, hidden)

        logits = logits / max(float(temperature), 1e-6)
        logits = mask_bad_tokens(logits)

        probs = torch.softmax(logits, dim=-1)

        if top_k <= 1:
            score, token = torch.max(probs, dim=-1)
        else:
            values, indices = torch.topk(
                probs,
                k=min(top_k, probs.size(-1)),
                dim=-1,
            )

            sampled_index = torch.multinomial(values[0], num_samples=1).item()
            token = indices[:, sampled_index]
            score = values[:, sampled_index]

        token_id = int(token.item())
        score_value = float(score.item())

        if token_id == EOS_ID:
            break

        if token_id in (PAD_ID, BOS_ID, UNK_ID):
            break

        if (
            min_score_stop is not None
            and score_value < min_score_stop
            and len(pred_ids) > 0
        ):
            break

        pred_ids.append(token_id)
        pred_scores.append(score_value)

        prev_token = token

    answer = decode_chunk_ids(pred_ids + [EOS_ID], output_chunks)

    if not answer:
        answer = "I don’t know enough to answer that clearly."

    return {
        "answer": answer,
        "ids": pred_ids,
        "scores": pred_scores,
    }


# ============================================================
# MATCH DEBUG
# ============================================================


def find_matching_rows(query, rows, limit=8):
    q = normalize_text(query)

    exact = []
    partial = []

    for row in rows:
        question = str(row.get("question", ""))
        question_norm = normalize_text(question)

        if q == question_norm:
            exact.append(row)
        elif q in question_norm or question_norm in q:
            partial.append(row)
        elif any(tok and tok in question_norm for tok in q.split()):
            partial.append(row)

    return (exact + partial)[:limit]


# ============================================================
# DISPLAY
# ============================================================


def print_prediction(
    result,
    answer_pack,
    rewrite_info,
    original_question,
    show_chunks=True,
    show_context=True,
):
    output_chunks = answer_pack["output_chunks"]

    print()
    print("ANSWER:")
    print(result["answer"])

    if show_context:
        print()
        print("REWRITE:")
        print(
            json.dumps(
                {
                    "original_question": original_question,
                    "subject": rewrite_info.get("subject", ""),
                    "subject_score": rewrite_info.get("subject_score", 0.0),
                    "op": rewrite_info.get("op", ""),
                    "op_score": rewrite_info.get("op_score", 0.0),
                    "standalone_question": rewrite_info.get(
                        "standalone_question", original_question
                    ),
                },
                indent=2,
            )
        )

    if show_chunks:
        print()
        print("CHUNKS:")

        for idx, score in zip(result["ids"], result["scores"]):
            if 0 <= idx < len(output_chunks):
                item = output_chunks[idx]
                text = item["text"] if isinstance(item, dict) else str(item)
            else:
                text = "???"

            print(f"  {score:.3f}  {idx:5d}  {text}")

    print()


def print_help():
    print()
    print("Commands:")
    print("  /quit")
    print("  /chunks off")
    print("  /chunks on")
    print("  /context off")
    print("  /context on")
    print("  /finder off")
    print("  /finder on")
    print("  /inserter off")
    print("  /inserter on")
    print("  /history")
    print("  /reset")
    print("  /temp 0.8")
    print("  /topk 3")
    print("  /stop 0.05")
    print("  /stop off")
    print("  /match question text")
    print("  /rewrite question text")
    print("  /help")
    print()


# ============================================================
# MAIN LOOP
# ============================================================


def interactive_loop():
    answer_pack = load_answer_model()
    subject_pack = load_subject_finder()
    inserter_pack = load_subject_inserter()

    dataset_rows = load_jsonl(DATASET_PATH)

    use_subject_finder = USE_SUBJECT_FINDER and subject_pack is not None
    use_subject_inserter = USE_SUBJECT_INSERTER and inserter_pack is not None

    show_chunks = True
    show_context = True

    temperature = DEFAULT_TEMPERATURE
    top_k = DEFAULT_TOP_K
    min_score_stop = DEFAULT_MIN_SCORE_STOP

    dialogue_history = []

    print()
    print("Loaded Meatball dynamic run.")
    print(f"Device: {DEVICE}")
    print()
    print("Answer Model:")
    print(f"  Input vocab:       {len(answer_pack['input_vocab'])}")
    print(f"  Output chunks:     {len(answer_pack['output_chunks'])}")
    print(f"  Max output chunks: {answer_pack['max_output_chunks']}")
    print()
    print("Subject Finder:")
    if subject_pack is None:
        print("  Not loaded.")
    else:
        print("  Loaded:            yes")
        print(f"  Vocab:             {len(subject_pack['vocab'])}")
        print(f"  Max len:           {subject_pack['max_len']}")
    print()
    print("Subject Inserter:")
    if inserter_pack is None:
        print("  Not loaded.")
    else:
        print("  Loaded:            yes")
        print(f"  Vocab:             {len(inserter_pack['vocab'])}")
        print(f"  Ops:               {len(inserter_pack['labels'])}")
    print()

    print_help()

    while True:
        try:
            user_text = input("You: ").strip()
        except KeyboardInterrupt:
            print()
            break

        if not user_text:
            continue

        lower = user_text.lower()

        if lower in ["/quit", "quit", "exit"]:
            break

        if lower == "/help":
            print_help()
            continue

        if lower == "/chunks off":
            show_chunks = False
            print("Chunk display off.")
            continue

        if lower == "/chunks on":
            show_chunks = True
            print("Chunk display on.")
            continue

        if lower == "/context off":
            show_context = False
            print("Rewrite/context display off.")
            continue

        if lower == "/context on":
            show_context = True
            print("Rewrite/context display on.")
            continue

        if lower == "/finder off":
            use_subject_finder = False
            print("Subject finder disabled.")
            continue

        if lower == "/finder on":
            if subject_pack is None:
                print("Subject finder is not loaded.")
            else:
                use_subject_finder = True
                print("Subject finder enabled.")
            continue

        if lower == "/inserter off":
            use_subject_inserter = False
            print("Subject inserter disabled.")
            continue

        if lower == "/inserter on":
            if inserter_pack is None:
                print("Subject inserter is not loaded.")
            else:
                use_subject_inserter = True
                print("Subject inserter enabled.")
            continue

        if lower == "/history":
            print()
            print("SUBJECT FINDER HISTORY:")
            if not dialogue_history:
                print("  <empty>")
            else:
                for item in dialogue_history[-MAX_CONTEXT_HISTORY_ITEMS:]:
                    print(f"  {item}")
            print()
            continue

        if lower == "/reset":
            dialogue_history = []
            print("History reset.")
            continue

        if lower.startswith("/temp "):
            try:
                temperature = float(user_text.split(" ", 1)[1])
                print(f"Temperature set to {temperature}")
            except ValueError:
                print("Bad temperature.")
            continue

        if lower.startswith("/topk "):
            try:
                top_k = int(user_text.split(" ", 1)[1])
                top_k = max(1, top_k)
                print(f"top_k set to {top_k}")
            except ValueError:
                print("Bad top_k.")
            continue

        if lower.startswith("/stop "):
            value = user_text.split(" ", 1)[1].strip().lower()

            if value == "off":
                min_score_stop = None
                print("Low-score stop disabled.")
                continue

            try:
                min_score_stop = float(value)
                print(f"Low-score stop set to {min_score_stop}")
            except ValueError:
                print("Bad stop value.")
            continue

        if lower.startswith("/match "):
            query = user_text.split(" ", 1)[1]
            matches = find_matching_rows(query, dataset_rows, limit=8)

            if not matches:
                print("No matching dataset rows found.")
                continue

            print()
            print("MATCHES:")

            for i, row in enumerate(matches, start=1):
                print()
                print(f"[{i}] Q: {row.get('question', '')}")
                print(f"    A: {row.get('answer', '')}")

            print()
            continue

        if lower.startswith("/rewrite "):
            query = user_text.split(" ", 1)[1]

            if use_subject_finder:
                temp_subject_pack = subject_pack
            else:
                temp_subject_pack = None

            if use_subject_inserter:
                temp_inserter_pack = inserter_pack
            else:
                temp_inserter_pack = None

            rewrite_info = rewrite_question(
                subject_pack=temp_subject_pack,
                inserter_pack=temp_inserter_pack,
                message=query,
                history=dialogue_history,
            )

            print()
            print("REWRITE ONLY:")
            print(json.dumps(rewrite_info, indent=2))
            print()
            continue

        if use_subject_finder:
            active_subject_pack = subject_pack
        else:
            active_subject_pack = None

        if use_subject_inserter:
            active_inserter_pack = inserter_pack
        else:
            active_inserter_pack = None

        rewrite_info = rewrite_question(
            subject_pack=active_subject_pack,
            inserter_pack=active_inserter_pack,
            message=user_text,
            history=dialogue_history,
        )

        standalone_question = rewrite_info["standalone_question"]

        result = predict_answer(
            answer_pack=answer_pack,
            question=standalone_question,
            temperature=temperature,
            top_k=top_k,
            min_score_stop=min_score_stop,
        )

        print_prediction(
            result=result,
            answer_pack=answer_pack,
            rewrite_info=rewrite_info,
            original_question=user_text,
            show_chunks=show_chunks,
            show_context=show_context,
        )

        # History is ONLY for Subject Finder.
        # The Answer Model only receives the rewritten standalone question.
        dialogue_history.append(f"User: {user_text}")
        dialogue_history.append(f"Bot: {result['answer']}")

        if len(dialogue_history) > MAX_CONTEXT_HISTORY_ITEMS:
            dialogue_history = dialogue_history[-MAX_CONTEXT_HISTORY_ITEMS:]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    interactive_loop()
