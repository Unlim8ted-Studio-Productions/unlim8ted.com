import argparse
import ast
import json
import math
import operator
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

REACTION_DIR = Path("assets/models/meatball_reaction_model")
COMPLEXITY_DIR = Path("assets/models/complexity_classifier")
MATH_CLASSIFIER_DIR = Path("assets/models/math_classifier")

SUBJECT_FINDER_DIR = Path("assets/models/subject_finder")
SUBJECT_INSERTER_DIR = Path("assets/models/subject_inserter")
INPUT_CORRECTOR_DIR = Path("assets/models/input_text_corrector")
OUTPUT_SANITY_DIR = Path("assets/models/output_sanity_checker")

GENERATOR_DIR = Path("assets/models/general_cover_chunks_noisy_continue")
MATH_MODEL_PATH = Path(
    "assets/models/math_equation_translator/math_equation_translator_final.pt"
)

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

INPUT_NGRAMS = (1, 2, 3)

PROMPT_SIZE = 128
HIDDEN_SIZE = 192
EMBED_SIZE = 128
DROPOUT = 0.35
MAX_OUTPUT_CHUNKS = 24
INPUT_CORRECTOR_EMBED = 160
INPUT_CORRECTOR_HIDDEN = 320
OUTPUT_SANITY_HIDDEN = 256
OUTPUT_SANITY_DROPOUT = 0.22


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def print_step(name, value):
    print(f"\n[{name}]")
    if isinstance(value, (dict, list)):
        print(json.dumps(value, indent=2, ensure_ascii=False))
    else:
        print(value)


def normalize(text):
    text = str(text).lower().replace("\n", " ")
    text = re.sub(r"[^a-z0-9!?.,' +\-*/=()%$]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_no_punc(text):
    text = normalize(text)
    text = re.sub(r"[!?.,:;\"'`()\[\]{}]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def maybe_load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return load_json(path)


def restore_entity_casing(text):
    out = str(text or "")
    replacements = [
        ("the glitch", "The Glitch"),
        ("timecat", "TimeCat"),
        ("time cat", "TimeCat"),
        ("meatball ai", "Meatball AI"),
        ("unlim8ted", "Unlim8ted"),
    ]
    for old, new in replacements:
        out = re.sub(rf"\b{re.escape(old)}\b", new, out, flags=re.I)
    return out


def heuristic_input_correction(text):
    out = str(text or "").strip()
    if not out:
        return out

    fixes = [
        (r"\bteh\b", "the"),
        (r"\bwaht\b", "what"),
        (r"\bwich\b", "which"),
        (r"\bglich\b", "glitch"),
        (r"\bgltich\b", "glitch"),
        (r"\bglotch\b", "glitch"),
        (r"\bmeetball\b", "meatball"),
        (r"\bmeatbal\b", "meatball"),
        (r"\btime cat\b", "TimeCat"),
        (r"\bunlimited\b", "Unlim8ted"),
    ]

    for pattern, repl in fixes:
        out = re.sub(pattern, repl, out, flags=re.I)

    out = restore_entity_casing(out)
    out = re.sub(r"\s+", " ", out).strip()

    if re.match(r"^(what|who|where|when|why|how|does|do|did|is|are|can)\b", out, flags=re.I) and not re.search(r"[?.!]$", out):
        out += "?"

    return out


# ============================================================
# GENERIC NGRAM CLASSIFIER
# ============================================================


def classifier_features(text, char_ngrams=(2, 3, 4, 5), word_ngrams=(1, 2, 3)):
    text = normalize(text)
    no_punc = normalize_no_punc(text)
    feats = []

    s = f"<{text}>"
    for n in char_ngrams:
        for i in range(len(s) - n + 1):
            feats.append("c:" + s[i : i + n])

    words = text.split()
    for n in word_ngrams:
        for i in range(len(words) - n + 1):
            feats.append("w:" + "_".join(words[i : i + n]))

    if not no_punc:
        feats.append("flag:empty")
    if "?" in text:
        feats.append("flag:question")
    if "!" in text:
        feats.append("flag:bang")
    if re.search(r"\d", text):
        feats.append("flag:number")
    if re.search(r"[+\-*/=]", text):
        feats.append("flag:operator")
    if re.search(r"(.)\1\1", text):
        feats.append("flag:repeated_chars")

    if re.search(r"\b(facts|list|examples|features|types|projects)\b", no_punc):
        feats.append("flag:list_word")
    if re.search(
        r"\b(compare|contrast|vs|versus|difference|different|better)\b", no_punc
    ):
        feats.append("flag:compare_word")
    if re.search(r"\b(and|also|plus)\b", no_punc):
        feats.append("flag:connector")
    if re.search(r"\b(it|that|this|more|they|them|their)\b", no_punc):
        feats.append("flag:followup_pronoun")
    if no_punc in {"hi", "hello", "hey", "yo", "sup", "thanks", "thank you"}:
        feats.append("flag:smalltalk_exact")
    if no_punc in {
        "what does that mean",
        "what do you mean",
        "explain that",
        "what was that",
        "tell me more",
        "more",
    }:
        feats.append("flag:followup_exact")

    return feats


def vectorize_classifier(text, vocab, char_ngrams=(2, 3, 4, 5), word_ngrams=(1, 2, 3)):
    x = torch.zeros(1, len(vocab), dtype=torch.float32)
    counts = Counter(classifier_features(text, char_ngrams, word_ngrams))

    for feat, count in counts.items():
        idx = vocab.get(feat, 0)
        x[0, idx] = min(float(count), 5.0)

    return x


def output_sanity_pair_features(question, answer, char_ngrams=(2, 3, 4, 5), word_ngrams=(1, 2, 3)):
    q = normalize(question)
    a = normalize(answer)
    merged = f"question: {q} answer: {a}"
    wrapped = f"<{merged}>"
    feats = []

    for n in char_ngrams:
        for i in range(0, max(0, len(wrapped) - n + 1)):
            feats.append("c:" + wrapped[i : i + n])

    words = merged.split()
    for n in word_ngrams:
        for i in range(0, max(0, len(words) - n + 1)):
            feats.append("w:" + "_".join(words[i : i + n]))

    q_words = set(re.findall(r"[a-z0-9']+", q))
    a_words = set(re.findall(r"[a-z0-9']+", a))
    overlap = len(
        (q_words & a_words) - {"what", "is", "the", "a", "an", "tell", "me", "about"}
    )
    if overlap == 0:
        feats.append("flag:no_overlap")
    if overlap >= 2:
        feats.append("flag:good_overlap")
    if a.startswith("- "):
        feats.append("flag:list")
    if a in {"i'm not", "the meatball chooses to interpret that as"}:
        feats.append("flag:known_bad")
    if len(a.split()) <= 2:
        feats.append("flag:very_short")
    if a.count("?") > 2:
        feats.append("flag:many_questions")
    if re.fullmatch(r"(yes|yeah|yep|no|nope)\.?", a):
        feats.append("flag:bare_yes_no")
    if (
        a.startswith("i don't know")
        or a.startswith("im not")
        or a.startswith("i'm not")
    ):
        feats.append("flag:weak_fallback")
    return feats


def vectorize_output_sanity(question, answer, vocab, char_ngrams=(2, 3, 4, 5), word_ngrams=(1, 2, 3)):
    x = torch.zeros(1, len(vocab), dtype=torch.float32)
    counts = Counter(
        output_sanity_pair_features(question, answer, char_ngrams, word_ngrams)
    )

    for feat, count in counts.items():
        idx = vocab.get(feat, 0)
        x[0, idx] = min(float(count), 5.0)

    return x


class GenericClassifier(nn.Module):
    def __init__(self, input_size, classes, hidden=320, dropout=0.22):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, classes),
        )

    def forward(self, x):
        return self.net(x)


class InputCorrector(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, INPUT_CORRECTOR_EMBED, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, INPUT_CORRECTOR_EMBED, padding_idx=0)
        self.encoder = nn.GRU(INPUT_CORRECTOR_EMBED, INPUT_CORRECTOR_HIDDEN, batch_first=True)
        self.decoder = nn.GRU(INPUT_CORRECTOR_EMBED, INPUT_CORRECTOR_HIDDEN, batch_first=True)
        self.head = nn.Linear(INPUT_CORRECTOR_HIDDEN, tgt_vocab_size)

    def forward(self, src_ids, tgt_ids):
        src_emb = self.src_embed(src_ids)
        _, hidden = self.encoder(src_emb)
        decoder_input = tgt_ids[:, :-1]
        tgt_emb = self.tgt_embed(decoder_input)
        decoded, _ = self.decoder(tgt_emb, hidden)
        return self.head(decoded)


class OutputSanityClassifier(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=OUTPUT_SANITY_HIDDEN, dropout=OUTPUT_SANITY_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def resolve_labels(model_dir, config, fallback_labels=None):
    labels_path = model_dir / "labels.json"

    if labels_path.exists():
        return load_json(labels_path)

    if "labels" in config:
        print(f"[warn] {model_dir} has no labels.json; using config labels", flush=True)
        return config["labels"]

    if fallback_labels is not None:
        print(
            f"[warn] {model_dir} has no labels.json/config labels; using fallback {fallback_labels}",
            flush=True,
        )
        return fallback_labels

    raise FileNotFoundError(f"Missing labels.json for {model_dir}")


def load_classifier(
    model_dir,
    pt_name,
    default_hidden=320,
    default_dropout=0.22,
    fallback_labels=None,
):
    model_dir = Path(model_dir)

    vocab = load_json(model_dir / "input_vocab.json")

    config_path = model_dir / "config.json"
    config = load_json(config_path) if config_path.exists() else {}

    labels = resolve_labels(model_dir, config, fallback_labels=fallback_labels)

    hidden = int(config.get("hidden", default_hidden))
    dropout = float(config.get("dropout", default_dropout))

    ckpt = torch.load(model_dir / pt_name, map_location=DEVICE)

    model = GenericClassifier(
        input_size=len(vocab),
        classes=len(labels),
        hidden=hidden,
        dropout=dropout,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return {
        "model": model,
        "vocab": vocab,
        "labels": labels,
        "config": config,
        "dir": str(model_dir),
    }


def load_output_sanity(model_dir):
    model_dir = Path(model_dir)
    pt_path = model_dir / "output_sanity_checker.pt"
    vocab_path = model_dir / "input_vocab.json"
    labels_path = model_dir / "labels.json"

    if not (pt_path.exists() and vocab_path.exists() and labels_path.exists()):
        return None

    vocab = load_json(vocab_path)
    labels = load_json(labels_path)
    config = maybe_load_json(model_dir / "config.json") or {}
    ckpt = torch.load(pt_path, map_location=DEVICE)
    state_dict = ckpt["model_state_dict"]
    first_weight = state_dict.get("net.0.weight")
    hidden_size = (
        int(first_weight.shape[0])
        if first_weight is not None
        else int(config.get("hidden", OUTPUT_SANITY_HIDDEN))
    )
    dropout = float(config.get("dropout", OUTPUT_SANITY_DROPOUT))

    model = OutputSanityClassifier(
        len(vocab), len(labels), hidden_size=hidden_size, dropout=dropout
    ).to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return {
        "model": model,
        "vocab": vocab,
        "labels": labels,
        "config": config,
        "dir": str(model_dir),
    }


def load_input_corrector(model_dir):
    model_dir = Path(model_dir)
    pt_path = model_dir / "input_text_corrector.pt"
    src_vocab_path = model_dir / "input_vocab.json"
    tgt_vocab_path = model_dir / "output_vocab.json"

    if not (pt_path.exists() and src_vocab_path.exists() and tgt_vocab_path.exists()):
        return None

    src_vocab = load_json(src_vocab_path)
    tgt_vocab = load_json(tgt_vocab_path)
    config = maybe_load_json(model_dir / "config.json") or {}
    ckpt = torch.load(pt_path, map_location=DEVICE)

    model = InputCorrector(len(src_vocab), len(tgt_vocab)).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    id_to = {int(idx): token for token, idx in tgt_vocab.items()}

    return {
        "model": model,
        "src_vocab": src_vocab,
        "tgt_vocab": tgt_vocab,
        "id_to": id_to,
        "config": config,
        "dir": str(model_dir),
    }


@torch.no_grad()
def predict_classifier(text, runtime):
    config = runtime.get("config", {})
    char_ngrams = tuple(config.get("char_ngrams", [2, 3, 4, 5]))
    word_ngrams = tuple(config.get("word_ngrams", [1, 2, 3]))

    x = vectorize_classifier(text, runtime["vocab"], char_ngrams, word_ngrams).to(
        DEVICE
    )
    logits = runtime["model"](x)

    probs = torch.softmax(logits, dim=-1)[0]
    idx = int(torch.argmax(probs).item())

    return {
        "label": runtime["labels"][idx],
        "confidence": float(probs[idx].item()),
        "probs": {
            runtime["labels"][i]: float(probs[i].item())
            for i in range(len(runtime["labels"]))
        },
    }


def encode_chars(text, vocab, max_len, add_bos=False, add_eos=True):
    ids = []
    if add_bos:
        ids.append(vocab.get("<bos>", BOS_ID))
    for ch in str(text or "")[: max(1, max_len - 2)]:
        ids.append(vocab.get(ch, vocab.get("<unk>", UNK_ID)))
    if add_eos:
        ids.append(vocab.get("<eos>", EOS_ID))
    return ids[:max_len]


@torch.no_grad()
def run_input_corrector(text, runtime):
    heuristic = heuristic_input_correction(text)
    if runtime is None:
        return {
            "text": heuristic,
            "changed": heuristic.strip() != str(text or "").strip(),
            "source": "heuristic",
        }

    max_len = int(runtime.get("config", {}).get("max_len", 96))
    src_ids = encode_chars(heuristic, runtime["src_vocab"], max_len, add_bos=True, add_eos=True)
    src = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)
    model = runtime["model"]

    src_emb = model.src_embed(src)
    _, hidden = model.encoder(src_emb)
    next_token = torch.tensor([[runtime["tgt_vocab"].get("<bos>", BOS_ID)]], dtype=torch.long, device=DEVICE)
    out_tokens = []

    for _ in range(max_len):
        emb = model.tgt_embed(next_token[:, -1:])
        decoded, hidden = model.decoder(emb, hidden)
        logits = model.head(decoded[:, -1, :])
        pred = int(torch.argmax(logits, dim=-1).item())
        if pred == runtime["tgt_vocab"].get("<eos>", EOS_ID):
            break
        out_tokens.append(runtime["id_to"].get(pred, ""))
        next_token = torch.cat(
            [next_token, torch.tensor([[pred]], dtype=torch.long, device=DEVICE)],
            dim=1,
        )

    corrected = "".join(out_tokens).strip() or heuristic
    corrected = restore_entity_casing(re.sub(r"\s+", " ", corrected).strip())
    return {
        "text": corrected,
        "changed": corrected.strip() != str(text or "").strip(),
        "source": "model",
    }


@torch.no_grad()
def run_output_sanity_check(question, answer, runtime):
    text = str(answer or "").strip()
    prompt = str(question or "").strip()
    normalized = normalize_no_punc(text)
    fallback = "I'm not quite sure what you mean. It might just be the sauce getting tangled up in itself."

    if not text:
        return {
            "label": "confused_fallback",
            "confidence": 1.0,
            "answer": fallback,
            "used_fallback": True,
            "source": "heuristic",
        }

    if normalized == "im not":
        return {
            "label": "confused_fallback",
            "confidence": 1.0,
            "answer": fallback,
            "used_fallback": True,
            "source": "heuristic",
        }

    if runtime is None:
        return {
            "label": "accept",
            "confidence": 0.0,
            "answer": text,
            "used_fallback": False,
            "source": "disabled",
        }

    config = runtime.get("config", {})
    char_ngrams = tuple(config.get("char_ngrams", [2, 3, 4, 5]))
    word_ngrams = tuple(config.get("word_ngrams", [1, 2, 3]))

    if config.get("feature_mode") == "qa_pair":
        x = vectorize_output_sanity(
            prompt, text, runtime["vocab"], char_ngrams, word_ngrams
        ).to(DEVICE)
        logits = runtime["model"](x)
        probs = torch.softmax(logits, dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        pred = {
            "label": runtime["labels"][idx],
            "confidence": float(probs[idx].item()),
            "probs": {
                runtime["labels"][i]: float(probs[i].item())
                for i in range(len(runtime["labels"]))
            },
        }
    else:
        pred = predict_classifier(text, runtime)

    if pred["label"] == "confused_fallback" and pred["confidence"] >= 0.72:
        return {
            **pred,
            "answer": fallback,
            "used_fallback": True,
            "source": "model",
        }

    return {
        **pred,
        "answer": text,
        "used_fallback": False,
        "source": "model",
    }


# ============================================================
# SUBJECT FINDER + SUBJECT INSERTER
# ============================================================


def predict_subject(text, subject_runtime, min_conf=0.45):
    pred = predict_classifier(text, subject_runtime)
    label = pred["label"]

    if pred["confidence"] < min_conf:
        label = "NONE"

    return {
        "subject": label,
        "confidence": pred["confidence"],
        "raw": pred,
    }


def predict_subject_insert_action(
    text, subject, complexity, memory, inserter_runtime, min_conf=0.45
):
    # The actual model gets current text plus context encoded as plain text.
    model_input = (
        f"question: {text} "
        f"subject: {subject} "
        f"complexity: {complexity} "
        f"previous_subject: {memory.subjects[-1] if memory.subjects else 'NONE'} "
        f"last_answer: {memory.last_answer}"
    )

    pred = predict_classifier(model_input, inserter_runtime)
    action = pred["label"]

    if pred["confidence"] < min_conf:
        action = "keep"

    return {
        "action": action,
        "confidence": pred["confidence"],
        "raw": pred,
        "model_input": model_input,
    }


def apply_subject_insertion(text, action, subject, memory):
    if action not in {"insert", "replace_pronoun", "use_previous_subject"}:
        return text

    if subject == "NONE":
        subject = memory.subjects[-1] if memory.subjects else "NONE"

    if subject == "NONE":
        return text

    out = text

    replacements = [
        (r"\bit\b", subject),
        (r"\bits\b", f"{subject}'s"),
        (r"\bthey\b", subject),
        (r"\bthem\b", subject),
        (r"\btheir\b", f"{subject}'s"),
        (r"\bthat\b", subject),
        (r"\bthis\b", subject),
    ]

    for pattern, repl in replacements:
        out = re.sub(pattern, repl, out, flags=re.I)

    if out == text:
        out = f"{text} about {subject}"

    return out


# ============================================================
# GENERATOR
# ============================================================


def input_normalize(text):
    text = str(text).lower().replace("\n", " ")
    text = re.sub(r"[^a-z0-9_!?.,' -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def input_tokenize(text):
    text = input_normalize(text)
    return text.split() if text else []


def make_input_ngrams(tokens, ngrams=INPUT_NGRAMS):
    feats = []
    for n in ngrams:
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            feats.append("_".join(tokens[i : i + n]))
    return feats


def vectorize_generator_question(question, input_vocab):
    text = f"question: {question}"
    feats = make_input_ngrams(input_tokenize(text))

    x = torch.zeros(1, len(input_vocab), dtype=torch.float32)
    counts = Counter(feats)
    unk = input_vocab.get("<UNK>", 1)

    for feat, count in counts.items():
        idx = input_vocab.get(feat, unk)
        x[0, idx] = min(float(count), 5.0)

    return x


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

    return re.sub(r"\s+", " ", out).strip()


def decode_generator_ids(ids, output_chunks):
    texts = []

    for idx in ids:
        idx = int(idx)

        if idx == EOS_ID:
            break

        if idx in (PAD_ID, BOS_ID, UNK_ID):
            continue

        if 0 <= idx < len(output_chunks):
            texts.append(output_chunks[idx]["text"])

    return join_chunk_texts(texts)


class ChunkAnswerModel(nn.Module):
    def __init__(self, input_size, output_vocab_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, PROMPT_SIZE),
            nn.LayerNorm(PROMPT_SIZE),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(PROMPT_SIZE, PROMPT_SIZE),
            nn.LayerNorm(PROMPT_SIZE),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        self.embedding = nn.Embedding(output_vocab_size, EMBED_SIZE)
        self.decoder_cell = nn.GRUCell(PROMPT_SIZE + EMBED_SIZE, HIDDEN_SIZE)

        self.output = nn.Sequential(
            nn.LayerNorm(PROMPT_SIZE + HIDDEN_SIZE),
            nn.Dropout(DROPOUT),
            nn.Linear(PROMPT_SIZE + HIDDEN_SIZE, output_vocab_size),
        )

    def forward(self, x, max_len=MAX_OUTPUT_CHUNKS + 1):
        batch_size = x.size(0)
        prompt_context = self.encoder(x)

        write_hidden = torch.zeros(batch_size, HIDDEN_SIZE, device=x.device)
        prev_token = torch.full(
            (batch_size,), BOS_ID, dtype=torch.long, device=x.device
        )

        logits_steps = []

        for _ in range(max_len):
            emb = self.embedding(prev_token)
            write_hidden = self.decoder_cell(
                torch.cat([emb, prompt_context], dim=-1), write_hidden
            )
            logits = self.output(torch.cat([prompt_context, write_hidden], dim=-1))
            logits_steps.append(logits.unsqueeze(1))
            prev_token = torch.argmax(logits, dim=-1)

        return torch.cat(logits_steps, dim=1)


def load_generator(model_dir):
    model_dir = Path(model_dir)

    input_vocab = load_json(model_dir / "input_vocab.json")
    output_chunks = load_json(model_dir / "output_chunks.json")
    ckpt = torch.load(model_dir / "model.pt", map_location=DEVICE)

    model = ChunkAnswerModel(len(input_vocab), len(output_chunks)).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return {
        "model": model,
        "input_vocab": input_vocab,
        "output_chunks": output_chunks,
        "dir": str(model_dir),
    }


@torch.no_grad()
def generate_answer(question, generator):
    x = vectorize_generator_question(question, generator["input_vocab"]).to(DEVICE)
    logits = generator["model"](x)
    ids = torch.argmax(logits[0], dim=-1).detach().cpu().tolist()
    return decode_generator_ids(ids, generator["output_chunks"])


# ============================================================
# MATH MODEL
# ============================================================


def math_replace_symbols(text):
    text = str(text)
    for old, new in {
        "×": "*",
        "÷": "/",
        "−": "-",
        "²": " ** 2 ",
        "³": " ** 3 ",
        "√": " sqrt ",
    }.items():
        text = text.replace(old, new)
    return text


def post_process_answer(text):
    text = str(text)

    # normalize spaces
    text = re.sub(r"\s+", " ", text).strip()

    # fix spaces before punctuation
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)

    # fix spaces after opening punctuation
    text = re.sub(r"([(\[{])\s+", r"\1", text)

    # fix spaces before closing punctuation
    text = re.sub(r"\s+([)\]}])", r"\1", text)

    # fix missing spaces after punctuation
    text = re.sub(r"([,.;:!?])([A-Za-z0-9])", r"\1 \2", text)

    # fix common quote spacing from chunk joins
    text = re.sub(r"\s+'\s*", "'", text)
    text = re.sub(r"\s+’\s*", "’", text)

    # capitalize first alphabetical character
    m = re.search(r"[A-Za-z]", text)
    if m:
        i = m.start()
        text = text[:i] + text[i].upper() + text[i + 1 :]

    # capitalize after sentence endings
    def cap_after_sentence(match):
        return match.group(1) + match.group(2).upper()

    text = re.sub(r"([.!?]\s+)([a-z])", cap_after_sentence, text)

    # fix standalone lowercase i
    text = re.sub(r"\bi\b", "I", text)

    # common formatting cleanup
    text = text.replace(" i’m ", " I’m ")
    text = text.replace(" i'm ", " I'm ")

    return text.strip()


def apply_answer_overrides(answer, reaction, animation, sanity):
    next_answer = str(answer or "").strip()
    next_reaction = reaction
    next_animation = animation

    if sanity.get("used_fallback"):
        next_answer = sanity["answer"]
        next_reaction = "confused"
        next_animation = "confused"

    if next_answer == "The Meatball chooses to interpret that as":
        next_answer = "The Meatball chooses to interpret that as completely true."

    if next_answer in {"I'm not", "I’m not"}:
        next_answer = "I'm not quite sure what you mean. It might just be the sauce getting tangled up in itself."
        next_reaction = "confused"
        next_animation = "confused"

    if next_answer == "Thank you. The sauce accepts the compliment.":
        next_reaction = "excited"
        next_animation = "excited"

    return {
        "answer": next_answer,
        "reaction": next_reaction,
        "animation": next_animation,
        "sanity": sanity,
    }


def math_normalize_question(q):
    q = math_replace_symbols(q).lower().replace("\n", " ")

    reps = [
        (r"\btimes\b", " * "),
        (r"\bplus\b", " + "),
        (r"\bminus\b", " - "),
        (r"\bdivided\s+by\b", " / "),
        (r"\bsquared\b", " ^ 2 "),
        (r"\bcubed\b", " ^ 3 "),
    ]

    for pattern, repl in reps:
        q = re.sub(pattern, repl, q)

    q = re.sub(r"[^a-z0-9_+\-*/^().,?:;$%=\s']", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def math_tokenize(text):
    text = math_replace_symbols(str(text).lower())
    return re.findall(r"\d+\.\d+|\d+|\*\*|sqrt|pi|[a-z_]+|[+\-*/^=().,?:;$%]", text)


def math_detok(tokens):
    out = ""

    for t in tokens:
        if t in {".", ",", "?", "!", ":", ";", "%", ")"}:
            out = out.rstrip() + t
        elif t == "(":
            if out and not out.endswith(" "):
                out += " "
            out += t
        elif t in {"+", "-", "*", "/", "**", "^", "="}:
            out += f" {t} "
        else:
            if out and not out.endswith((" ", "(")):
                out += " "
            out += t

    return re.sub(r"\s+", " ", out).strip()


def math_encode(tokens, vocab, max_len):
    ids = [BOS_ID]

    for t in tokens:
        ids.append(vocab.get(t, UNK_ID))
        if len(ids) >= max_len - 1:
            break

    ids.append(EOS_ID)
    ids = ids[:max_len]

    while len(ids) < max_len:
        ids.append(PAD_ID)

    return ids


def math_decode(ids, id_to_token):
    toks = []

    for idx in ids:
        idx = int(idx)

        if idx == EOS_ID:
            break

        if idx in {PAD_ID, BOS_ID, UNK_ID}:
            continue

        toks.append(id_to_token.get(idx, ""))

    return math_detok(toks)


class MathSeq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed, hidden, dropout):
        super().__init__()

        self.input_emb = nn.Embedding(input_vocab_size, embed, padding_idx=PAD_ID)
        self.output_emb = nn.Embedding(output_vocab_size, embed, padding_idx=PAD_ID)

        self.encoder = nn.GRU(embed, hidden, batch_first=True, bidirectional=True)
        self.bridge = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh())
        self.decoder = nn.GRU(embed, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, output_vocab_size)

    def encode_context(self, x):
        emb = self.dropout(self.input_emb(x))
        _, h = self.encoder(emb)
        return self.bridge(torch.cat([h[-2], h[-1]], dim=-1)).unsqueeze(0)

    def forward(self, x, max_len):
        batch = x.size(0)
        h = self.encode_context(x)

        prev = torch.full((batch, 1), BOS_ID, dtype=torch.long, device=x.device)
        ids = []

        for _ in range(max_len):
            emb = self.dropout(self.output_emb(prev))
            dec_out, h = self.decoder(emb, h)
            logits = self.out(dec_out[:, -1])
            prev = torch.argmax(logits, dim=-1, keepdim=True)
            ids.append(prev)

        return torch.cat(ids, dim=1)


def load_math_model(path):
    path = Path(path)

    if not path.exists():
        return None

    ckpt = torch.load(path, map_location=DEVICE)
    cfg = ckpt["config"]

    input_vocab = ckpt["input_vocab"]
    output_vocab = ckpt["output_vocab"]
    id_to = {int(v): k for k, v in output_vocab.items()}

    model = MathSeq2Seq(
        len(input_vocab),
        len(output_vocab),
        cfg["embed"],
        cfg["hidden"],
        cfg["dropout"],
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return {
        "model": model,
        "input_vocab": input_vocab,
        "id_to": id_to,
        "config": cfg,
        "path": str(path),
    }


def extract_equation_and_answer(text):
    eq = ""
    ans = ""

    m = re.search(r"equation\s*:\s*(.*?)(?:\s+answer\s*:|$)", text, flags=re.I)
    if m:
        eq = m.group(1).strip()

    m = re.search(r"answer\s*:\s*(.*)$", text, flags=re.I)
    if m:
        ans = m.group(1).strip()

    return eq, ans


SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_eval_expr(expr):
    expr = expr.strip().replace("^", "**")

    if not expr or not re.fullmatch(r"[0-9+\-*/().\s*]+", expr):
        raise ValueError("unsafe expression")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            return SAFE_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return SAFE_OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsafe expression")

    val = ev(ast.parse(expr, mode="eval"))

    if isinstance(val, float) and abs(val - round(val)) < 1e-9:
        val = int(round(val))

    return val


@torch.no_grad()
def answer_math(question, runtime):
    q = math_normalize_question(question)
    cfg = runtime["config"]

    x = torch.tensor(
        [math_encode(math_tokenize(q), runtime["input_vocab"], cfg["max_input_len"])],
        dtype=torch.long,
        device=DEVICE,
    )

    ids = runtime["model"](x, cfg["max_output_len"])[0].detach().cpu().tolist()
    decoded = math_decode(ids, runtime["id_to"])
    eq, pred_ans = extract_equation_and_answer(decoded)

    computed = ""
    if eq:
        try:
            computed = str(safe_eval_expr(eq))
        except Exception:
            computed = ""

    return {
        "normalized": q,
        "decoded": decoded,
        "equation": eq,
        "predicted_answer": pred_ans,
        "computed_answer": computed,
        "final": computed or pred_ans or decoded,
    }


# ============================================================
# MEMORY + ROUTING
# ============================================================


class RuntimeMemory:
    def __init__(self):
        self.history = []
        self.subjects = []
        self.last_answer = ""
        self.previous_reaction = "neutral"
        self.angry_streak = 0
        self.sauce_attack_cooldown = 0

    def update(self, user_text, answer, reaction, subject=None, preserve_angry_state=False):
        self.history.append(user_text)
        self.history.append(answer)
        self.history = self.history[-8:]

        self.last_answer = answer

        if subject and subject != "NONE":
            self.subjects.append(subject)
            self.subjects = self.subjects[-5:]

        if preserve_angry_state:
            self.previous_reaction = reaction
            return

        if reaction == "angry":
            self.angry_streak += 1
        else:
            self.angry_streak = 0

        self.previous_reaction = reaction


def split_multi(text):
    parts = re.split(r"\s+\band\s+|\s+\balso\s+|\s+\bplus\s+|;", text, flags=re.I)
    parts = [p.strip(" ?.!,") for p in parts if p.strip(" ?.!,")]
    return parts if len(parts) > 1 else [text]


def format_list(answer):
    dashed = [part.strip().lstrip("-").strip() for part in re.split(r"\s+-\s+", str(answer or "").strip()) if part.strip()]
    if len(dashed) > 1:
        return "\n".join(f"- {part}" for part in dashed[:8])

    parts = re.split(r"(?<=[.!?])\s+", answer)
    parts = [p.strip() for p in parts if p.strip()]

    if len(parts) <= 1:
        return f"- {parts[0]}" if parts else ""

    return "\n".join(f"- {p}" for p in parts[:8])


def normalize_subject_text(text):
    out = normalize_no_punc(text)
    out = re.sub(r"^(between|of)\s+", "", out, flags=re.I)
    return restore_entity_casing(out).strip()


def parse_compare_subjects(text):
    clean = str(text or "").strip().rstrip("?")
    if not clean:
        return []

    patterns = [
        r"\bcompare\s+(.+?)\s+(?:and|vs|versus)\s+(.+)$",
        r"\bcomparison\s+of\s+(.+?)\s+(?:and|vs|versus)\s+(.+)$",
        r"\bdifference\s+between\s+(.+?)\s+(?:and|vs|versus)\s+(.+)$",
        r"^(.+?)\s+(?:vs|versus)\s+(.+)$",
        r"^(.+?)\s+and\s+(.+)$",
    ]

    for pattern in patterns:
        match = re.search(pattern, clean, flags=re.I)
        if not match:
            continue
        subjects = [
            normalize_subject_text(match.group(1)),
            normalize_subject_text(match.group(2)),
        ]
        subjects = [item for item in subjects if item]
        if len(subjects) == 2:
            return subjects

    return []


def strip_route_prefix(text):
    out = str(text or "").strip()
    out = re.sub(r"^(please\s+)?(list|facts|examples|features)\s+(out\s+)?", "", out, flags=re.I)
    out = re.sub(r"^(give me|tell me)\s+(a\s+)?(list of\s+|facts about\s+|examples of\s+|features of\s+)", "", out, flags=re.I)
    return out.strip()


def build_generator_question_for_route(raw_input, rewritten_question, complexity, subject, memory):
    base = str(rewritten_question or raw_input or "").strip()
    normalized = normalize_no_punc(raw_input)
    previous_subject = memory.subjects[-1] if memory.subjects else "NONE"

    if complexity == "list":
        stripped = strip_route_prefix(base)
        if re.search(r"\babout\b", stripped, flags=re.I) or re.search(r"\bof\b", stripped, flags=re.I):
            return stripped
        if stripped:
            return f"facts about {stripped}"
        return base

    if complexity in {"followup", "normal_qa"} and subject == "NONE" and previous_subject != "NONE":
        if re.search(r"\bit\b", raw_input, flags=re.I):
            return apply_subject_insertion(raw_input, "replace_pronoun", previous_subject, memory)
        if re.search(r"\bthis\b", raw_input, flags=re.I):
            return re.sub(r"\bthis\b", previous_subject, raw_input, flags=re.I)
        if re.search(r"\bthat\b", raw_input, flags=re.I):
            return re.sub(r"\bthat\b", previous_subject, raw_input, flags=re.I)
        if re.search(r"\bthey\b|\bthem\b|\btheir\b", raw_input, flags=re.I):
            return apply_subject_insertion(raw_input, "replace_pronoun", previous_subject, memory)

    if complexity == "normal_qa" and previous_subject != "NONE" and normalized in {
        "what is it",
        "who is it",
        "what is this",
        "what is that",
    }:
        return apply_subject_insertion(raw_input, "replace_pronoun", previous_subject, memory)

    return base


def answer_once(text, runtimes, memory, debug=True):
    reaction_rt = runtimes["reaction"]
    complexity_rt = runtimes["complexity"]
    math_classifier_rt = runtimes["math_classifier"]
    subject_finder_rt = runtimes["subject_finder"]
    subject_inserter_rt = runtimes["subject_inserter"]
    generator = runtimes["generator"]
    math_model = runtimes["math_model"]
    input_corrector_rt = runtimes.get("input_corrector")
    output_sanity_rt = runtimes.get("output_sanity")

    if debug:
        print_step("RAW_INPUT", text)
        print_step(
            "MEMORY_BEFORE",
            {
                "history": memory.history,
                "subjects": memory.subjects,
                "last_answer": memory.last_answer,
                "previous_reaction": memory.previous_reaction,
                "angry_streak": memory.angry_streak,
                "sauce_attack_cooldown": memory.sauce_attack_cooldown,
            },
        )

    if memory.sauce_attack_cooldown > 0:
        memory.sauce_attack_cooldown -= 1

    correction = run_input_corrector(text, input_corrector_rt)
    corrected = correction["text"]
    normalized = normalize_no_punc(corrected)

    if debug:
        print_step("INPUT_CORRECTOR", correction)

    reaction_pred = predict_classifier(corrected, reaction_rt)
    complexity_pred = predict_classifier(corrected, complexity_rt)
    math_pred = predict_classifier(corrected, math_classifier_rt)

    reaction = reaction_pred["label"]
    complexity = complexity_pred["label"]
    is_math = math_pred["label"] == "math" and math_pred["confidence"] >= 0.55

    subject_pred = predict_subject(corrected, subject_finder_rt)
    subject = subject_pred["subject"]

    inserter_pred = predict_subject_insert_action(
        text=corrected,
        subject=subject,
        complexity=complexity,
        memory=memory,
        inserter_runtime=subject_inserter_rt,
    )

    rewritten = apply_subject_insertion(
        text=corrected,
        action=inserter_pred["action"],
        subject=subject,
        memory=memory,
    )
    generator_question = build_generator_question_for_route(
        corrected,
        rewritten,
        complexity,
        subject,
        memory,
    )
    animation = reaction
    animation_path = "none"

    if debug:
        print_step("REACTION_MODEL", reaction_pred)
        print_step("COMPLEXITY_CLASSIFIER", complexity_pred)
        print_step("MATH_CLASSIFIER", math_pred)
        print_step("SUBJECT_FINDER_MODEL", subject_pred)
        print_step("SUBJECT_INSERTER_MODEL", inserter_pred)
        print_step(
            "REACTION_STATE",
            {
                "final_reaction": reaction,
                "animation": animation,
                "animation_path": animation_path,
            },
        )
        print_step("REWRITTEN_INPUT", rewritten)
        print_step("GENERATOR_QUESTION", generator_question)

    route = ""
    final = ""

    if normalized in {"ok", "okay"}:
        route = "smalltalk"
        final = "Yep."
        animation = "neutral"

    elif normalized == "yep":
        route = "smalltalk"
        final = "Yes."
        animation = "neutral"

    elif reaction == "angry" and memory.angry_streak >= 1 and memory.sauce_attack_cooldown <= 0:
        route = "anger_escalation_attack"
        final = "YOU DONT LIKE ME??? THEN FACE THE SAUCE."
        animation = "angry"
        animation_path = "sad_to_sauce_attack_cutscene"
        memory.sauce_attack_cooldown = 15
        memory.angry_streak = 0
        memory.previous_reaction = "angry"

    elif is_math:
        route = "math"

        if math_model is None:
            final = "The math brain is not loaded."
        else:
            math_out = answer_math(rewritten, math_model)
            if debug:
                print_step("MATH_OUTPUT", math_out)
            final = math_out["final"]

    elif complexity == "smalltalk":
        route = "smalltalk"
        final = generate_answer(generator_question, generator)

    elif complexity == "unknown":
        route = "unknown"
        final = "The sauce blinked twice. I need a clearer question."

    elif complexity == "followup" and normalized in {
        "what does that mean",
        "what did that mean",
        "explain that",
        "what do you mean",
        "what was that",
    }:
        route = "explain_previous"

        if memory.last_answer:
            final = f"It means: {memory.last_answer}"
        else:
            final = "I do not have a previous answer to explain yet."

    elif complexity == "compare":
        route = "compare"
        compare_subjects = parse_compare_subjects(corrected)
        if len(compare_subjects) == 2:
            left = generate_answer(f"what is {compare_subjects[0]}", generator)
            right = generate_answer(f"what is {compare_subjects[1]}", generator)
            final = " ".join(part for part in [left, right] if part).strip()
        if not final:
            final = "Comparing two things at once might make this tiny meatball brain explode."

    elif complexity == "multi_part":
        route = "multi_part"
        parts = split_multi(rewritten)
        answers = []

        for part in parts:
            ans = generate_answer(part, generator)
            answers.append(ans)

            if debug:
                print_step(
                    "GENERATOR_CALL",
                    {
                        "subquestion": part,
                        "answer": ans,
                    },
                )

        final = " ".join(a for a in answers if a).strip()

    elif complexity == "list":
        route = "list"
        raw = generate_answer(generator_question, generator)
        final = format_list(raw)

        if debug:
            print_step(
                "GENERATOR_CALL",
                {
                    "question": rewritten,
                    "generator_question": generator_question,
                    "answer": raw,
                },
            )

    else:
        route = "normal_qa"
        final = generate_answer(generator_question, generator)

        if debug:
            print_step(
                "GENERATOR_CALL",
                {
                    "question": rewritten,
                    "generator_question": generator_question,
                    "answer": final,
                },
            )

    final = post_process_answer(final)
    sanity = run_output_sanity_check(generator_question, final, output_sanity_rt)
    overridden = apply_answer_overrides(final, reaction, animation, sanity)

    packet = {
        "answer": overridden["answer"],
        "route": route,
        "reaction": overridden["reaction"],
        "animation": overridden["animation"],
        "animation_path": animation_path,
        "complexity": complexity,
        "math": is_math,
        "subject": subject,
        "subject_action": inserter_pred["action"],
        "corrected_input": corrected,
        "rewritten_input": rewritten,
        "generator_question": generator_question,
        "sanity": sanity,
    }

    if debug:
        print_step("FINAL_PACKET", packet)

    memory.update(
        text,
        overridden["answer"],
        overridden["reaction"],
        subject,
        preserve_angry_state=(route == "anger_escalation_attack"),
    )

    if debug:
        print_step(
            "MEMORY_AFTER",
            {
                "history": memory.history,
                "subjects": memory.subjects,
                "last_answer": memory.last_answer,
                "previous_reaction": memory.previous_reaction,
                "angry_streak": memory.angry_streak,
                "sauce_attack_cooldown": memory.sauce_attack_cooldown,
            },
        )

    return packet


def load_all(args):
    reaction = load_classifier(
        args.reaction_dir,
        "reaction_model.pt",
        default_hidden=256,
        default_dropout=0.2,
    )

    complexity = load_classifier(
        args.complexity_dir,
        "complexity_classifier.pt",
        default_hidden=320,
        default_dropout=0.22,
    )

    math_classifier = load_classifier(
        args.math_classifier_dir,
        "math_classifier.pt",
        default_hidden=384,
        default_dropout=0.25,
        fallback_labels=["general", "math"],
    )

    subject_finder = load_classifier(
        args.subject_finder_dir,
        args.subject_finder_pt,
        default_hidden=384,
        default_dropout=0.25,
    )

    subject_inserter = load_classifier(
        args.subject_inserter_dir,
        args.subject_inserter_pt,
        default_hidden=384,
        default_dropout=0.25,
    )

    input_corrector = load_input_corrector(args.input_corrector_dir)
    output_sanity = load_output_sanity(args.output_sanity_dir)
    generator = load_generator(args.generator_dir)
    math_model = load_math_model(args.math_model)

    return {
        "reaction": reaction,
        "complexity": complexity,
        "math_classifier": math_classifier,
        "subject_finder": subject_finder,
        "subject_inserter": subject_inserter,
        "input_corrector": input_corrector,
        "output_sanity": output_sanity,
        "generator": generator,
        "math_model": math_model,
    }


def smoke_tests(runtimes):
    memory = RuntimeMemory()

    tests = [
        "hi",
        "what is the glich",
        "what is The Glitch",
        "facts about dogs",
        "what does that mean",
        "what is 1+1",
        "cats vs dogs",
        "ok",
        "yep",
        "I HATE YOU",
        "YOUR STUPII",
        "be angry",
        "be angry again",
        "sauce attack",
    ]

    for q in tests:
        print("\n" + "=" * 90)
        print("TEST:", q)
        print("=" * 90)

        out = answer_once(q, runtimes, memory, debug=True)

        print("\nANSWER:")
        print(out["answer"])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--reaction_dir", default=str(REACTION_DIR))
    parser.add_argument("--complexity_dir", default=str(COMPLEXITY_DIR))
    parser.add_argument("--math_classifier_dir", default=str(MATH_CLASSIFIER_DIR))

    parser.add_argument("--subject_finder_dir", default=str(SUBJECT_FINDER_DIR))
    parser.add_argument("--subject_finder_pt", default="subject_finder.pt")

    parser.add_argument("--subject_inserter_dir", default=str(SUBJECT_INSERTER_DIR))
    parser.add_argument("--subject_inserter_pt", default="subject_inserter.pt")

    parser.add_argument("--input_corrector_dir", default=str(INPUT_CORRECTOR_DIR))
    parser.add_argument("--output_sanity_dir", default=str(OUTPUT_SANITY_DIR))

    parser.add_argument("--generator_dir", default=str(GENERATOR_DIR))
    parser.add_argument("--math_model", default=str(MATH_MODEL_PATH))

    parser.add_argument("--question", default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no_debug", action="store_true")

    args = parser.parse_args()

    print("Loading Meatball runtime v3...")
    print("Device:", DEVICE)

    runtimes = load_all(args)

    print("Loaded reaction:", args.reaction_dir)
    print("Loaded complexity:", args.complexity_dir)
    print("Loaded math classifier:", args.math_classifier_dir)
    print("Loaded subject finder:", args.subject_finder_dir)
    print("Loaded subject inserter:", args.subject_inserter_dir)
    print("Loaded input corrector:", bool(runtimes["input_corrector"]))
    print("Loaded output sanity:", bool(runtimes["output_sanity"]))
    print("Loaded generator:", args.generator_dir)
    print("Loaded math model:", bool(runtimes["math_model"]))

    if args.smoke:
        smoke_tests(runtimes)
        return

    memory = RuntimeMemory()

    if args.question:
        out = answer_once(args.question, runtimes, memory, debug=not args.no_debug)
        print("\nANSWER:")
        print(out["answer"])
        return

    print("\nInteractive mode. Type quit / exit / stop.")

    while True:
        q = input("\nYou: ").strip()

        if q.lower() in {"quit", "exit", "stop"}:
            break

        if not q:
            continue

        out = answer_once(q, runtimes, memory, debug=not args.no_debug)

        print("\nANSWER:")
        print(out["answer"])


if __name__ == "__main__":
    main()
