import argparse
import csv
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[2]
SPECIALIZED_DIR = ROOT / "assets" / "data" / "specialized_QA"
SMART_QA_PATH = ROOT / "tools" / "SmartMeatballQA.jsonl"
WIZARD_PATH = ROOT / "assets" / "data" / "subject_finder_wizard_of_wikipedia.jsonl"
PRODUCTS_PATH = ROOT / "assets" / "data" / "products.json"
TYPO_DATA_CANDIDATES = [
    ROOT / "assets" / "data" / "github_typo_corpus.jsonl",
    ROOT / "assets" / "data" / "github_typo_corpus.tsv",
    ROOT / "assets" / "data" / "github_typo_corpus.csv",
    ROOT / "assets" / "data" / "spelling_pairs.jsonl",
    ROOT / "assets" / "data" / "spelling_pairs.tsv",
]

INPUT_OUT_DIR = ROOT / "assets" / "models" / "input_text_corrector"
ALIGN_OUT_DIR = ROOT / "assets" / "models" / "output_sanity_checker"

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

LABELS = ["accept", "confused_fallback"]

CHAR_NGRAMS = (2, 3, 4, 5)
WORD_NGRAMS = (1, 2, 3)

CORRECTOR_MAX_LEN = 96
CORRECTOR_EMBED = 160
CORRECTOR_HIDDEN = 320
CORRECTOR_BATCH = 64
CORRECTOR_EPOCHS = 18
CORRECTOR_PATIENCE = 4
CORRECTOR_LR = 8e-4
CORRECTOR_WEIGHT_DECAY = 1e-4

ALIGN_HIDDEN = 320
ALIGN_DROPOUT = 0.22
ALIGN_BATCH = 128
ALIGN_EPOCHS = 26
ALIGN_PATIENCE = 6
ALIGN_LR = 8e-4
ALIGN_WEIGHT_DECAY = 2e-3
ALIGN_MAX_VOCAB = 24000

ENTITY_FIXES = {
    "waht": "what",
    "teh": "the",
    "wich": "which",
    "hwat": "what",
    "th": "the",
    "gltich": "glitch",
    "glich": "glitch",
    "gltich": "glitch",
    "glotch": "glitch",
    "unlimited": "Unlim8ted",
    "unlimted": "Unlim8ted",
    "meatbal": "meatball",
    "meetball": "meatball",
    "time cat": "TimeCat",
    "seinfield": "Seinfeld",
    "seinfeld": "Seinfeld",
}

random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return {} if default is None else default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def normalize_text(text):
    text = str(text or "").replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_pair_text(text):
    text = normalize_text(text).lower()
    text = re.sub(r"[^a-z0-9!?.,' +\-*/=()%$:_]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def titlecase_known_entities(text):
    out = str(text or "")
    replacements = [
        ("the glitch", "The Glitch"),
        ("timecat", "TimeCat"),
        ("time cat", "TimeCat"),
        ("unlim8ted", "Unlim8ted"),
        ("meatball ai", "Meatball AI"),
        ("seinfeld", "Seinfeld"),
    ]
    for source, target in replacements:
        out = re.sub(rf"\b{re.escape(source)}\b", target, out, flags=re.IGNORECASE)
    return out


def iter_product_rows():
    data = load_json(PRODUCTS_PATH, default={})
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        rows = data.get("products") or data.get("items") or []
    else:
        rows = []

    for row in rows:
        if isinstance(row, dict):
            yield row


def build_product_answer(row):
    name = normalize_text(row.get("title") or row.get("name") or "")
    description = normalize_text(row.get("description", ""))
    product_type = normalize_text(
        row.get("product-type") or row.get("type") or "product"
    )
    link = normalize_text(row.get("link") or row.get("file") or "")
    price = row.get("price")

    segments = []
    if name and product_type:
        segments.append(f"{name} is an Unlim8ted {product_type}.")
    elif name:
        segments.append(f"{name} is an Unlim8ted product.")
    if description:
        segments.append(description)
    if isinstance(price, (int, float)) and price == 0:
        segments.append("It is currently listed with no price.")
    elif isinstance(price, (int, float)):
        segments.append(f"It is listed at {price}.")
    if link:
        segments.append("It has a published link in the product catalog.")
    return normalize_text(" ".join(segments))


def iter_product_qa_pairs():
    seen = set()
    for row in iter_product_rows():
        name = normalize_text(row.get("title") or row.get("name") or "")
        if not name:
            continue
        answer = build_product_answer(row)
        for question in (
            f"What is {name}?",
            f"Tell me about {name}.",
            f"Give me an overview of {name}.",
            f"What should I know about {name}?",
        ):
            key = (question.lower(), answer.lower())
            if key in seen:
                continue
            seen.add(key)
            yield {
                "question": question,
                "answer": answer,
                "source": "unlim8ted:products.json",
            }


def iter_local_qa_pairs():
    for pair in iter_product_qa_pairs():
        yield pair

    for path in sorted(SPECIALIZED_DIR.glob("*.jsonl")):
        for row in load_jsonl(path):
            question = normalize_text(row.get("question", ""))
            answer = normalize_text(row.get("answer", ""))
            if question and answer:
                yield {
                    "question": question,
                    "answer": answer,
                    "source": f"unlim8ted:{path.name}",
                }

    for row in load_jsonl(SMART_QA_PATH):
        question = normalize_text(row.get("question", ""))
        answer = normalize_text(row.get("answer", ""))
        if question and answer:
            yield {
                "question": question,
                "answer": answer,
                "source": "unlim8ted:SmartMeatballQA",
            }


def iter_public_qa_pairs():
    # if CELESTIAL_PATH.exists():
    #    with CELESTIAL_PATH.open("r", encoding="utf-8-sig", newline="") as f:
    #        reader = csv.DictReader(f)
    #        for idx, row in enumerate(reader):
    #            name = normalize_text(row.get("Common / Nickname", ""))
    #            kind = normalize_text(row.get("Type", ""))
    #            mag = normalize_text(row.get("Magnitude (SIMBAD)", ""))
    #            if not name or not kind:
    #                continue
    #            answer = f"{name} is a {kind.lower()} listed in a public celestial object catalog."
    #            if mag and mag != "--":
    #                answer += f" Its catalog magnitude is {mag}."
    #            yield {
    #                "question": f"What is {name}?",
    #                "answer": answer,
    #                "source": "public:celestial_catalog",
    #            }
    #            yield {
    #                "question": f"Tell me about {name}.",
    #                "answer": answer,
    #                "source": "public:celestial_catalog",
    #            }
    #            if idx >= 1600:
    #                break

    for row in load_jsonl(WIZARD_PATH):
        message = normalize_text(row.get("message", ""))
        subject = normalize_text(row.get("target_subject", "")) or normalize_text(
            row.get("conversation_topic", "")
        )
        if not message or not subject:
            continue
        yield {
            "question": message,
            "answer": f"{subject} is the subject being discussed in this public Wizard of Wikipedia example.",
            "source": "public:wizard_of_wikipedia",
        }


def iter_product_titles():
    for row in iter_product_rows():
        title = normalize_text(row.get("title") or row.get("name") or "")
        if title:
            yield title


def iter_real_typo_pairs():
    for path in TYPO_DATA_CANDIDATES:
        if not path.exists():
            continue

        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            for row in load_jsonl(path):
                noisy = normalize_text(
                    row.get("noisy")
                    or row.get("misspelling")
                    or row.get("misspelled")
                    or row.get("source")
                    or row.get("input")
                    or row.get("typo")
                    or ""
                )
                clean = normalize_text(
                    row.get("clean")
                    or row.get("correction")
                    or row.get("target")
                    or row.get("output")
                    or row.get("correct")
                    or ""
                )
                if noisy and clean and noisy.lower() != clean.lower():
                    yield noisy, clean, path.name
            continue

        if suffix in {".tsv", ".csv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            with path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f, delimiter=delimiter)
                if reader.fieldnames:
                    for row in reader:
                        noisy = normalize_text(
                            row.get("noisy")
                            or row.get("misspelling")
                            or row.get("misspelled")
                            or row.get("source")
                            or row.get("input")
                            or row.get("typo")
                            or ""
                        )
                        clean = normalize_text(
                            row.get("clean")
                            or row.get("correction")
                            or row.get("target")
                            or row.get("output")
                            or row.get("correct")
                            or ""
                        )
                        if noisy and clean and noisy.lower() != clean.lower():
                            yield noisy, clean, path.name
                else:
                    f.seek(0)
                    raw_reader = csv.reader(f, delimiter=delimiter)
                    for row in raw_reader:
                        if len(row) < 2:
                            continue
                        noisy = normalize_text(row[0])
                        clean = normalize_text(row[1])
                        if noisy and clean and noisy.lower() != clean.lower():
                            yield noisy, clean, path.name


def simple_word_forms(text):
    raw = normalize_text(text)
    variants = {raw}
    if raw:
        variants.add(raw.replace("-", " "))
        variants.add(raw.replace("_", " "))
        variants.add(raw.replace("—", " "))
    for variant in list(variants):
        variants.add(normalize_text(re.sub(r"(?<=[a-z])(?=[A-Z])", " ", variant)))
    return {variant for variant in variants if variant}


def collect_canonical_questions(limit):
    seen = set()
    out = []
    for row in iter_product_rows():
        title = normalize_text(row.get("title") or row.get("name") or "")
        if not title:
            continue
        for question in (
            f"What is {title}?",
            f"Tell me about {title}.",
            f"Give me an overview of {title}.",
            f"What should I know about {title}?",
        ):
            key = question.lower()
            if key not in seen:
                seen.add(key)
                out.append(question)

    for pair in iter_local_qa_pairs():
        key = pair["question"].lower()
        if key not in seen:
            seen.add(key)
            out.append(pair["question"])
        if len(out) >= limit:
            return out

    for pair in iter_public_qa_pairs():
        key = pair["question"].lower()
        if key not in seen:
            seen.add(key)
            out.append(pair["question"])
        if len(out) >= limit:
            return out

    return out[:limit]


def keyboard_mutation(ch):
    swaps = {
        "a": "s",
        "s": "a",
        "e": "r",
        "r": "e",
        "i": "o",
        "o": "i",
        "t": "y",
        "y": "t",
        "n": "m",
        "m": "n",
        "c": "v",
        "v": "c",
        "g": "h",
        "h": "g",
        "l": "k",
        "k": "l",
        "w": "q",
    }
    return swaps.get(ch, ch)


def mutate_word(word):
    if len(word) < 2:
        return word

    mutation = random.choice(
        ["delete", "transpose", "substitute", "duplicate", "drop_vowel"]
    )
    index = random.randrange(len(word))

    if mutation == "delete":
        return word[:index] + word[index + 1 :]

    if mutation == "transpose" and index < len(word) - 1:
        chars = list(word)
        chars[index], chars[index + 1] = chars[index + 1], chars[index]
        return "".join(chars)

    if mutation == "substitute":
        chars = list(word)
        chars[index] = keyboard_mutation(chars[index])
        return "".join(chars)

    if mutation == "duplicate":
        return word[: index + 1] + word[index] + word[index + 1 :]

    vowels = [i for i, ch in enumerate(word) if ch in "aeiou"]
    if vowels:
        vowel_index = random.choice(vowels)
        return word[:vowel_index] + word[vowel_index + 1 :]
    return word


def damage_text(text):
    text = normalize_text(text)
    lower = text.lower()

    for wrong, right in ENTITY_FIXES.items():
        if right.lower() in lower and random.random() < 0.55:
            lower = re.sub(rf"\b{re.escape(right.lower())}\b", wrong, lower)

    chars = []
    for ch in lower:
        roll = random.random()
        if ch.isalpha() and roll < 0.018:
            continue
        if ch.isalpha() and 0.018 <= roll < 0.05:
            chars.append(ch)
            chars.append(ch)
            continue
        if ch.isalpha() and 0.05 <= roll < 0.085:
            chars.append(keyboard_mutation(ch))
            continue
        chars.append(ch)

    out = "".join(chars)

    if random.random() < 0.42:
        out = re.sub(r"[?!.:,;\"'()]", "", out)
    if random.random() < 0.28:
        out = re.sub(r"\b(the|a|an|about|of|is)\b", "", out)
    if random.random() < 0.18:
        out = out.replace("what is", "whats")
    if random.random() < 0.18:
        out = out.replace("tell me about", "tell me abt")
    if random.random() < 0.25:
        out = out.rstrip(" ?!.") + random.choice(["", "?", "??", " pls"])

    return normalize_text(out)


def algorithmic_typo_variants(text, copies=8):
    text = normalize_text(text)
    outputs = set()
    words = text.split()
    if not words:
        return outputs

    for _ in range(copies):
        next_words = words[:]
        edits = max(1, min(3, len(next_words) // 3 or 1))
        for _edit in range(edits):
            idx = random.randrange(len(next_words))
            token = next_words[idx]
            letters_only = re.sub(r"[^A-Za-z]", "", token)
            if len(letters_only) >= 2:
                next_words[idx] = mutate_word(token.lower())
        variant = normalize_text(" ".join(next_words))
        if variant and variant.lower() != text.lower():
            outputs.add(variant)
    return outputs


def build_corrector_pairs(limit, augment_per_question):
    pairs = []
    seen = set()

    for noisy, clean, _source_name in iter_real_typo_pairs():
        canonical_clean = titlecase_known_entities(clean)
        key = (noisy.lower(), canonical_clean)
        if key in seen:
            continue
        seen.add(key)
        pairs.append((noisy, canonical_clean))
        for clean_form in simple_word_forms(clean):
            clean_key = (clean_form.lower(), clean_form)
            if clean_key in seen:
                continue
            seen.add(clean_key)
            normalized_clean = titlecase_known_entities(clean_form)
            pairs.append((normalized_clean, normalized_clean))

    for clean in collect_canonical_questions(limit):
        canonical = titlecase_known_entities(clean)
        key = (canonical.lower(), canonical)
        if key not in seen:
            pairs.append((canonical, canonical))
            seen.add(key)

        for _ in range(augment_per_question):
            noisy = damage_text(canonical)
            if not noisy or noisy.lower() == canonical.lower():
                continue
            noisy_key = (noisy.lower(), canonical)
            if noisy_key in seen:
                continue
            seen.add(noisy_key)
            pairs.append((noisy, canonical))

        for noisy in algorithmic_typo_variants(
            canonical, copies=max(4, augment_per_question * 2)
        ):
            noisy_key = (noisy.lower(), canonical)
            if noisy_key in seen:
                continue
            seen.add(noisy_key)
            pairs.append((noisy, canonical))

    random.shuffle(pairs)
    return pairs


def build_pair_features(question, answer):
    q = normalize_pair_text(question)
    a = normalize_pair_text(answer)
    merged = f"question: {q} answer: {a}"
    wrapped = f"<{merged}>"
    features = []

    for n in CHAR_NGRAMS:
        for i in range(0, max(0, len(wrapped) - n + 1)):
            features.append(f"c:{wrapped[i:i+n]}")

    words = merged.split()
    for n in WORD_NGRAMS:
        for i in range(0, max(0, len(words) - n + 1)):
            features.append(f"w:{'_'.join(words[i:i+n])}")

    q_words = set(re.findall(r"[a-z0-9']+", q))
    a_words = set(re.findall(r"[a-z0-9']+", a))
    overlap = len(
        (q_words & a_words) - {"what", "is", "the", "a", "an", "tell", "me", "about"}
    )
    if overlap == 0:
        features.append("flag:no_overlap")
    if overlap >= 2:
        features.append("flag:good_overlap")
    if len(a.split()) <= 3:
        features.append("flag:short_answer")
    if re.fullmatch(r"(yes|yeah|yep|no|nope)\.?", a):
        features.append("flag:bare_yes_no")
    if re.search(
        r"\b(speed of light|quantum|nebula|planet|seinfeld)\b", a
    ) and not re.search(r"\b(speed of light|quantum|nebula|planet|seinfeld)\b", q):
        features.append("flag:topic_drift")
    if (
        a.startswith("i don't know")
        or a.startswith("im not")
        or a.startswith("i'm not")
    ):
        features.append("flag:weak_fallback")
    return features


def build_alignment_vocab(rows):
    counts = Counter()
    for row in rows:
        counts.update(build_pair_features(row["question"], row["answer"]))
    vocab = {"<unk>": 0}
    for token, _ in counts.most_common(ALIGN_MAX_VOCAB - 1):
        vocab[token] = len(vocab)
    return vocab


def vectorize_alignment_row(question, answer, vocab):
    vec = torch.zeros(len(vocab), dtype=torch.float32)
    counts = Counter(build_pair_features(question, answer))
    for token, count in counts.items():
        vec[vocab.get(token, 0)] = min(float(count), 5.0)
    return vec


def build_alignment_rows(limit_public, limit_local):
    local_pairs = []
    public_pairs = []
    for idx, pair in enumerate(iter_local_qa_pairs()):
        local_pairs.append(pair)
        if limit_local and idx + 1 >= limit_local:
            break
    for idx, pair in enumerate(iter_public_qa_pairs()):
        public_pairs.append(pair)
        if limit_public and idx + 1 >= limit_public:
            break

    positives = local_pairs + public_pairs
    rows = [
        {
            "question": p["question"],
            "answer": p["answer"],
            "label": "accept",
            "source": p["source"],
        }
        for p in positives
    ]

    all_answers = [p["answer"] for p in positives]
    for idx, pair in enumerate(positives):
        wrong_answer = all_answers[(idx * 7 + 13) % len(all_answers)]
        if wrong_answer != pair["answer"]:
            rows.append(
                {
                    "question": pair["question"],
                    "answer": wrong_answer,
                    "label": "confused_fallback",
                    "source": f"{pair['source']}:mismatch_swap",
                }
            )

        words = pair["answer"].split()
        if words:
            rows.append(
                {
                    "question": pair["question"],
                    "answer": " ".join(words[: max(1, len(words) // 3)]),
                    "label": "confused_fallback",
                    "source": f"{pair['source']}:truncate",
                }
            )

    rows.extend(
        [
            {
                "question": "yes, what year was seinfield started",
                "answer": "the speed of light is approximately 299,792,458 meters per second.",
                "label": "confused_fallback",
                "source": "synthetic:explicit_mismatch",
            },
            {
                "question": "yes",
                "answer": "yep, we can talk about something else too.",
                "label": "accept",
                "source": "synthetic:short_chat_match",
            },
            {
                "question": "what is The Glitch",
                "answer": "The Glitch is an Unlim8ted project.",
                "label": "accept",
                "source": "synthetic:entity_match",
            },
            {
                "question": "what is The Glitch",
                "answer": "Seinfeld started in 1989.",
                "label": "confused_fallback",
                "source": "synthetic:entity_mismatch",
            },
        ]
    )

    random.shuffle(rows)
    return rows


def build_char_vocab(texts):
    chars = sorted({ch for text in texts for ch in text})
    tokens = [PAD, BOS, EOS, UNK] + chars
    return {token: idx for idx, token in enumerate(tokens)}


def encode_text(text, vocab, max_len, add_bos=False, add_eos=True):
    ids = []
    if add_bos:
        ids.append(vocab[BOS])
    for ch in normalize_text(text)[: max(1, max_len - 2)]:
        ids.append(vocab.get(ch, vocab[UNK]))
    if add_eos:
        ids.append(vocab[EOS])
    return ids[:max_len]


class CorrectionDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.rows = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        src, tgt = self.rows[idx]
        src_ids = encode_text(
            src, self.src_vocab, CORRECTOR_MAX_LEN, add_bos=True, add_eos=True
        )
        tgt_ids = encode_text(
            tgt, self.tgt_vocab, CORRECTOR_MAX_LEN, add_bos=True, add_eos=True
        )
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(
            tgt_ids, dtype=torch.long
        )


def collate_corrections(batch):
    srcs, tgts = zip(*batch)
    src_len = max(x.numel() for x in srcs)
    tgt_len = max(x.numel() for x in tgts)
    src_pad = torch.full((len(batch), src_len), 0, dtype=torch.long)
    tgt_pad = torch.full((len(batch), tgt_len), 0, dtype=torch.long)
    for i, (src, tgt) in enumerate(zip(srcs, tgts)):
        src_pad[i, : src.numel()] = src
        tgt_pad[i, : tgt.numel()] = tgt
    return src_pad, tgt_pad


class GreedyInputCorrector(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, max_len, bos_id=BOS_ID):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, CORRECTOR_EMBED, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, CORRECTOR_EMBED, padding_idx=0)
        self.encoder = nn.GRU(CORRECTOR_EMBED, CORRECTOR_HIDDEN, batch_first=True)
        self.decoder = nn.GRU(CORRECTOR_EMBED, CORRECTOR_HIDDEN, batch_first=True)
        self.head = nn.Linear(CORRECTOR_HIDDEN, tgt_vocab_size)
        self.max_len = int(max_len)
        self.bos_id = int(BOS_ID if bos_id is None else bos_id)

    def forward_train(self, src_ids, tgt_ids):
        src_emb = self.src_embed(src_ids)
        _, hidden = self.encoder(src_emb)
        decoder_input = tgt_ids[:, :-1]
        tgt_emb = self.tgt_embed(decoder_input)
        decoded, _ = self.decoder(tgt_emb, hidden)
        return self.head(decoded)

    def forward(self, src_ids):
        batch = src_ids.size(0)
        src_emb = self.src_embed(src_ids)
        _, hidden = self.encoder(src_emb)
        prev = torch.full(
            (batch, 1), self.bos_id, dtype=torch.long, device=src_ids.device
        )
        steps = []
        for _ in range(self.max_len):
            emb = self.tgt_embed(prev[:, -1:])
            decoded, hidden = self.decoder(emb, hidden)
            logits = self.head(decoded[:, -1, :])
            steps.append(logits.unsqueeze(1))
            prev = torch.cat([prev, torch.argmax(logits, dim=-1, keepdim=True)], dim=1)
        return torch.cat(steps, dim=1)


class AlignmentDataset(Dataset):
    def __init__(self, rows, vocab):
        self.rows = rows
        self.vocab = vocab

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        x = vectorize_alignment_row(row["question"], row["answer"], self.vocab)
        y = LABELS.index(row["label"])
        return x, torch.tensor(y, dtype=torch.long)


class TinyAlignmentClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        num_classes,
        hidden_size=ALIGN_HIDDEN,
        dropout=ALIGN_DROPOUT,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def export_input_corrector_onnx(model, output_path, max_len):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu")
    model.eval()
    dummy = torch.zeros(1, int(max_len), dtype=torch.long)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


def export_alignment_onnx(model, output_path, input_size):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.to("cpu")
    model.eval()
    dummy = torch.zeros(1, int(input_size), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


@torch.no_grad()
def evaluate_corrector(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    exact = 0
    total = 0
    for src_ids, tgt_ids in loader:
        src_ids = src_ids.to(DEVICE)
        tgt_ids = tgt_ids.to(DEVICE)
        logits = model.forward_train(src_ids, tgt_ids)
        target = tgt_ids[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        total_loss += float(loss.item()) * target.numel()
        total_tokens += target.numel()
        pred = logits.argmax(dim=-1)
        exact += int((pred == target).all(dim=1).sum().item())
        total += target.size(0)
    return {"loss": total_loss / max(1, total_tokens), "exact": exact / max(1, total)}


@torch.no_grad()
def evaluate_alignment(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item()) * y.size(0)
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += y.size(0)
    return {"loss": total_loss / max(1, total), "acc": correct / max(1, total)}


def train_input_corrector(args):
    pairs = build_corrector_pairs(args.corrector_limit, args.augment_per_question)
    split = max(1, int(len(pairs) * 0.12))
    val_pairs = pairs[:split]
    train_pairs = pairs[split:]

    src_vocab = build_char_vocab([src for src, _ in pairs])
    tgt_vocab = build_char_vocab([tgt for _, tgt in pairs])

    train_ds = CorrectionDataset(train_pairs, src_vocab, tgt_vocab)
    val_ds = CorrectionDataset(val_pairs, src_vocab, tgt_vocab)
    train_loader = DataLoader(
        train_ds,
        batch_size=CORRECTOR_BATCH,
        shuffle=True,
        collate_fn=collate_corrections,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CORRECTOR_BATCH,
        shuffle=False,
        collate_fn=collate_corrections,
    )

    model = GreedyInputCorrector(
        len(src_vocab), len(tgt_vocab), CORRECTOR_MAX_LEN, tgt_vocab[BOS]
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CORRECTOR_LR, weight_decay=CORRECTOR_WEIGHT_DECAY
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    best_loss = float("inf")
    best_state = None
    patience_left = CORRECTOR_PATIENCE

    for epoch in range(1, args.corrector_epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0
        for src_ids, tgt_ids in train_loader:
            src_ids = src_ids.to(DEVICE)
            tgt_ids = tgt_ids.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_train(src_ids, tgt_ids)
            target = tgt_ids[:, 1:]
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * target.numel()
            total_tokens += target.numel()

        train_loss = total_loss / max(1, total_tokens)
        metrics = evaluate_corrector(model, val_loader, loss_fn)
        print(
            f"[corrector] epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={metrics['loss']:.4f} exact={metrics['exact']:.4f}"
        )
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            patience_left = CORRECTOR_PATIENCE
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("[corrector] early stop")
                break

    INPUT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state, strict=True)
    model = model.to("cpu")
    model.eval()
    torch.save(
        {
            "model_state_dict": best_state,
            "src_vocab_size": len(src_vocab),
            "tgt_vocab_size": len(tgt_vocab),
            "max_len": CORRECTOR_MAX_LEN,
            "bos_id": tgt_vocab[BOS],
        },
        INPUT_OUT_DIR / "input_text_corrector.pt",
    )

    export_input_corrector_onnx(
        model, INPUT_OUT_DIR / "input_text_corrector.onnx", CORRECTOR_MAX_LEN
    )

    save_json(INPUT_OUT_DIR / "input_vocab.json", src_vocab)
    save_json(INPUT_OUT_DIR / "output_vocab.json", tgt_vocab)
    save_json(
        INPUT_OUT_DIR / "config.json",
        {
            "model_type": "char_seq2seq_input_corrector_v2",
            "max_len": CORRECTOR_MAX_LEN,
            "embed": CORRECTOR_EMBED,
            "hidden": CORRECTOR_HIDDEN,
            "sources": [
                "optional:github_typo_corpus",
                "public:wizard_of_wikipedia",
                "public:celestial_catalog",
                "unlim8ted:specialized_QA",
                "unlim8ted:SmartMeatballQA",
                "unlim8ted:products",
            ],
            "note": "Greedy char-level corrector trained on public prompts, every product in products.json, optional real typo pairs, and automated typo augmentation.",
        },
    )


def train_alignment_model(args):
    rows = build_alignment_rows(args.public_pairs, args.local_pairs)
    split = max(1, int(len(rows) * 0.12))
    val_rows = rows[:split]
    train_rows = rows[split:]
    vocab = build_alignment_vocab(train_rows)

    train_ds = AlignmentDataset(train_rows, vocab)
    val_ds = AlignmentDataset(val_rows, vocab)
    train_loader = DataLoader(train_ds, batch_size=ALIGN_BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=ALIGN_BATCH, shuffle=False)

    model = TinyAlignmentClassifier(len(vocab), len(LABELS)).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=ALIGN_LR, weight_decay=ALIGN_WEIGHT_DECAY
    )
    loss_fn = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_state = None
    bad_epochs = 0

    for epoch in range(1, args.align_epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * y.size(0)
            total += y.size(0)

        train_loss = total_loss / max(1, total)
        metrics = evaluate_alignment(model, val_loader, loss_fn)
        print(
            f"[alignment] epoch {epoch:03d} train_loss={train_loss:.4f} val_loss={metrics['loss']:.4f} acc={metrics['acc']:.4f}"
        )
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            bad_epochs = 0
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            bad_epochs += 1
            if bad_epochs >= ALIGN_PATIENCE:
                print("[alignment] early stop")
                break

    ALIGN_OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state, strict=True)
    model = model.to("cpu")
    model.eval()
    torch.save(
        {
            "model_state_dict": best_state,
            "input_size": len(vocab),
            "num_classes": len(LABELS),
        },
        ALIGN_OUT_DIR / "output_sanity_checker.pt",
    )

    export_alignment_onnx(
        model, ALIGN_OUT_DIR / "output_sanity_checker.onnx", len(vocab)
    )

    save_json(ALIGN_OUT_DIR / "input_vocab.json", vocab)
    save_json(ALIGN_OUT_DIR / "labels.json", LABELS)
    save_json(
        ALIGN_OUT_DIR / "config.json",
        {
            "model_type": "qa_pair_alignment_classifier_v2",
            "feature_mode": "qa_pair",
            "char_ngrams": list(CHAR_NGRAMS),
            "word_ngrams": list(WORD_NGRAMS),
            "hidden": ALIGN_HIDDEN,
            "dropout": ALIGN_DROPOUT,
            "sources": [
                "public:wizard_of_wikipedia",
                "public:celestial_catalog",
                "unlim8ted:specialized_QA",
                "unlim8ted:SmartMeatballQA",
                "unlim8ted:products.json",
            ],
            "note": "Scores whether an answer fits the input question rather than judging the answer in isolation.",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corrector_limit", type=int, default=48000)
    parser.add_argument("--augment_per_question", type=int, default=4)
    parser.add_argument("--corrector_epochs", type=int, default=CORRECTOR_EPOCHS)
    parser.add_argument("--public_pairs", type=int, default=18000)
    parser.add_argument("--local_pairs", type=int, default=22000)
    parser.add_argument("--align_epochs", type=int, default=ALIGN_EPOCHS)
    args = parser.parse_args()

    train_input_corrector(args)
    train_alignment_model(args)
    print(INPUT_OUT_DIR / "input_text_corrector.pt")
    print(INPUT_OUT_DIR / "input_text_corrector.onnx")
    print(ALIGN_OUT_DIR / "output_sanity_checker.pt")
    print(ALIGN_OUT_DIR / "output_sanity_checker.onnx")


if __name__ == "__main__":
    main()
