import argparse
import json
import random
import re
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SPECIALIZED_DIR = Path("assets/data/specialized_QA")
SMART_QA_PATH = Path("tools/SmartMeatballQA.jsonl")
SUBJECT_FINDER_PATH = Path("assets/data/subject_QA/SubjectFinder.jsonl")
OUT_DIR = Path("assets/models/input_text_corrector")

PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"

BATCH_SIZE = 64
EPOCHS = 18
PATIENCE = 4
MAX_SAMPLES = 40000
MAX_LEN = 96
EMBED = 160
HIDDEN = 320
LR = 8e-4
WEIGHT_DECAY = 1e-4
TEACHER_FORCING = 0.55

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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
                pass
    return rows


def normalize_text(text):
    text = str(text or "").strip()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def gather_canonical_questions(limit):
    questions = []
    seen = set()

    for path in sorted(SPECIALIZED_DIR.glob("*.jsonl")):
        for row in load_jsonl(path):
            question = normalize_text(row.get("question", ""))
            if not question:
                continue
            key = question.lower()
            if key in seen:
                continue
            seen.add(key)
            questions.append(question)
            if len(questions) >= limit:
                return questions

    for row in load_jsonl(SMART_QA_PATH):
        question = normalize_text(row.get("question", ""))
        if not question:
            continue
        key = question.lower()
        if key in seen:
            continue
        seen.add(key)
        questions.append(question)
        if len(questions) >= limit:
            return questions

    for row in load_jsonl(SUBJECT_FINDER_PATH):
        message = normalize_text(row.get("message", ""))
        if not message:
            continue
        key = message.lower()
        if key in seen:
            continue
        seen.add(key)
        questions.append(message)
        if len(questions) >= limit:
            return questions

    return questions


COMMON_FIXES = {
    "teh": "the",
    "waht": "what",
    "wich": "which",
    "gltich": "glitch",
    "glich": "glitch",
    "meetball": "meatball",
    "meatbal": "meatball",
    "unlimited": "Unlim8ted",
    "time cat": "TimeCat",
}


def corrupt_text(text):
    text = normalize_text(text)
    lower = text.lower()

    for wrong, right in COMMON_FIXES.items():
        if right.lower() in lower and random.random() < 0.45:
            lower = re.sub(rf"\b{re.escape(right.lower())}\b", wrong, lower)

    chars = []
    for ch in lower:
        r = random.random()
        if ch.isalpha() and r < 0.02:
            continue
        if ch.isalpha() and 0.02 <= r < 0.05:
            chars.append(ch)
            chars.append(ch)
            continue
        if ch.isalpha() and 0.05 <= r < 0.085:
            swaps = {
                "a": "s", "s": "a", "e": "r", "r": "e", "i": "o", "o": "i",
                "t": "y", "y": "t", "n": "m", "m": "n", "c": "v", "v": "c",
                "g": "h", "h": "g", "l": "k", "k": "l",
            }
            chars.append(swaps.get(ch, ch))
            continue
        chars.append(ch)

    out = "".join(chars)

    if random.random() < 0.45:
        out = re.sub(r"[?!.:,;\"'()]", "", out)

    if random.random() < 0.4:
        out = re.sub(r"\b(is|the|a|an|about|of)\b", "", out)

    out = re.sub(r"\s+", " ", out).strip()

    if random.random() < 0.5 and not out.endswith("?"):
        out += random.choice(["?", "??", "", " pls"])

    return normalize_text(out)


def build_pairs(limit, augment_per_question):
    canonical = gather_canonical_questions(limit)
    pairs = []
    seen = set()

    for clean in canonical:
        key = (clean.lower(), clean)
        if key not in seen:
            pairs.append((clean, clean))
            seen.add(key)

        for _ in range(augment_per_question):
            bad = corrupt_text(clean)
            if not bad or bad.lower() == clean.lower():
                continue
            key = (bad.lower(), clean)
            if key in seen:
                continue
            seen.add(key)
            pairs.append((bad, clean))

    random.shuffle(pairs)
    return pairs


def build_vocab(texts):
    chars = sorted({ch for text in texts for ch in text})
    tokens = [PAD, BOS, EOS, UNK] + chars
    return {token: idx for idx, token in enumerate(tokens)}


def encode_text(text, vocab, add_bos=False, add_eos=True):
    ids = []
    if add_bos:
        ids.append(vocab[BOS])
    for ch in normalize_text(text)[: MAX_LEN - 2]:
        ids.append(vocab.get(ch, vocab[UNK]))
    if add_eos:
        ids.append(vocab[EOS])
    return ids[:MAX_LEN]


class PairDataset(Dataset):
    def __init__(self, pairs, src_vocab, tgt_vocab):
        self.rows = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        src, tgt = self.rows[idx]
        src_ids = encode_text(src, self.src_vocab, add_bos=True, add_eos=True)
        tgt_ids = encode_text(tgt, self.tgt_vocab, add_bos=True, add_eos=True)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)


def collate(batch):
    srcs, tgts = zip(*batch)
    src_len = max(x.numel() for x in srcs)
    tgt_len = max(x.numel() for x in tgts)
    src_pad = torch.full((len(batch), src_len), 0, dtype=torch.long)
    tgt_pad = torch.full((len(batch), tgt_len), 0, dtype=torch.long)
    for i, (src, tgt) in enumerate(zip(srcs, tgts)):
        src_pad[i, : src.numel()] = src
        tgt_pad[i, : tgt.numel()] = tgt
    return src_pad, tgt_pad


class InputCorrector(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, EMBED, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, EMBED, padding_idx=0)
        self.encoder = nn.GRU(EMBED, HIDDEN, batch_first=True)
        self.decoder = nn.GRU(EMBED, HIDDEN, batch_first=True)
        self.head = nn.Linear(HIDDEN, tgt_vocab_size)

    def forward(self, src_ids, tgt_ids):
        src_emb = self.src_embed(src_ids)
        _, hidden = self.encoder(src_emb)

        decoder_input = tgt_ids[:, :-1]
        tgt_emb = self.tgt_embed(decoder_input)
        decoded, _ = self.decoder(tgt_emb, hidden)
        return self.head(decoded)


@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    exact = 0
    total = 0

    for src_ids, tgt_ids in loader:
        src_ids = src_ids.to(DEVICE)
        tgt_ids = tgt_ids.to(DEVICE)
        logits = model(src_ids, tgt_ids)
        target = tgt_ids[:, 1:]
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        total_loss += float(loss.item()) * target.numel()
        total_tokens += target.numel()

        pred = logits.argmax(dim=-1)
        match = (pred == target).all(dim=1)
        exact += int(match.sum().item())
        total += target.size(0)

    return {
        "loss": total_loss / max(1, total_tokens),
        "exact": exact / max(1, total),
    }


def train(args):
    pairs = build_pairs(args.limit or MAX_SAMPLES, args.augment_per_question)
    random.shuffle(pairs)
    split = max(1, int(len(pairs) * 0.12))
    val_pairs = pairs[:split]
    train_pairs = pairs[split:]

    src_vocab = build_vocab([src for src, _ in pairs])
    tgt_vocab = build_vocab([tgt for _, tgt in pairs])

    train_ds = PairDataset(train_pairs, src_vocab, tgt_vocab)
    val_ds = PairDataset(val_pairs, src_vocab, tgt_vocab)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = InputCorrector(len(src_vocab), len(tgt_vocab)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    best_loss = float("inf")
    best_state = None
    patience_left = PATIENCE

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for src_ids, tgt_ids in train_loader:
            src_ids = src_ids.to(DEVICE)
            tgt_ids = tgt_ids.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(src_ids, tgt_ids)
            target = tgt_ids[:, 1:]
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(loss.item()) * target.numel()
            total_tokens += target.numel()

        train_loss = total_loss / max(1, total_tokens)
        metrics = evaluate(model, val_loader, loss_fn)
        print(
            f"epoch {epoch:03d} | train_loss {train_loss:.4f} | "
            f"val_loss {metrics['loss']:.4f} | exact {metrics['exact']:.4f}"
        )

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            patience_left = PATIENCE
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": best_state,
                    "src_vocab_size": len(src_vocab),
                    "tgt_vocab_size": len(tgt_vocab),
                },
                OUT_DIR / "input_text_corrector.pt",
            )
            print("[saved best]")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("[early stop]")
                break

    save_json(OUT_DIR / "input_vocab.json", src_vocab)
    save_json(OUT_DIR / "output_vocab.json", tgt_vocab)
    save_json(
        OUT_DIR / "config.json",
        {
            "model_type": "char_seq2seq_input_corrector",
            "max_len": MAX_LEN,
            "embed": EMBED,
            "hidden": HIDDEN,
            "note": "Corrects noisy user questions before subject finding and routing.",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--limit", type=int, default=MAX_SAMPLES)
    parser.add_argument("--augment_per_question", type=int, default=3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
