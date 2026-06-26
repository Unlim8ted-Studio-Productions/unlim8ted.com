# train_general_cover_chunks_noisy_continue.py
# Continues training an existing general_cover_chunks model with noisy inputs.
# Uses existing input_vocab/output_chunks/model.pt.
# Run original trainer with --export_pt first if model.pt does not exist.

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
VAL_SPLIT = 0.12

INPUT_NGRAMS = (1, 2, 3)

MAX_OUTPUT_CHUNKS = 24
BATCH_SIZE = 64
EPOCHS = 25
PATIENCE = 8
MIN_DELTA = 1e-4

LR = 2e-4
WEIGHT_DECAY = 2e-3
LABEL_SMOOTHING = 0.03
GRAD_CLIP = 1.0

PROMPT_SIZE = 128
HIDDEN_SIZE = 192
EMBED_SIZE = 128
DROPOUT = 0.35

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            q = str(row.get("question", "")).strip()
            a = str(row.get("answer", "")).strip()
            if q and a:
                rows.append({"question": q, "answer": a})
    return rows


def dedupe_rows(rows):
    out = []
    seen = set()
    for row in rows:
        key = row["question"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def load_combined_rows(specialized_dir, smart_qa_path):
    rows = []

    files = sorted(Path(specialized_dir).glob("*.jsonl"))
    if not files:
        raise RuntimeError(f"No specialized .jsonl files found in {specialized_dir}")

    print("Loading specialized QA topic files...")
    for path in files:
        part = load_jsonl(path)
        rows.extend(part)
        print(f"{path.name}: {len(part)} rows")

    smart_qa_path = Path(smart_qa_path)
    if smart_qa_path.exists():
        part = load_jsonl(smart_qa_path)
        rows.extend(part)
        print(f"{smart_qa_path.name}: {len(part)} rows")
    else:
        raise FileNotFoundError(f"Missing smart QA dataset: {smart_qa_path}")

    rows = dedupe_rows(rows)
    print(f"combined rows after dedupe: {len(rows)}")
    return rows


def normalize_spaces(text):
    return re.sub(r"\s+", " ", str(text).replace("\n", " ")).strip()


def corrupt_text(text):
    text = str(text).lower()

    # remove punctuation often
    if random.random() < 0.75:
        text = re.sub(r"[!?.,:;\"'`“”‘’()\[\]{}]", " ", text)

    # slang / typo replacements
    replacements = {
        "what is": ["whats", "wut is", "what"],
        "tell me about": ["tell me bout", "info on", "more on"],
        "who made": ["who made", "who built", "who created"],
        "does it": ["does it", "do it", "it"],
        "unlim8ted": ["unlimited", "unlimted", "unlim8ed", "unlim8ted"],
        "timecat": ["time cat", "time-cat", "tmecat", "timecat", "cat game"],
        "meatball": ["meetball", "meat ball", "meatbal", "meatball"],
        "glitch": ["gltich", "glitch", "the glitch"],
    }

    for key, vals in replacements.items():
        if key in text and random.random() < 0.65:
            text = text.replace(key, random.choice(vals))

    chars = []
    for ch in text:
        r = random.random()

        # delete characters
        if ch.isalpha() and r < 0.025:
            continue

        # duplicate characters
        if ch.isalpha() and 0.025 <= r < 0.045:
            chars.append(ch)
            chars.append(ch)
            continue

        # adjacent keyboard-ish substitutions
        if ch.isalpha() and 0.045 <= r < 0.065:
            substitutions = {
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
            }
            chars.append(substitutions.get(ch, ch))
            continue

        chars.append(ch)

    text = "".join(chars)

    # randomly drop small words
    if random.random() < 0.45:
        toks = text.split()
        toks = [
            t
            for t in toks
            if not (
                t in {"the", "a", "an", "is", "are", "do", "does", "to"}
                and random.random() < 0.5
            )
        ]
        text = " ".join(toks)

    # remove spaces sometimes in named-ish terms
    if random.random() < 0.25:
        text = text.replace("time cat", "timecat")
        text = text.replace("meat ball", "meatball")

    return re.sub(r"\s+", " ", text).strip()


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
        for i in range(len(tokens) - n + 1):
            feats.append("_".join(tokens[i : i + n]))
    return feats


def row_to_input_text(row, noisy=False):
    q = row.get("question", "")
    if noisy:
        q = corrupt_text(q)
    return f"question: {q}"


def vectorize_input(row, input_vocab, noisy=False):
    feats = make_input_ngrams(input_tokenize(row_to_input_text(row, noisy=noisy)))
    x = torch.zeros(len(input_vocab), dtype=torch.float32)
    counts = Counter(feats)
    unk = input_vocab.get("<UNK>", 1)

    for feat, count in counts.items():
        idx = input_vocab.get(feat, unk)
        x[idx] = min(float(count), 5.0)

    return x


def answer_tokenize(text):
    text = normalize_spaces(text)
    return re.findall(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*|[^\w\s]", text)


def canonical_token(token):
    if re.match(r"[A-Za-z0-9_]+$", token):
        return token.lower()
    return token


def canonical_tokens(tokens):
    return [canonical_token(t) for t in tokens]


def build_chunk_lookup(output_chunks):
    key_to_id = {}
    max_len = 1
    for chunk in output_chunks:
        key_to_id[chunk["key"]] = int(chunk["id"])
        max_len = max(max_len, int(chunk.get("length", 1)))
    return key_to_id, max_len


def encode_answer_to_chunks(answer, key_to_id, max_chunk_words):
    tokens = answer_tokenize(answer)
    canon = canonical_tokens(tokens)

    ids = []
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

        if len(ids) >= MAX_OUTPUT_CHUNKS:
            break

    ids.append(EOS_ID)
    while len(ids) < MAX_OUTPUT_CHUNKS + 1:
        ids.append(PAD_ID)

    return ids[: MAX_OUTPUT_CHUNKS + 1]


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


class NoisyContinueDataset(Dataset):
    def __init__(
        self, rows, input_vocab, key_to_id, max_chunk_words, noisy_probability=0.85
    ):
        self.rows = rows
        self.input_vocab = input_vocab
        self.key_to_id = key_to_id
        self.max_chunk_words = max_chunk_words
        self.noisy_probability = noisy_probability

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        noisy = random.random() < self.noisy_probability

        x = vectorize_input(row, self.input_vocab, noisy=noisy)
        y = torch.tensor(
            encode_answer_to_chunks(
                row["answer"], self.key_to_id, self.max_chunk_words
            ),
            dtype=torch.long,
        )
        return x, y


class ManualGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.weight_ih = nn.Parameter(torch.empty(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(3 * hidden_size))
        self.hidden_size = hidden_size
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, x, h):
        gi = torch.matmul(x, self.weight_ih.t()) + self.bias_ih
        gh = torch.matmul(h, self.weight_hh.t()) + self.bias_hh
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return n + z * (h - n)


class ChunkAnswerModel(nn.Module):
    def __init__(self, input_size, output_vocab_size, manual_gru=False):
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

        if manual_gru:
            self.decoder_cell = ManualGRUCell(PROMPT_SIZE + EMBED_SIZE, HIDDEN_SIZE)
        else:
            self.decoder_cell = nn.GRUCell(PROMPT_SIZE + EMBED_SIZE, HIDDEN_SIZE)

        self.output = nn.Sequential(
            nn.LayerNorm(PROMPT_SIZE + HIDDEN_SIZE),
            nn.Dropout(DROPOUT if not manual_gru else 0.0),
            nn.Linear(PROMPT_SIZE + HIDDEN_SIZE, output_vocab_size),
        )

    def encode(self, x):
        return self.encoder(x)

    def decoder_step(self, prev_token, prompt_context, write_hidden):
        emb = self.embedding(prev_token)
        write_hidden = self.decoder_cell(
            torch.cat([emb, prompt_context], dim=-1), write_hidden
        )
        logits = self.output(torch.cat([prompt_context, write_hidden], dim=-1))
        return logits, write_hidden

    def forward(
        self, x, target=None, teacher_forcing=True, max_len=MAX_OUTPUT_CHUNKS + 1
    ):
        batch_size = x.size(0)
        prompt_context = self.encode(x)
        write_hidden = torch.zeros(batch_size, HIDDEN_SIZE, device=x.device)
        prev_token = torch.full(
            (batch_size,), BOS_ID, dtype=torch.long, device=x.device
        )

        logits_steps = []

        for t in range(max_len):
            logits, write_hidden = self.decoder_step(
                prev_token, prompt_context, write_hidden
            )
            logits_steps.append(logits.unsqueeze(1))

            if teacher_forcing and target is not None:
                prev_token = target[:, t]
            else:
                prev_token = torch.argmax(logits, dim=-1)

        return torch.cat(logits_steps, dim=1)


class ExportWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(
            x, target=None, teacher_forcing=False, max_len=MAX_OUTPUT_CHUNKS + 1
        )


def clone_state_dict(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


@torch.no_grad()
def evaluate(model, loader, criterion):
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

        logits = model(x, target=y, teacher_forcing=True, max_len=MAX_OUTPUT_CHUNKS + 1)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

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


def export_onnx(model, path, input_size):
    model.eval()
    dummy = torch.zeros(1, input_size, dtype=torch.float32, device=DEVICE)
    wrapper = ExportWrapper(model).to(DEVICE)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialized_dir", default="assets/data/specialized_QA")
    parser.add_argument("--smart_qa_path", default="tools/SmartMeatballQA.jsonl")
    parser.add_argument(
        "--model_dir", default="assets/models/general_cover_chunks_test_retrained"
    )
    parser.add_argument(
        "--out_dir", default="assets/models/general_cover_chunks_noisy_continue"
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--noisy_probability", type=float, default=0.85)
    parser.add_argument("--export_pt", action="store_true")
    args = parser.parse_args()

    print(f"device: {DEVICE}")

    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_vocab = load_json(model_dir / "input_vocab.json")
    output_chunks = load_json(model_dir / "output_chunks.json")
    key_to_id, max_chunk_words = build_chunk_lookup(output_chunks)

    checkpoint_path = model_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing {checkpoint_path}. Re-run original trainer with --export_pt first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    rows = load_combined_rows(Path(args.specialized_dir), Path(args.smart_qa_path))

    if args.limit and len(rows) > args.limit:
        rows = rows[: args.limit]

    random.shuffle(rows)

    split_idx = int(len(rows) * (1.0 - VAL_SPLIT))
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:] or rows[:]

    train_ds = NoisyContinueDataset(
        train_rows,
        input_vocab,
        key_to_id,
        max_chunk_words,
        noisy_probability=args.noisy_probability,
    )

    val_ds = NoisyContinueDataset(
        val_rows,
        input_vocab,
        key_to_id,
        max_chunk_words,
        noisy_probability=1.0,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
    )

    model = ChunkAnswerModel(len(input_vocab), len(output_chunks), manual_gru=False).to(
        DEVICE
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_ID, label_smoothing=LABEL_SMOOTHING
    )

    best_val = math.inf
    bad_epochs = 0
    best_checkpoint = None

    print("continue training with noisy/randomized inputs")
    print(f"rows: {len(rows)}")
    print(f"train: {len(train_rows)}")
    print(f"val: {len(val_rows)}")
    print(f"input vocab: {len(input_vocab)}")
    print(f"output chunks: {len(output_chunks)}")
    print(f"noisy_probability: {args.noisy_probability}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        batches = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(
                x, target=y, teacher_forcing=True, max_len=MAX_OUTPUT_CHUNKS + 1
            )
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total += float(loss.item())
            batches += 1

        train_loss = total / max(batches, 1)
        metrics = evaluate(model, val_loader, criterion)

        print(
            f"[noisy-continue] epoch {epoch:03d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {metrics['loss']:.4f} | "
            f"val_token_acc {metrics['token_acc']:.4f} | "
            f"val_exact_seq {metrics['exact_seq']:.4f}"
        )

        if metrics["loss"] < best_val - MIN_DELTA:
            best_val = metrics["loss"]
            bad_epochs = 0
            best_checkpoint = {
                "model_state_dict": clone_state_dict(model),
                "input_vocab_size": len(input_vocab),
                "output_vocab_size": len(output_chunks),
                "best_val_loss": best_val,
                "metrics": metrics,
            }

            torch.save(best_checkpoint, out_dir / "model.pt")
            print(f"[saved best] {out_dir / 'model.pt'}")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("[early stop]")
                break

    if best_checkpoint is None:
        best_checkpoint = {
            "model_state_dict": clone_state_dict(model),
            "input_vocab_size": len(input_vocab),
            "output_vocab_size": len(output_chunks),
            "best_val_loss": best_val,
        }

    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    onnx_model = ChunkAnswerModel(
        len(input_vocab), len(output_chunks), manual_gru=True
    ).to(DEVICE)
    onnx_model.load_state_dict(best_checkpoint["model_state_dict"], strict=False)
    onnx_model.eval()

    save_json(out_dir / "input_vocab.json", input_vocab)
    save_json(out_dir / "output_chunks.json", output_chunks)
    save_json(
        out_dir / "config.json",
        {
            "model_type": "general_cover_chunk_noisy_continue",
            "base_model_dir": str(model_dir).replace("\\", "/"),
            "input_type": "bag_of_words_ngrams",
            "input_ngrams": list(INPUT_NGRAMS),
            "max_output_chunks": MAX_OUTPUT_CHUNKS,
            "noise": "random punctuation removal, typo chars, dropped chars, duplicated chars, slang replacements",
            "noisy_probability": args.noisy_probability,
            "model_onnx_path": "model.onnx",
            "input_vocab_path": "input_vocab.json",
            "output_chunks_path": "output_chunks.json",
        },
    )

    export_onnx(onnx_model, out_dir / "model.onnx", len(input_vocab))

    if args.export_pt:
        torch.save(best_checkpoint, out_dir / "model.pt")

    print("DONE")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
