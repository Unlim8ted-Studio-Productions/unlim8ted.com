import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SEED = 42
VAL_SPLIT = 0.12

INPUT_NGRAMS = (1, 2, 3)
MAX_INPUT_VOCAB_SIZE = 12000
MIN_INPUT_TOKEN_FREQ = 1

MAX_OUTPUT_CHUNKS = 24
MAX_CHUNK_WORDS = 8
MIN_CHUNK_OCCURRENCES = 2
MIN_CHUNK_GAIN = 4

BATCH_SIZE = 64
EPOCHS = 80
PATIENCE = 15
MIN_DELTA = 1e-4

LR = 8e-4
WEIGHT_DECAY = 2e-3
LABEL_SMOOTHING = 0.03
GRAD_CLIP = 1.0

PROMPT_SIZE = 128
HIDDEN_SIZE = 192
EMBED_SIZE = 128
DROPOUT = 0.35

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

SPECIAL_OUTPUT_CHUNKS = [PAD, BOS, EOS, UNK]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_jsonl(path: Path):
    rows = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()

            if not question or not answer:
                print(f"[skip] missing question/answer line {line_num}")
                continue

            rows.append({"question": question, "answer": answer})

    return rows


def dedupe_rows(rows):
    deduped = []
    seen_questions = set()

    for row in rows:
        key = str(row["question"]).strip().lower()
        if key in seen_questions:
            continue
        seen_questions.add(key)
        deduped.append(
            {
                "question": str(row["question"]).strip(),
                "answer": str(row["answer"]).strip(),
            }
        )

    return deduped


def load_combined_rows(specialized_dir: Path, smart_qa_path: Path):
    topic_files = sorted(specialized_dir.glob("*.jsonl"))

    if not topic_files:
        raise RuntimeError(f"No specialized .jsonl files found in {specialized_dir}")

    rows = []

    print()
    print("Loading specialized QA topic files...")
    for path in topic_files:
        topic_rows = load_jsonl(path)
        rows.extend(topic_rows)
        print(f"{path.name}: {len(topic_rows)} rows")

    if smart_qa_path.exists():
        smart_rows = load_jsonl(smart_qa_path)
        rows.extend(smart_rows)
        print(f"{smart_qa_path.name}: {len(smart_rows)} rows")
    else:
        raise FileNotFoundError(f"Missing smart QA dataset: {smart_qa_path}")

    deduped = dedupe_rows(rows)

    print()
    print(f"combined rows before dedupe: {len(rows)}")
    print(f"combined rows after dedupe:  {len(deduped)}")

    return deduped


def normalize_spaces(text: str) -> str:
    text = str(text).replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def input_normalize(text: str) -> str:
    text = str(text).lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[^a-z0-9_!?.,' -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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
    return f"question: {row.get('question', '')}"


def answer_tokenize(text: str):
    text = normalize_spaces(text)
    return re.findall(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)*|[^\w\s]", text)


def canonical_token(token: str) -> str:
    if re.match(r"[A-Za-z0-9_]+$", token):
        return token.lower()
    return token


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


def build_input_vocab(rows):
    counter = Counter()
    for row in rows:
        feats = make_input_ngrams(input_tokenize(row_to_input_text(row)))
        counter.update(feats)

    vocab = {"<PAD>": 0, "<UNK>": 1}

    for feat, count in counter.most_common(MAX_INPUT_VOCAB_SIZE - len(vocab)):
        if count < MIN_INPUT_TOKEN_FREQ:
            continue
        vocab[feat] = len(vocab)

    return vocab


def vectorize_input(row, input_vocab):
    feats = make_input_ngrams(input_tokenize(row_to_input_text(row)))
    x = torch.zeros(len(input_vocab), dtype=torch.float32)
    counts = Counter(feats)

    for feat, count in counts.items():
        idx = input_vocab.get(feat, input_vocab["<UNK>"])
        x[idx] = min(float(count), 5.0)

    return x


def build_candidate_occurrences(rows, max_chunk_words):
    occurrences = defaultdict(list)
    readable = {}

    tokenized_answers = []
    for row in rows:
        tokenized_answers.append(answer_tokenize(row["answer"]))

    for answer_idx, tokens in enumerate(tokenized_answers):
        for n in range(2, min(max_chunk_words, len(tokens)) + 1):
            for start in range(len(tokens) - n + 1):
                span = tokens[start : start + n]

                if not any(re.match(r"[A-Za-z0-9_]+$", t) for t in span):
                    continue

                key = phrase_key(span)
                occurrences[key].append((answer_idx, start, start + n))
                readable[key] = phrase_text(span)

    return tokenized_answers, occurrences, readable


def count_uncovered_gain(occ_list, uncovered_positions):
    gain = 0
    for answer_idx, start, end in occ_list:
        uncovered = uncovered_positions[answer_idx]
        for pos in range(start, end):
            if pos in uncovered:
                gain += 1
    return gain


def mark_covered(occ_list, uncovered_positions):
    for answer_idx, start, end in occ_list:
        uncovered_positions[answer_idx].difference_update(range(start, end))


def build_cover_chunks(rows, max_chunk_words=MAX_CHUNK_WORDS):
    tokenized_answers, occurrences, readable = build_candidate_occurrences(rows, max_chunk_words)

    uncovered_positions = [set(range(len(tokens))) for tokens in tokenized_answers]
    phrase_counts = {key: len(occ_list) for key, occ_list in occurrences.items()}
    selected_phrase_keys = []
    used = set()

    while True:
        best_key = None
        best_gain = 0
        best_len = 0
        best_count = 0

        for key, occ_list in occurrences.items():
            if key in used:
                continue
            if len(occ_list) < MIN_CHUNK_OCCURRENCES:
                continue

            gain = count_uncovered_gain(occ_list, uncovered_positions)
            length = len(key.split(" "))
            count = len(occ_list)

            if gain > best_gain or (
                gain == best_gain and length > best_len
            ) or (
                gain == best_gain and length == best_len and count > best_count
            ):
                best_key = key
                best_gain = gain
                best_len = length
                best_count = count

        if best_key is None or best_gain < MIN_CHUNK_GAIN:
            break

        used.add(best_key)
        selected_phrase_keys.append(best_key)
        mark_covered(occurrences[best_key], uncovered_positions)

    single_counter = Counter()
    single_text = {}
    for answer_idx, tokens in enumerate(tokenized_answers):
        for pos, token in enumerate(tokens):
            if pos not in uncovered_positions[answer_idx]:
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

    for key in sorted(
        selected_phrase_keys,
        key=lambda k: (len(k.split(" ")), phrase_counts[k], len(readable[k])),
        reverse=True,
    ):
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

    return output_chunks


def build_chunk_lookup(output_chunks):
    key_to_id = {}
    max_len = 1

    for chunk in output_chunks:
        key_to_id[chunk["key"]] = chunk["id"]
        max_len = max(max_len, int(chunk["length"]))

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


class GeneralChunkDataset(Dataset):
    def __init__(self, rows, input_vocab, key_to_id, max_chunk_words):
        self.rows = rows
        self.input_vocab = input_vocab
        self.key_to_id = key_to_id
        self.max_chunk_words = max_chunk_words

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        x = vectorize_input(row, self.input_vocab)
        y = torch.tensor(
            encode_answer_to_chunks(row["answer"], self.key_to_id, self.max_chunk_words),
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
        self.hidden_size = HIDDEN_SIZE
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
        write_hidden = self.decoder_cell(torch.cat([emb, prompt_context], dim=-1), write_hidden)
        logits = self.output(torch.cat([prompt_context, write_hidden], dim=-1))
        return logits, write_hidden

    def forward(self, x, target=None, teacher_forcing=True, max_len=MAX_OUTPUT_CHUNKS + 1):
        batch_size = x.size(0)
        prompt_context = self.encode(x)
        write_hidden = torch.zeros(batch_size, HIDDEN_SIZE, device=x.device)
        prev_token = torch.full((batch_size,), BOS_ID, dtype=torch.long, device=x.device)
        logits_steps = []

        for t in range(max_len):
            logits, write_hidden = self.decoder_step(prev_token, prompt_context, write_hidden)
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
        return self.model(x, target=None, teacher_forcing=False, max_len=MAX_OUTPUT_CHUNKS + 1)


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


def export_onnx(model, path: Path, input_size: int):
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
    parser.add_argument("--out_dir", default="assets/models/general_cover_chunks_test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_chunk_words", type=int, default=MAX_CHUNK_WORDS)
    parser.add_argument("--export_pt", action="store_true")
    args = parser.parse_args()

    print(f"device: {DEVICE}")

    rows = load_combined_rows(Path(args.specialized_dir), Path(args.smart_qa_path))
    if args.limit and len(rows) > args.limit:
        rows = rows[: args.limit]

    if len(rows) < 10:
        raise RuntimeError(f"Need at least 10 usable rows, found {len(rows)}")

    print(f"general rows: {len(rows)}")

    output_chunks = build_cover_chunks(rows, max_chunk_words=args.max_chunk_words)
    key_to_id, max_chunk_words_used = build_chunk_lookup(output_chunks)
    input_vocab = build_input_vocab(rows)

    encoded_lengths = []
    unk_count = 0
    for row in rows:
        ids = encode_answer_to_chunks(row["answer"], key_to_id, max_chunk_words_used)
        stripped = strip_special(ids)
        encoded_lengths.append(len(stripped))
        unk_count += sum(1 for idx in stripped if idx == UNK_ID)

    real_chunks = [c for c in output_chunks if not c.get("special")]
    multi_chunks = [c for c in real_chunks if c["length"] >= 2]
    single_chunks = [c for c in real_chunks if c["length"] == 1]

    print(f"input vocab size:           {len(input_vocab)}")
    print(f"output chunk vocab size:    {len(output_chunks)}")
    print(f"multi-token chunks:         {len(multi_chunks)}")
    print(f"single-token chunks:        {len(single_chunks)}")
    print(f"avg chunks per answer:      {sum(encoded_lengths) / max(len(encoded_lengths), 1):.2f}")
    print(f"max chunks per answer:      {max(encoded_lengths)}")
    print(f"encoded unknown chunk uses: {unk_count}")

    random.shuffle(rows)
    split_idx = int(len(rows) * (1.0 - VAL_SPLIT))
    train_rows = rows[:split_idx]
    val_rows = rows[split_idx:] or rows[:]

    train_ds = GeneralChunkDataset(train_rows, input_vocab, key_to_id, max_chunk_words_used)
    val_ds = GeneralChunkDataset(val_rows, input_vocab, key_to_id, max_chunk_words_used)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = ChunkAnswerModel(len(input_vocab), len(output_chunks), manual_gru=False).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID, label_smoothing=LABEL_SMOOTHING)

    best_val_loss = math.inf
    epochs_without_improvement = 0
    best_checkpoint = None

    print()
    print("[training general cover-chunk model]")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss_total = 0.0
        batches = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, target=y, teacher_forcing=True, max_len=MAX_OUTPUT_CHUNKS + 1)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss_total += float(loss.item())
            batches += 1

        train_loss = train_loss_total / max(batches, 1)
        metrics = evaluate(model, val_loader, criterion)

        print(
            f"[general-cover] epoch {epoch:03d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {metrics['loss']:.4f} | "
            f"val_token_acc {metrics['token_acc']:.4f} | "
            f"val_exact_seq {metrics['exact_seq']:.4f}"
        )

        if metrics["loss"] < best_val_loss - MIN_DELTA:
            best_val_loss = metrics["loss"]
            epochs_without_improvement = 0
            best_checkpoint = {
                "model_state_dict": clone_state_dict(model),
                "input_vocab_size": len(input_vocab),
                "output_vocab_size": len(output_chunks),
                "best_val_loss": best_val_loss,
            }
            print("[saved best state]")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"[early stop] no val improvement for {PATIENCE} epochs")
                break

    if best_checkpoint is None:
        best_checkpoint = {
            "model_state_dict": clone_state_dict(model),
            "input_vocab_size": len(input_vocab),
            "output_vocab_size": len(output_chunks),
            "best_val_loss": best_val_loss,
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.load_state_dict(best_checkpoint["model_state_dict"])
    model.eval()

    onnx_model = ChunkAnswerModel(len(input_vocab), len(output_chunks), manual_gru=True).to(DEVICE)
    onnx_model.load_state_dict(best_checkpoint["model_state_dict"], strict=False)
    onnx_model.eval()

    save_json(out_dir / "input_vocab.json", input_vocab)
    save_json(out_dir / "output_chunks.json", output_chunks)
    save_json(
        out_dir / "config.json",
        {
            "model_type": "general_cover_chunk_experiment",
            "topic": "general",
            "input_type": "bag_of_words_ngrams",
            "input_ngrams": list(INPUT_NGRAMS),
            "max_output_chunks": MAX_OUTPUT_CHUNKS,
            "chunk_strategy": "greedy_coverage_min_chunk_vocab",
            "max_chunk_words": int(args.max_chunk_words),
            "min_chunk_occurrences": MIN_CHUNK_OCCURRENCES,
            "min_chunk_gain": MIN_CHUNK_GAIN,
            "model_onnx_path": "model.onnx",
            "input_vocab_path": "input_vocab.json",
            "output_chunks_path": "output_chunks.json",
            "note": "Standalone experimental general trainer using greedy cover chunks instead of the shared specialist chunk mining.",
        },
    )

    export_onnx(onnx_model, out_dir / "model.onnx", len(input_vocab))

    if args.export_pt:
        torch.save(best_checkpoint, out_dir / "model.pt")

    print()
    print("DONE")
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
