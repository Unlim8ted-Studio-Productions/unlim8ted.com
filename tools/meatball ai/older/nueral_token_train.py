import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sentence_transformers import SentenceTransformer

# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_FRAGMENTS_PATH = Path("assets/data/fragments.jsonl")
DEFAULT_TRAINING_PATH = Path("assets/data/fragments-training.jsonl")
DEFAULT_OUT_PATH = Path("assets/models/token-fragment-selector.pt")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

SPECIAL_TOKENS = {
    "<PAD>": PAD_ID,
    "<BOS>": BOS_ID,
    "<EOS>": EOS_ID,
}


# ============================================================
# UTIL
# ============================================================


def clean_text(x):
    x = str(x or "").strip()
    x = re.sub(r"\s+", " ", x)
    return x


def load_jsonl(path):
    rows = []

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"Bad JSON skipped at {path}:{line_num}: {e}")
                continue

            if isinstance(obj, dict):
                rows.append(obj)

    return rows


def assemble_fragments(fragment_ids, id_to_text):
    out = ""

    for fid in fragment_ids:
        piece = clean_text(id_to_text.get(fid, ""))

        if not piece:
            continue

        if not out:
            out = piece
        elif piece in {".", ",", "?", "!", ":", ";"}:
            out += piece
        elif out.endswith((" ", "\n")):
            out += piece
        else:
            out += " " + piece

    return out.strip()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================


class TokenFragmentDataset(Dataset):
    def __init__(
        self,
        training_rows,
        fragment_id_to_model_id,
        input_embeddings,
        max_output_len,
    ):
        self.rows = []
        self.input_embeddings = input_embeddings
        self.max_output_len = max_output_len
        self.fragment_id_to_model_id = fragment_id_to_model_id

        skipped = 0

        for i, row in enumerate(training_rows):
            target = row.get("target", {})
            frag_ids = target.get("fragments", [])

            if not isinstance(frag_ids, list) or not frag_ids:
                skipped += 1
                continue

            token_ids = []

            ok = True
            for fid in frag_ids:
                if fid not in fragment_id_to_model_id:
                    ok = False
                    break

                token_ids.append(fragment_id_to_model_id[fid])

            if not ok:
                skipped += 1
                continue

            # decoder target = BOS + fragments + EOS
            seq = [BOS_ID] + token_ids + [EOS_ID]

            if len(seq) > max_output_len:
                seq = seq[:max_output_len]
                seq[-1] = EOS_ID

            while len(seq) < max_output_len:
                seq.append(PAD_ID)

            self.rows.append(
                {
                    "row_index": i,
                    "input": clean_text(row.get("input")),
                    "seq": torch.tensor(seq, dtype=torch.long),
                }
            )

        print("Training rows loaded:", len(self.rows))
        print("Training rows skipped:", skipped)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        emb = self.input_embeddings[row["row_index"]]

        return {
            "x": torch.tensor(emb, dtype=torch.float32),
            "seq": row["seq"],
            "input": row["input"],
            "row_index": row["row_index"],
        }


# ============================================================
# MODEL
# ============================================================


class TokenFragmentSelector(nn.Module):
    def __init__(
        self,
        input_dim,
        vocab_size,
        hidden_dim=384,
        max_output_len=16,
        dropout=0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.max_output_len = max_output_len

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        self.decoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )

        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, decoder_input_ids):
        """
        x: [batch, input_dim]
        decoder_input_ids: [batch, seq_len]
        """
        encoded = self.encoder(x)
        h0 = encoded.unsqueeze(0).repeat(2, 1, 1)

        tok = self.token_embedding(decoder_input_ids)

        out, _ = self.decoder(tok, h0)

        logits = self.output(out)

        return logits

    @torch.inference_mode()
    def generate(self, x, max_len=None):
        self.eval()

        if max_len is None:
            max_len = self.max_output_len

        batch_size = x.shape[0]
        device = x.device

        encoded = self.encoder(x)
        hidden = encoded.unsqueeze(0).repeat(2, 1, 1)

        current = torch.full(
            (batch_size, 1),
            BOS_ID,
            dtype=torch.long,
            device=device,
        )

        generated = []

        for _ in range(max_len - 1):
            tok = self.token_embedding(current)
            out, hidden = self.decoder(tok, hidden)
            logits = self.output(out[:, -1, :])

            next_id = torch.argmax(logits, dim=-1)

            generated.append(next_id)
            current = next_id.unsqueeze(1)

        return torch.stack(generated, dim=1)


# ============================================================
# METRICS
# ============================================================


def strip_special(seq):
    out = []

    for x in seq:
        x = int(x)

        if x == PAD_ID:
            continue
        if x == BOS_ID:
            continue
        if x == EOS_ID:
            break

        out.append(x)

    return out


def seq_f1(pred, gold):
    pred = strip_special(pred)
    gold = strip_special(gold)

    if not pred and not gold:
        return 1.0

    if not pred or not gold:
        return 0.0

    pred_counts = {}
    gold_counts = {}

    for x in pred:
        pred_counts[x] = pred_counts.get(x, 0) + 1

    for x in gold:
        gold_counts[x] = gold_counts.get(x, 0) + 1

    overlap = 0

    for x, c in pred_counts.items():
        overlap += min(c, gold_counts.get(x, 0))

    precision = overlap / max(len(pred), 1)
    recall = overlap / max(len(gold), 1)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def exact_order_match(pred, gold):
    return strip_special(pred) == strip_special(gold)


def exact_set_match(pred, gold):
    return set(strip_special(pred)) == set(strip_special(gold))


# ============================================================
# TRAIN / EVAL
# ============================================================


def run_epoch(model, loader, optimizer, device):
    model.train()

    total_loss = 0.0
    total_items = 0

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    for batch in loader:
        x = batch["x"].to(device)
        seq = batch["seq"].to(device)

        decoder_in = seq[:, :-1]
        labels = seq[:, 1:]

        logits = model(x, decoder_in)

        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()

    total_f1 = 0.0
    total_exact_order = 0
    total_exact_set = 0
    total = 0

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    total_loss = 0.0

    for batch in loader:
        x = batch["x"].to(device)
        seq = batch["seq"].to(device)

        decoder_in = seq[:, :-1]
        labels = seq[:, 1:]

        logits = model(x, decoder_in)

        loss = loss_fn(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
        )

        pred = model.generate(x, max_len=seq.shape[1])

        for p, g in zip(pred.cpu().tolist(), seq.cpu().tolist()):
            total_f1 += seq_f1(p, g)
            total_exact_order += int(exact_order_match(p, g))
            total_exact_set += int(exact_set_match(p, g))
            total += 1

        total_loss += float(loss.item()) * x.shape[0]

    return {
        "loss": total_loss / max(total, 1),
        "f1": total_f1 / max(total, 1),
        "exact_order": total_exact_order / max(total, 1),
        "exact_set": total_exact_set / max(total, 1),
    }


@torch.inference_mode()
def print_samples(
    model,
    dataset,
    training_rows,
    model_id_to_fragment_id,
    id_to_text,
    device,
    count=5,
):
    model.eval()

    indices = random.sample(range(len(dataset)), min(count, len(dataset)))

    print()
    print("SAMPLES")
    print("=" * 60)

    for idx in indices:
        item = dataset[idx]
        x = item["x"].unsqueeze(0).to(device)
        row_index = int(item["row_index"])
        row = training_rows[row_index]

        pred_seq = model.generate(x, max_len=dataset.max_output_len)[0].cpu().tolist()
        gold_seq = item["seq"].cpu().tolist()

        pred_ids = [
            model_id_to_fragment_id[x]
            for x in strip_special(pred_seq)
            if x in model_id_to_fragment_id
        ]

        gold_ids = [
            model_id_to_fragment_id[x]
            for x in strip_special(gold_seq)
            if x in model_id_to_fragment_id
        ]

        pred_text = assemble_fragments(pred_ids, id_to_text)
        gold_text = assemble_fragments(gold_ids, id_to_text)

        print("INPUT:", row.get("input"))
        print("PRED IDS:", pred_ids)
        print("PRED:", pred_text)
        print("GOLD IDS:", gold_ids)
        print("GOLD:", gold_text)
        print("-" * 60)


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fragments", default=str(DEFAULT_FRAGMENTS_PATH))
    parser.add_argument("--training", default=str(DEFAULT_TRAINING_PATH))
    parser.add_argument("--out", default=str(DEFAULT_OUT_PATH))

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--max-output-len", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-split", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=1)

    args = parser.parse_args()

    set_seed(args.seed)

    fragments_path = Path(args.fragments)
    training_path = Path(args.training)
    out_path = Path(args.out)

    print("Fragments:", fragments_path)
    print("Training:", training_path)
    print("Out:", out_path)
    print()

    fragments = load_jsonl(fragments_path)
    training_rows = load_jsonl(training_path)

    print("Fragment rows:", len(fragments))
    print("Training rows:", len(training_rows))

    fragment_ids = []
    id_to_text = {}

    for row in fragments:
        fid = clean_text(row.get("id"))
        text = clean_text(row.get("text"))

        if not fid or not text:
            continue

        if fid in id_to_text:
            continue

        fragment_ids.append(fid)
        id_to_text[fid] = text

    fragment_id_to_model_id = {}
    model_id_to_fragment_id = {}

    next_id = 3

    for fid in fragment_ids:
        fragment_id_to_model_id[fid] = next_id
        model_id_to_fragment_id[next_id] = fid
        next_id += 1

    vocab_size = next_id

    print("Usable fragment IDs:", len(fragment_ids))
    print("Model vocab size:", vocab_size)
    print()

    print("Loading embedder:", EMBEDDING_MODEL_NAME)
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    inputs = [clean_text(row.get("input")) for row in training_rows]

    print("Embedding inputs...")
    input_embeddings = embedder.encode(
        inputs,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=128,
        show_progress_bar=True,
    ).astype("float32")

    dataset = TokenFragmentDataset(
        training_rows=training_rows,
        fragment_id_to_model_id=fragment_id_to_model_id,
        input_embeddings=input_embeddings,
        max_output_len=args.max_output_len,
    )

    if len(dataset) < 10:
        raise RuntimeError("Not enough training rows.")

    val_size = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print()

    model = TokenFragmentSelector(
        input_dim=input_embeddings.shape[1],
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        max_output_len=args.max_output_len,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
        )

        metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
        )

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"val_f1={metrics['f1']:.4f} "
            f"exact_order={metrics['exact_order']:.4f} "
            f"exact_set={metrics['exact_set']:.4f}"
        )

        if epoch % args.log_every == 0:
            print_samples(
                model=model,
                dataset=dataset,
                training_rows=training_rows,
                model_id_to_fragment_id=model_id_to_fragment_id,
                id_to_text=id_to_text,
                device=device,
                count=5,
            )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "fragment_id_to_model_id": fragment_id_to_model_id,
                "model_id_to_fragment_id": model_id_to_fragment_id,
                "id_to_text": id_to_text,
                "special_tokens": SPECIAL_TOKENS,
                "embedding_model": EMBEDDING_MODEL_NAME,
                "input_dim": int(input_embeddings.shape[1]),
                "hidden_dim": int(args.hidden_dim),
                "max_output_len": int(args.max_output_len),
                "vocab_size": int(vocab_size),
                "best_f1": float(best_f1),
                "fragments_path": str(fragments_path),
                "training_path": str(training_path),
            }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if best_state is None:
        best_state = {
            "model_state_dict": model.state_dict(),
            "fragment_id_to_model_id": fragment_id_to_model_id,
            "model_id_to_fragment_id": model_id_to_fragment_id,
            "id_to_text": id_to_text,
            "special_tokens": SPECIAL_TOKENS,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "input_dim": int(input_embeddings.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "max_output_len": int(args.max_output_len),
            "vocab_size": int(vocab_size),
            "best_f1": float(best_f1),
            "fragments_path": str(fragments_path),
            "training_path": str(training_path),
        }

    torch.save(best_state, out_path)

    print()
    print("DONE")
    print("Saved:", out_path)
    print("Best F1:", round(best_f1, 4))


if __name__ == "__main__":
    main()
