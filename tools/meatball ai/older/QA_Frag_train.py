# python "tools/meatball ai/QA_Frag_train.py" --index "assets\models\fragment-embeddings\fragment_index.json" --embeddings "assets\models\fragment-embeddings\fragment_embeddings.npy" --qa "assets\data\Smart-Meatball-Data.jsonl" --out "assets\data\fragment_training_from_qa_local_test.jsonl" --target-rows 200 --labeler-model "Qwen/Qwen3-1.7B" --batch-size 1 --top-k-question 6 --top-k-answer 10 --max-candidates 14 --max-bridge-per-topic 2 --max-new-tokens 700 --print-prompt-chars
import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sentence_transformers import SentenceTransformer

# ============================================================
# BASIC IO
# ============================================================


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))


def load_jsonl(path, limit=None):
    rows = []

    with Path(path).open("r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                print(f"Bad JSON skipped at {path}:{line_num}")
                continue

            if isinstance(obj, dict):
                rows.append(obj)

            if limit and len(rows) >= limit:
                break

    return rows


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def clean_text(x):
    x = str(x or "").strip()
    x = re.sub(r"\s+", " ", x)
    return x


def load_embeddings(path, quantized=False):
    emb = np.load(path)

    if quantized or emb.dtype == np.int8:
        emb = emb.astype("float32") / 127.0
    else:
        emb = emb.astype("float32")

    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0

    return emb / norms


def assemble_preview(ids, id_to_text):
    parts = []

    for fid in ids:
        text = clean_text(id_to_text.get(fid, ""))
        if text:
            parts.append(text)

    out = ""

    for part in parts:
        if not out:
            out = part
        elif out.endswith((" ", "\n")):
            out += part
        elif part.startswith((".", ",", "!", "?", ";", ":")):
            out += part
        else:
            out += " " + part

    return out


# ============================================================
# INDEX / EMBEDDING SETUP
# ============================================================


def build_fragment_maps(index, semantic_embeddings, embedder, device_for_embed="cpu"):
    fragments = index.get("fragments", [])
    bridges = index.get("bridges", [])

    if len(fragments) != semantic_embeddings.shape[0]:
        raise ValueError(
            f"Fragment count mismatch: {len(fragments)} fragments vs {semantic_embeddings.shape[0]} embedding rows"
        )

    id_to_text = {}
    id_to_kind = {}
    id_to_emb = {}

    for i, f in enumerate(fragments):
        fid = f.get("id")
        if not fid:
            continue

        id_to_text[fid] = clean_text(f.get("text", ""))
        id_to_kind[fid] = "semantic"
        id_to_emb[fid] = semantic_embeddings[i].astype("float32")

    bridge_texts = []
    bridge_ids = []

    for b in bridges:
        bid = b.get("id")
        if not bid:
            continue

        bridge_ids.append(bid)
        bridge_texts.append(clean_text(b.get("text", "")))
        id_to_text[bid] = clean_text(b.get("text", ""))
        id_to_kind[bid] = "bridge"

    print("Embedding bridge fragments:", len(bridge_ids))

    if bridge_ids:
        bridge_emb = embedder.encode(
            bridge_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=128,
            show_progress_bar=True,
        ).astype("float32")

        for bid, vec in zip(bridge_ids, bridge_emb):
            id_to_emb[bid] = vec

    all_bridge_ids = bridge_ids

    return id_to_text, id_to_kind, id_to_emb, all_bridge_ids


# ============================================================
# TRAINING DATASET
# ============================================================


def get_target_ids(row):
    target = row.get("target", {})
    ids = target.get("fragments", [])

    if not isinstance(ids, list):
        return []

    return [x for x in ids if isinstance(x, str)]


def get_question(row):
    return clean_text(row.get("input") or row.get("question"))


def get_candidate_ids(row, all_bridge_ids, use_all_bridges=True):
    semantic_ids = row.get("semantic_fragment_ids", [])
    if not isinstance(semantic_ids, list):
        semantic_ids = []

    semantic_ids = [x for x in semantic_ids if isinstance(x, str)]

    if use_all_bridges:
        bridge_ids = list(all_bridge_ids)
    else:
        bridge_ids = row.get("bridge_fragment_ids", [])
        if not isinstance(bridge_ids, list):
            bridge_ids = []
        bridge_ids = [x for x in bridge_ids if isinstance(x, str)]

    out = []
    seen = set()

    for fid in semantic_ids + bridge_ids:
        if fid in seen:
            continue
        seen.add(fid)
        out.append(fid)

    return out


def filter_rows(
    raw_rows, all_bridge_ids, id_to_emb, use_all_bridges=True, max_output_len=8
):
    good = []
    skipped = {
        "missing_question": 0,
        "missing_target": 0,
        "missing_candidate": 0,
        "target_not_in_candidates": 0,
        "too_long": 0,
    }

    for row in raw_rows:
        q = get_question(row)
        target_ids = get_target_ids(row)

        if not q:
            skipped["missing_question"] += 1
            continue

        if not target_ids:
            skipped["missing_target"] += 1
            continue

        if len(target_ids) > max_output_len:
            skipped["too_long"] += 1
            continue

        candidate_ids = get_candidate_ids(
            row,
            all_bridge_ids=all_bridge_ids,
            use_all_bridges=use_all_bridges,
        )

        candidate_ids = [fid for fid in candidate_ids if fid in id_to_emb]

        if not candidate_ids:
            skipped["missing_candidate"] += 1
            continue

        cset = set(candidate_ids)

        if any(fid not in cset for fid in target_ids):
            skipped["target_not_in_candidates"] += 1
            continue

        good.append(
            {
                "question": q,
                "official_answer": row.get("official_answer", ""),
                "candidate_ids": candidate_ids,
                "target_ids": target_ids,
                "shape": row.get("target", {}).get("shape", "direct_answer"),
                "assembled_preview": row.get("assembled_preview", ""),
            }
        )

    print("Usable training rows:", len(good))
    print("Skipped:", skipped)

    if not good:
        raise ValueError("No usable rows after filtering.")

    return good


def build_question_cache(rows, embedder, cache_path=None):
    unique_questions = sorted(set(r["question"] for r in rows))

    if cache_path and Path(cache_path).exists():
        print("Loading question embedding cache:", cache_path)
        data = np.load(cache_path, allow_pickle=True)
        questions = data["questions"].tolist()
        embeddings = data["embeddings"].astype("float32")
        cache = {q: embeddings[i] for i, q in enumerate(questions)}

        missing = [q for q in unique_questions if q not in cache]

        if not missing:
            return cache

        print("Cache missing questions:", len(missing))
    else:
        cache = {}
        missing = unique_questions

    print("Embedding unique questions:", len(missing))

    if missing:
        new_emb = embedder.encode(
            missing,
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=128,
            show_progress_bar=True,
        ).astype("float32")

        for q, vec in zip(missing, new_emb):
            cache[q] = vec

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        qs = sorted(cache.keys())
        embs = np.stack([cache[q] for q in qs]).astype("float32")
        np.savez_compressed(
            cache_path, questions=np.array(qs, dtype=object), embeddings=embs
        )
        print("Saved question embedding cache:", cache_path)

    return cache


class FragmentSelectorDataset(Dataset):
    def __init__(self, rows, q_cache, id_to_emb, max_output_len):
        self.rows = rows
        self.q_cache = q_cache
        self.id_to_emb = id_to_emb
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        q_emb = self.q_cache[row["question"]].astype("float32")

        candidate_ids = row["candidate_ids"]
        candidate_embs = np.stack(
            [self.id_to_emb[fid] for fid in candidate_ids]
        ).astype("float32")

        id_to_pos = {fid: i for i, fid in enumerate(candidate_ids)}

        # Targets are position indexes into candidate_ids.
        # The stop token is handled in collate using stop_index=max_candidates.
        target_positions = [id_to_pos[fid] for fid in row["target_ids"]]

        return {
            "question": row["question"],
            "candidate_ids": candidate_ids,
            "target_ids": row["target_ids"],
            "q_emb": q_emb,
            "candidate_embs": candidate_embs,
            "target_positions": target_positions,
        }


def collate_batch(batch, max_output_len):
    bsz = len(batch)
    emb_dim = batch[0]["q_emb"].shape[0]
    max_candidates = max(x["candidate_embs"].shape[0] for x in batch)

    q_emb = torch.zeros((bsz, emb_dim), dtype=torch.float32)
    cand_emb = torch.zeros((bsz, max_candidates, emb_dim), dtype=torch.float32)
    cand_mask = torch.zeros((bsz, max_candidates), dtype=torch.bool)

    # Output positions include one STOP after the true fragment list.
    # Unused positions are ignored in loss.
    target = torch.full((bsz, max_output_len + 1), -100, dtype=torch.long)

    candidate_ids = []
    target_ids = []
    questions = []

    for i, item in enumerate(batch):
        q = torch.from_numpy(item["q_emb"])
        c = torch.from_numpy(item["candidate_embs"])
        n = c.shape[0]

        q_emb[i] = q
        cand_emb[i, :n] = c
        cand_mask[i, :n] = True

        t = item["target_positions"][:max_output_len]

        for p, pos in enumerate(t):
            target[i, p] = int(pos)

        # STOP token class is always max_candidates.
        stop_pos = len(t)
        if stop_pos <= max_output_len:
            target[i, stop_pos] = max_candidates

        candidate_ids.append(item["candidate_ids"])
        target_ids.append(item["target_ids"])
        questions.append(item["question"])

    return {
        "q_emb": q_emb,
        "cand_emb": cand_emb,
        "cand_mask": cand_mask,
        "target": target,
        "candidate_ids": candidate_ids,
        "target_ids": target_ids,
        "questions": questions,
    }


# ============================================================
# MODEL
# ============================================================


class FragmentSelector(nn.Module):
    def __init__(self, emb_dim, hidden_dim=384, max_output_len=8, dropout=0.1):
        super().__init__()

        self.max_output_len = max_output_len

        self.q_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.c_proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.pos_emb = nn.Parameter(torch.randn(max_output_len + 1, hidden_dim) * 0.02)
        self.stop_emb = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        self.scale = math.sqrt(hidden_dim)

    def forward(self, q_emb, cand_emb, cand_mask):
        """
        q_emb: [B, D]
        cand_emb: [B, N, D]
        cand_mask: [B, N]
        returns logits: [B, P, N+1]
        """
        bsz, n, _ = cand_emb.shape
        p = self.max_output_len + 1

        qh = self.q_proj(q_emb)  # [B, H]
        ch = self.c_proj(cand_emb)  # [B, N, H]

        queries = qh[:, None, :] + self.pos_emb[None, :, :]  # [B, P, H]

        cand_logits = torch.einsum("bph,bnh->bpn", queries, ch) / self.scale

        # Mask padded candidates.
        cand_logits = cand_logits.masked_fill(~cand_mask[:, None, :], -1e9)

        stop_vec = self.stop_emb[None, None, :]  # [1, 1, H]
        stop_logits = torch.sum(queries * stop_vec, dim=-1, keepdim=True) / self.scale

        logits = torch.cat([cand_logits, stop_logits], dim=-1)  # [B, P, N+1]
        return logits


# ============================================================
# TRAIN / EVAL
# ============================================================


def decode_predictions(logits, candidate_ids, max_output_len):
    """
    logits: [B, P, N+1]
    candidate_ids: list[list[str]]
    """
    preds = []

    best = torch.argmax(logits, dim=-1).detach().cpu().numpy()

    for i, ids in enumerate(candidate_ids):
        n = len(ids)
        out = []
        seen = set()

        for p in range(max_output_len + 1):
            idx = int(best[i, p])

            # Stop token.
            if idx >= n:
                break

            fid = ids[idx]

            # Avoid repeated IDs in final output.
            if fid in seen:
                continue

            seen.add(fid)
            out.append(fid)

            if len(out) >= max_output_len:
                break

        preds.append(out)

    return preds


def compute_metrics(preds, targets):
    exact_order = 0
    exact_set = 0
    total = len(preds)

    f1_sum = 0.0

    for pred, tgt in zip(preds, targets):
        if pred == tgt:
            exact_order += 1

        if set(pred) == set(tgt):
            exact_set += 1

        ps = set(pred)
        ts = set(tgt)

        if not ps and not ts:
            f1 = 1.0
        elif not ps or not ts:
            f1 = 0.0
        else:
            precision = len(ps & ts) / max(len(ps), 1)
            recall = len(ps & ts) / max(len(ts), 1)
            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * precision * recall / (precision + recall)

        f1_sum += f1

    return {
        "exact_order": exact_order / max(total, 1),
        "exact_set": exact_set / max(total, 1),
        "fragment_f1": f1_sum / max(total, 1),
    }


def run_epoch(
    model, loader, optimizer, device, max_output_len, train=True, log_every=50
):
    if train:
        model.train()
    else:
        model.eval()

    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    total_loss = 0.0
    total_batches = 0
    total_rows = 0

    all_preds = []
    all_targets = []

    start = time.time()

    for step, batch in enumerate(loader, 1):
        q_emb = batch["q_emb"].to(device)
        cand_emb = batch["cand_emb"].to(device)
        cand_mask = batch["cand_mask"].to(device)
        target = batch["target"].to(device)

        with torch.set_grad_enabled(train):
            logits = model(q_emb, cand_emb, cand_mask)

            loss = loss_fn(
                logits.reshape(-1, logits.shape[-1]),
                target.reshape(-1),
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1
        total_rows += q_emb.shape[0]

        preds = decode_predictions(
            logits=logits,
            candidate_ids=batch["candidate_ids"],
            max_output_len=max_output_len,
        )

        all_preds.extend(preds)
        all_targets.extend(batch["target_ids"])

        if train and step % log_every == 0:
            elapsed = max(time.time() - start, 0.001)
            rows_per_sec = total_rows / elapsed
            rows_per_min = rows_per_sec * 60.0
            avg_loss = total_loss / max(total_batches, 1)
            metrics = compute_metrics(all_preds[-500:], all_targets[-500:])

            print(
                f"TRAIN step={step} "
                f"loss={avg_loss:.4f} "
                f"rows={total_rows} "
                f"rows/sec={rows_per_sec:.2f} "
                f"rows/min={rows_per_min:.1f} "
                f"exact_order_recent={metrics['exact_order']:.3f} "
                f"exact_set_recent={metrics['exact_set']:.3f} "
                f"f1_recent={metrics['fragment_f1']:.3f}",
                flush=True,
            )

    elapsed = max(time.time() - start, 0.001)
    metrics = compute_metrics(all_preds, all_targets)

    return {
        "loss": total_loss / max(total_batches, 1),
        "rows": total_rows,
        "seconds": elapsed,
        "rows_per_sec": total_rows / elapsed,
        "rows_per_min": (total_rows / elapsed) * 60.0,
        **metrics,
    }


def show_samples(model, dataset, id_to_text, device, max_output_len, count=5):
    model.eval()

    indexes = random.sample(range(len(dataset)), min(count, len(dataset)))

    print()
    print("SAMPLES")
    print("=" * 80)

    for idx in indexes:
        item = dataset[idx]
        batch = collate_batch([item], max_output_len=max_output_len)

        q_emb = batch["q_emb"].to(device)
        cand_emb = batch["cand_emb"].to(device)
        cand_mask = batch["cand_mask"].to(device)

        with torch.no_grad():
            logits = model(q_emb, cand_emb, cand_mask)

        pred_ids = decode_predictions(
            logits,
            batch["candidate_ids"],
            max_output_len=max_output_len,
        )[0]

        true_ids = batch["target_ids"][0]

        print("QUESTION:", batch["questions"][0])
        print("TRUE IDS:", true_ids)
        print("PRED IDS:", pred_ids)
        print("TRUE TEXT:", assemble_preview(true_ids, id_to_text))
        print("PRED TEXT:", assemble_preview(pred_ids, id_to_text))
        print("-" * 80)


# ============================================================
# SAVE / LOAD EXPORT
# ============================================================


def save_checkpoint(path, model, config, id_to_text):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "id_to_text": id_to_text,
        },
        path,
    )

    print("Saved checkpoint:", path)


# ============================================================
# MAIN
# ============================================================


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--training", default="assets/data/fragment_training_from_qa_local.jsonl"
    )
    parser.add_argument(
        "--index", default="assets/models/fragment-embeddings/fragment_index.json"
    )
    parser.add_argument(
        "--embeddings",
        default="assets/models/fragment-embeddings/fragment_embeddings.npy",
    )
    parser.add_argument("--out", default="assets/models/meatball-fragment-selector.pt")
    parser.add_argument(
        "--cache", default="assets/models/meatball-fragment-selector-question-cache.npz"
    )

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--quantized", action="store_true")

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--max-output-len", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--val-split", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)

    parser.add_argument("--bridges-from-row", action="store_true")
    parser.add_argument("--samples", type=int, default=5)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("Loading index...")
    index = load_json(args.index)

    print("Loading fragment embeddings...")
    semantic_embeddings = load_embeddings(args.embeddings, args.quantized)

    print("Loading embedder once...")
    embedder = SentenceTransformer(index["embedding_model"])

    id_to_text, id_to_kind, id_to_emb, all_bridge_ids = build_fragment_maps(
        index=index,
        semantic_embeddings=semantic_embeddings,
        embedder=embedder,
    )

    print("All bridge fragments:", len(all_bridge_ids))

    print("Loading labeled training rows...")
    raw_rows = load_jsonl(args.training, limit=args.limit if args.limit > 0 else None)
    print("Raw rows:", len(raw_rows))

    rows = filter_rows(
        raw_rows=raw_rows,
        all_bridge_ids=all_bridge_ids,
        id_to_emb=id_to_emb,
        use_all_bridges=not args.bridges_from_row,
        max_output_len=args.max_output_len,
    )

    print("Building question embedding cache...")
    q_cache = build_question_cache(
        rows,
        embedder=embedder,
        cache_path=args.cache,
    )

    full_dataset = FragmentSelectorDataset(
        rows=rows,
        q_cache=q_cache,
        id_to_emb=id_to_emb,
        max_output_len=args.max_output_len,
    )

    val_size = max(1, int(len(full_dataset) * args.val_split))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    print("Train rows:", len(train_dataset))
    print("Val rows:", len(val_dataset))

    collate = lambda batch: collate_batch(batch, max_output_len=args.max_output_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=(device == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=(device == "cuda"),
    )

    emb_dim = next(iter(id_to_emb.values())).shape[0]

    model = FragmentSelector(
        emb_dim=emb_dim,
        hidden_dim=args.hidden_dim,
        max_output_len=args.max_output_len,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    config = {
        "embedding_model": index["embedding_model"],
        "emb_dim": emb_dim,
        "hidden_dim": args.hidden_dim,
        "max_output_len": args.max_output_len,
        "dropout": args.dropout,
        "uses_all_bridges": not args.bridges_from_row,
        "all_bridge_ids": all_bridge_ids,
    }

    print()
    print("MODEL PARAMS:", sum(p.numel() for p in model.parameters()))
    print("This is the neural-net training step. Loss should print now.")
    print()

    best_val = float("inf")

    global_start = time.time()

    for epoch in range(1, args.epochs + 1):
        print()
        print("=" * 80)
        print(f"EPOCH {epoch}/{args.epochs}")
        print("=" * 80)

        train_stats = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            max_output_len=args.max_output_len,
            train=True,
            log_every=args.log_every,
        )

        val_stats = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            device=device,
            max_output_len=args.max_output_len,
            train=False,
            log_every=args.log_every,
        )

        print(
            f"EPOCH_DONE epoch={epoch} "
            f"train_loss={train_stats['loss']:.4f} "
            f"val_loss={val_stats['loss']:.4f} "
            f"train_rows/min={train_stats['rows_per_min']:.1f} "
            f"val_rows/min={val_stats['rows_per_min']:.1f} "
            f"val_exact_order={val_stats['exact_order']:.3f} "
            f"val_exact_set={val_stats['exact_set']:.3f} "
            f"val_f1={val_stats['fragment_f1']:.3f}",
            flush=True,
        )

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(args.out, model, config, id_to_text)

        show_samples(
            model=model,
            dataset=full_dataset,
            id_to_text=id_to_text,
            device=device,
            max_output_len=args.max_output_len,
            count=args.samples,
        )

    total_time = time.time() - global_start

    print()
    print("TRAINING COMPLETE")
    print("Seconds:", round(total_time, 2))
    print("Best val loss:", round(best_val, 4))
    print("Saved best model:", args.out)


if __name__ == "__main__":
    main()
