import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = Path("assets/models/fragment_assembler")
FRAGMENTS_DIR = Path("assets/data/fragments")

MODEL_PT = MODEL_DIR / "fragment_assembler.pt"

DEFAULT_THRESHOLD = 0.50
DEFAULT_TOP_K = 8
DEFAULT_MARGIN = 0.22

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# TEXT
# ============================================================


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize(text: str):
    text = normalize_text(text)
    words = re.findall(r"[a-z0-9']+", text)

    tokens = []

    for word in words:
        tokens.append(word)

    for i in range(len(words) - 1):
        tokens.append(words[i] + "_" + words[i + 1])

    for i in range(len(words) - 2):
        tokens.append(words[i] + "_" + words[i + 1] + "_" + words[i + 2])

    return tokens


def vectorize_text(text: str, vocab: dict):
    x = np.zeros(len(vocab), dtype=np.float32)

    for token in tokenize(text):
        idx = vocab.get(token)

        if idx is not None:
            x[idx] += 1.0

    return np.log1p(x)


# ============================================================
# FRAGMENTS
# ============================================================


def load_all_fragments():
    fragment_text_by_id = {}

    for path in FRAGMENTS_DIR.glob("*.jsonl"):
        with path.open("r", encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue

                frag_id = str(row.get("id", "")).strip()
                text = str(row.get("text", row.get("content", ""))).strip()

                if frag_id and text:
                    fragment_text_by_id[frag_id] = text

    return fragment_text_by_id


# ============================================================
# MODEL
# ============================================================


class FragmentAssembler(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.net(x)


def load_model():
    if not MODEL_PT.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PT}")

    checkpoint = torch.load(MODEL_PT, map_location=DEVICE)

    vocab = checkpoint["vocab"]
    fragment_ids = checkpoint["fragment_ids"]
    config = checkpoint["config"]

    model = FragmentAssembler(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        output_size=config["output_size"],
        dropout=config["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    threshold = config.get("threshold", DEFAULT_THRESHOLD)
    top_k = config.get("top_k", DEFAULT_TOP_K)
    margin = config.get("margin", DEFAULT_MARGIN)
    unknown_id = config.get("unknown_id", "i_cant_i_dont_know")

    return model, vocab, fragment_ids, threshold, top_k, margin, unknown_id


# ============================================================
# PREDICTION
# ============================================================


def select_fragments(ranked, threshold: float, top_k: int, margin: float):
    if not ranked:
        return []

    best = ranked[0]
    best_score = best["score"]

    selected = [best]

    for item in ranked[1:top_k]:
        score = item["score"]

        if score >= threshold and score >= best_score - margin:
            selected.append(item)

    return selected


def predict(text: str, model, vocab, fragment_ids, threshold, top_k, margin):
    x = vectorize_text(text, vocab)
    tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    ranked = sorted(
        [
            {
                "fragment_id": fragment_ids[i],
                "score": float(probs[i]),
            }
            for i in range(len(fragment_ids))
        ],
        key=lambda item: item["score"],
        reverse=True,
    )

    selected = select_fragments(
        ranked=ranked,
        threshold=threshold,
        top_k=top_k,
        margin=margin,
    )

    return selected, ranked[:top_k]


def assemble_answer(selected, fragment_text_by_id, unknown_id):
    ids = [item["fragment_id"] for item in selected]

    if unknown_id in ids:
        return unknown_id

    parts = []

    for item in selected:
        frag_id = item["fragment_id"]
        text = fragment_text_by_id.get(frag_id, "")

        if text:
            parts.append(text)

    if not parts:
        return unknown_id

    return " ".join(parts)


# ============================================================
# MAIN
# ============================================================


def main():
    model, vocab, fragment_ids, threshold, top_k, margin, unknown_id = load_model()
    fragment_text_by_id = load_all_fragments()

    print("Fragment assembler loaded.")
    print(f"Device: {DEVICE}")
    print(f"Fragment IDs: {len(fragment_ids)}")
    print(f"Vocab: {len(vocab)}")
    print(f"Threshold: {threshold}")
    print(f"Top K: {top_k}")
    print(f"Margin: {margin}")
    print(f"Unknown ID: {unknown_id}")
    print()
    print("Type a message. Type q/quit/exit to stop.")
    print()

    while True:
        text = input("You: ").strip()

        if text.lower() in {"q", "quit", "exit"}:
            break

        if not text:
            continue

        selected, top = predict(
            text=text,
            model=model,
            vocab=vocab,
            fragment_ids=fragment_ids,
            threshold=threshold,
            top_k=top_k,
            margin=margin,
        )

        answer = assemble_answer(
            selected=selected,
            fragment_text_by_id=fragment_text_by_id,
            unknown_id=unknown_id,
        )

        print()
        print("Selected fragments:")
        for item in selected:
            frag_id = item["fragment_id"]
            score = item["score"]
            frag_text = fragment_text_by_id.get(frag_id, "")
            print(f"- {frag_id}  {score:.3f}")
            if frag_text:
                print(f"  {frag_text}")

        print()
        print("Assembled answer:")
        print(answer)

        print()
        print("Top scores:")
        for item in top:
            frag_id = item["fragment_id"]
            score = item["score"]
            frag_text = fragment_text_by_id.get(frag_id, "")
            print(f"- {frag_id}  {score:.3f}")
            if frag_text:
                print(f"  {frag_text[:160]}")

        print()


if __name__ == "__main__":
    main()
