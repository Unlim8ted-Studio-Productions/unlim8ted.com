import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ============================================================
# CONFIG
# ============================================================

MODEL_DIR = Path("assets/models/category_router")
MODEL_PT = MODEL_DIR / "category_router.pt"

DEFAULT_THRESHOLD = 0.55
DEFAULT_TOP_K = 8
DEFAULT_MARGIN = 0.18

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# TEXT PROCESSING
# ============================================================


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
# MODEL
# ============================================================


class CategoryRouter(nn.Module):
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
    labels = checkpoint["labels"]
    config = checkpoint["config"]

    model = CategoryRouter(
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

    return model, vocab, labels, threshold, top_k, margin


# ============================================================
# PREDICTION
# ============================================================


def select_categories(ranked, threshold: float, top_k: int, margin: float):
    """
    Stricter selection:
    - always keep the best label
    - include extra labels only if:
      score >= threshold
      and score is close enough to best score
    """

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


def predict(text: str, model, vocab, labels, threshold, top_k, margin):
    x = vectorize_text(text, vocab)
    tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()

    ranked = sorted(
        [
            {
                "category": labels[i],
                "score": float(probs[i]),
            }
            for i in range(len(labels))
        ],
        key=lambda item: item["score"],
        reverse=True,
    )

    selected = select_categories(
        ranked=ranked,
        threshold=threshold,
        top_k=top_k,
        margin=margin,
    )

    return selected, ranked[:top_k]


# ============================================================
# MAIN
# ============================================================


def main():
    model, vocab, labels, threshold, top_k, margin = load_model()

    print("Category router loaded.")
    print(f"Device: {DEVICE}")
    print(f"Labels: {len(labels)}")
    print(f"Vocab: {len(vocab)}")
    print(f"Threshold: {threshold}")
    print(f"Top K: {top_k}")
    print(f"Margin: {margin}")
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
            labels=labels,
            threshold=threshold,
            top_k=top_k,
            margin=margin,
        )

        print()
        print("Selected categories:")
        for item in selected:
            print(f"- {item['category']}  {item['score']:.3f}")

        print()
        print("Top scores:")
        for item in top:
            print(f"- {item['category']}  {item['score']:.3f}")

        print()


if __name__ == "__main__":
    main()
