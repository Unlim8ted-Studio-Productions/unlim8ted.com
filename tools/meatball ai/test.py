import re
import sys
import traceback
from pathlib import Path

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

print("TEST.PY STARTED", flush=True)

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

MODEL_PATH = Path("assets/models/token-fragment-selector.pt")


def clean_text(x):
    x = str(x or "").strip()
    x = re.sub(r"\s+", " ", x)
    return x


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


class TokenFragmentSelector(nn.Module):
    def __init__(
        self, input_dim, vocab_size, hidden_dim=384, max_output_len=16, dropout=0.0
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
            dropout=0.0,
            batch_first=True,
        )

        self.output = nn.Linear(hidden_dim, vocab_size)

    @torch.inference_mode()
    def generate(self, x, max_len):
        self.eval()

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

            logits[:, PAD_ID] = -1e9
            logits[:, BOS_ID] = -1e9

            next_id = torch.argmax(logits, dim=-1)

            generated.append(next_id)
            current = next_id.unsqueeze(1)

            if torch.all(next_id == EOS_ID):
                break

        if not generated:
            return torch.empty((batch_size, 0), dtype=torch.long, device=device)

        return torch.stack(generated, dim=1)


def load_checkpoint():
    print("Checking model path:", MODEL_PATH.resolve(), flush=True)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"MODEL DOES NOT EXIST: {MODEL_PATH.resolve()}")

    print("Loading checkpoint...", flush=True)

    try:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    print("Checkpoint loaded.", flush=True)
    print("Checkpoint keys:", list(checkpoint.keys()), flush=True)

    return checkpoint


def fix_int_key_dict(d):
    out = {}

    for k, v in d.items():
        try:
            out[int(k)] = v
        except Exception:
            out[k] = v

    return out


def main():
    print("MAIN STARTED", flush=True)

    checkpoint = load_checkpoint()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, flush=True)

    input_dim = int(checkpoint["input_dim"])
    vocab_size = int(checkpoint["vocab_size"])
    hidden_dim = int(checkpoint["hidden_dim"])
    max_output_len = int(checkpoint["max_output_len"])

    print("input_dim:", input_dim, flush=True)
    print("vocab_size:", vocab_size, flush=True)
    print("hidden_dim:", hidden_dim, flush=True)
    print("max_output_len:", max_output_len, flush=True)

    model = TokenFragmentSelector(
        input_dim=input_dim,
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        max_output_len=max_output_len,
        dropout=0.0,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded.", flush=True)

    embedding_model_name = checkpoint.get(
        "embedding_model",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    print("Loading embedder:", embedding_model_name, flush=True)
    embedder = SentenceTransformer(embedding_model_name)
    print("Embedder loaded.", flush=True)

    model_id_to_fragment_id = fix_int_key_dict(checkpoint["model_id_to_fragment_id"])
    id_to_text = checkpoint["id_to_text"]

    print()
    print("READY. Type a question. Type quit to exit.", flush=True)
    print()

    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() in {"quit", "exit", "/quit", "/exit"}:
            print("bye")
            break

        emb = embedder.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        x = torch.tensor(emb, dtype=torch.float32).to(device)

        pred = model.generate(x, max_len=max_output_len)[0].detach().cpu().tolist()
        token_ids = strip_special(pred)

        fragment_ids = []

        for token_id in token_ids:
            fid = model_id_to_fragment_id.get(int(token_id))

            if fid:
                fragment_ids.append(fid)

        answer = assemble_fragments(fragment_ids, id_to_text)

        if not answer:
            answer = "I do not know."

        print()
        print("Bot:", answer)
        print("Fragment IDs:", fragment_ids)
        print("Token IDs:", token_ids)
        print()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print()
        print("SCRIPT CRASHED:", flush=True)
        traceback.print_exc()
        input("Press Enter to close...")
        sys.exit(1)
