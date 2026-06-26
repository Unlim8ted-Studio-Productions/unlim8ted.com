# test_general_cover_chunks_pt.py
#
# Test a PyTorch .pt checkpoint from the general cover chunk model.
#
# Usage:
# python test_general_cover_chunks_pt.py
#
# Or:
# python test_general_cover_chunks_pt.py --model_dir assets/models/general_cover_chunks_noisy_continue
#
# Or:
# python test_general_cover_chunks_pt.py --question "what is timecat"

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

DEFAULT_MODEL_DIR = Path("assets/models/general_cover_chunks_noisy_continue")

INPUT_NGRAMS = (1, 2, 3)

PROMPT_SIZE = 128
HIDDEN_SIZE = 192
EMBED_SIZE = 128
DROPOUT = 0.35

MAX_OUTPUT_CHUNKS = 24

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


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


def vectorize_question(question, input_vocab):
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


def decode_ids(ids, output_chunks):
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
            self.decoder_cell = ManualGRUCell(
                PROMPT_SIZE + EMBED_SIZE,
                HIDDEN_SIZE,
            )
        else:
            self.decoder_cell = nn.GRUCell(
                PROMPT_SIZE + EMBED_SIZE,
                HIDDEN_SIZE,
            )

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
            torch.cat([emb, prompt_context], dim=-1),
            write_hidden,
        )

        logits = self.output(torch.cat([prompt_context, write_hidden], dim=-1))

        return logits, write_hidden

    def forward(self, x, max_len=MAX_OUTPUT_CHUNKS + 1):
        batch_size = x.size(0)
        prompt_context = self.encode(x)

        write_hidden = torch.zeros(
            batch_size,
            HIDDEN_SIZE,
            device=x.device,
        )

        prev_token = torch.full(
            (batch_size,),
            BOS_ID,
            dtype=torch.long,
            device=x.device,
        )

        logits_steps = []

        for _ in range(max_len):
            logits, write_hidden = self.decoder_step(
                prev_token,
                prompt_context,
                write_hidden,
            )

            logits_steps.append(logits.unsqueeze(1))
            prev_token = torch.argmax(logits, dim=-1)

        return torch.cat(logits_steps, dim=1)


def load_runtime(model_dir):
    model_dir = Path(model_dir)

    input_vocab = load_json(model_dir / "input_vocab.json")
    output_chunks = load_json(model_dir / "output_chunks.json")

    checkpoint_path = model_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    model = ChunkAnswerModel(
        input_size=len(input_vocab),
        output_vocab_size=len(output_chunks),
        manual_gru=False,
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    return model, input_vocab, output_chunks


def predict(question, model, input_vocab, output_chunks, show_chunks=False):
    x = vectorize_question(question, input_vocab).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        ids = torch.argmax(logits[0], dim=-1).detach().cpu().tolist()

    answer = decode_ids(ids, output_chunks)

    print()
    print("=" * 80)
    print("QUESTION:")
    print(question)
    print()
    print("ANSWER:")
    print(answer)

    if show_chunks:
        print()
        print("CHUNKS:")
        for idx in ids:
            idx = int(idx)

            if idx == EOS_ID:
                print(f"{idx:6d}  <EOS>")
                break

            if idx in (PAD_ID, BOS_ID, UNK_ID):
                continue

            text = (
                output_chunks[idx]["text"] if 0 <= idx < len(output_chunks) else "???"
            )
            print(f"{idx:6d}  {text}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--question", default=None)
    parser.add_argument("--show_chunks", action="store_true")
    args = parser.parse_args()

    model, input_vocab, output_chunks = load_runtime(args.model_dir)

    print("Loaded:", args.model_dir)
    print("Device:", DEVICE)
    print("Input vocab:", len(input_vocab))
    print("Output chunks:", len(output_chunks))

    if args.question:
        predict(
            args.question,
            model,
            input_vocab,
            output_chunks,
            show_chunks=args.show_chunks,
        )
        return

    print()
    print("Interactive mode. Type quit / exit / stop.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"quit", "exit", "stop"}:
            break

        if q:
            predict(
                q,
                model,
                input_vocab,
                output_chunks,
                show_chunks=args.show_chunks,
            )


if __name__ == "__main__":
    main()
