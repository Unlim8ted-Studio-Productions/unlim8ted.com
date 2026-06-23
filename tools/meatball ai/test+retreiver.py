import re
from pathlib import Path

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = Path(r"tools\meatball ai\alternating_latent_chunk_qa_out\alternating_latent_chunk_qa_public_encoder.pt")

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


def decode_answer(ids, id_to_token):
    toks = []

    for idx in ids:
        idx = int(idx)

        if idx == EOS_ID:
            break

        if idx in (PAD_ID, BOS_ID):
            continue

        toks.append(id_to_token.get(idx, UNK))

    text = " ".join(toks)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = text.replace(" ’ ", "’")
    text = text.replace(" 's", "'s")
    text = text.replace(" n't", "n't")
    text = text.replace(" 'm", "'m")
    text = text.replace(" 're", "'re")
    text = text.replace(" 've", "'ve")
    text = text.replace(" 'll", "'ll")
    text = text.replace(" 'd", "'d")
    return text.strip()


class PublicEmbeddingToLatentModel(nn.Module):
    def __init__(self, public_embed_size, latent_size, latent_internal_steps, dropout):
        super().__init__()

        self.latent_size = latent_size
        self.latent_internal_steps = latent_internal_steps

        self.project = nn.Sequential(
            nn.Linear(public_embed_size, latent_size),
            nn.LayerNorm(latent_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.latent_refine = nn.GRUCell(latent_size, latent_size)
        self.latent_norm = nn.LayerNorm(latent_size)

        self.final = nn.Sequential(
            nn.Linear(latent_size, latent_size),
            nn.LayerNorm(latent_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, public_embeddings):
        z = self.project(public_embeddings)

        h = z
        x = z

        for _ in range(self.latent_internal_steps):
            h = self.latent_refine(x, h)
            h = self.latent_norm(h)
            x = h

        return self.final(h)


class PredictionModel(nn.Module):
    def __init__(self, answer_vocab_size, latent_size, answer_embed_size, pred_hidden, dropout):
        super().__init__()

        self.latent_size = latent_size

        self.answer_embedding = nn.Embedding(
            answer_vocab_size,
            answer_embed_size,
            padding_idx=PAD_ID,
        )

        self.hidden_init = nn.Linear(latent_size, pred_hidden)

        self.gru = nn.GRUCell(
            input_size=answer_embed_size + latent_size,
            hidden_size=pred_hidden,
        )

        self.output = nn.Sequential(
            nn.Linear(pred_hidden + latent_size, pred_hidden),
            nn.LayerNorm(pred_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_hidden, answer_vocab_size),
        )

    @torch.no_grad()
    def generate(self, z, max_len):
        batch_size = z.shape[0]

        h = torch.tanh(self.hidden_init(z))

        prev_ids = torch.full(
            (batch_size,),
            BOS_ID,
            dtype=torch.long,
            device=z.device,
        )

        outputs = []

        for _ in range(max_len):
            emb = self.answer_embedding(prev_ids)
            inp = torch.cat([emb, z], dim=-1)

            h = self.gru(inp, h)

            logits = self.output(torch.cat([h, z], dim=-1))
            next_ids = torch.argmax(logits, dim=-1)

            outputs.append(next_ids)
            prev_ids = next_ids

            if torch.all(next_ids == EOS_ID):
                break

        return torch.stack(outputs, dim=1)


@torch.no_grad()
def encode_question(public_encoder, question):
    return public_encoder.encode(
        [question],
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        device=DEVICE,
    )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Could not find checkpoint: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    config = checkpoint["config"]

    public_encoder_name = checkpoint.get(
        "public_encoder_name",
        config.get("public_encoder_name", "sentence-transformers/all-MiniLM-L6-v2"),
    )

    id_to_token = checkpoint["id_to_token"]

    # JSON can turn int keys into strings depending on how saved/loaded.
    id_to_token = {int(k): v for k, v in id_to_token.items()}

    public_encoder = SentenceTransformer(public_encoder_name, device=DEVICE)
    public_encoder.eval()

    latent_model = PublicEmbeddingToLatentModel(
        public_embed_size=config["public_embed_size"],
        latent_size=config["latent_size"],
        latent_internal_steps=config["latent_internal_steps"],
        dropout=config["dropout"],
    ).to(DEVICE)

    prediction_model = PredictionModel(
        answer_vocab_size=len(checkpoint["answer_vocab"]),
        latent_size=config["latent_size"],
        answer_embed_size=config["answer_embed_size"],
        pred_hidden=config["pred_hidden"],
        dropout=config["dropout"],
    ).to(DEVICE)

    latent_model.load_state_dict(checkpoint["latent_model_state"])
    prediction_model.load_state_dict(checkpoint["prediction_model_state"])

    latent_model.eval()
    prediction_model.eval()

    print("Model loaded successfully.")
    print("Device:", DEVICE)
    print("Encoder:", public_encoder_name)
    print()
    print("Type a question. Type 'quit' to exit.")
    print()

    while True:
        question = input("Question: ").strip()

        if question.lower() in {"quit", "exit", "q"}:
            break

        if not question:
            continue

        embeddings = encode_question(public_encoder, question)
        z = latent_model(embeddings)

        pred_ids = prediction_model.generate(
            z,
            max_len=config["max_generate_len"],
        )[0].detach().cpu().tolist()

        answer = decode_answer(pred_ids, id_to_token)

        print()
        print("Answer:", answer)
        print()


if __name__ == "__main__":
    main()