import json
from pathlib import Path

import torch
import torch.nn as nn

DEVICE = "cpu"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

PROMPT_SIZE = 128
HIDDEN_SIZE = 192
EMBED_SIZE = 128
DROPOUT = 0.35
MAX_OUTPUT_CHUNKS = 24


MODELS = {
    "reaction": {
        "dir": Path("assets/models/meatball_reaction_model"),
        "pt": "reaction_model.pt",
        "onnx": "reaction_model.onnx",
        "default_hidden": 256,
        "default_dropout": 0.2,
        "fallback_labels": [
            "neutral",
            "excited",
            "confused",
            "suspicious",
            "angry",
            "sad",
            "overwhelmed",
        ],
    },
    "complexity": {
        "dir": Path("assets/models/complexity_classifier"),
        "pt": "complexity_classifier.pt",
        "onnx": "complexity_classifier.onnx",
        "default_hidden": 320,
        "default_dropout": 0.22,
        "fallback_labels": [
            "normal_qa",
            "list",
            "compare",
            "multi_part",
            "followup",
            "smalltalk",
            "unknown",
        ],
    },
    "math_classifier": {
        "dir": Path("assets/models/math_classifier"),
        "pt": "math_classifier.pt",
        "onnx": "math_classifier.onnx",
        "default_hidden": 384,
        "default_dropout": 0.25,
        "fallback_labels": ["general", "math"],
    },
}


GENERATOR_DIR = Path("assets/models/general_cover_chunks_noisy_continue")
MATH_MODEL_PATH = Path(
    "assets/models/math_equation_translator/math_equation_translator_final.pt"
)


def load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_labels(model_dir, config, fallback_labels=None):
    labels_path = model_dir / "labels.json"

    if labels_path.exists():
        return load_json(labels_path)

    if "labels" in config:
        labels = config["labels"]
        save_json(labels_path, labels)
        print(f"[fixed] wrote missing labels.json: {labels_path}")
        return labels

    if fallback_labels is not None:
        save_json(labels_path, fallback_labels)
        print(f"[fixed] wrote fallback labels.json: {labels_path}")
        return fallback_labels

    raise FileNotFoundError(
        f"Missing labels.json and no fallback labels for {model_dir}"
    )


class GenericClassifier(nn.Module):
    def __init__(self, input_size, classes, hidden, dropout):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, classes),
        )

    def forward(self, x):
        return self.net(x)


class SubjectInserterNet(nn.Module):
    def __init__(self, input_dim, hidden, num_labels, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, x):
        return self.net(x)


class SubjectFinderNet(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, dropout, pad_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=pad_id)
        self.encoder = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.has_subject_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.start_head = nn.Linear(hidden_size, 1)
        self.end_head = nn.Linear(hidden_size, 1)

    def forward(self, x, attention_mask):
        emb = self.embedding(x)
        h = self.encoder(emb)
        mask = attention_mask.unsqueeze(-1)
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        start_logits = (
            self.start_head(h).squeeze(-1).masked_fill(attention_mask == 0, -1e9)
        )
        end_logits = self.end_head(h).squeeze(-1).masked_fill(attention_mask == 0, -1e9)
        has_subject_logits = self.has_subject_head(pooled)
        return has_subject_logits, start_logits, end_logits


class ChunkAnswerModel(nn.Module):
    def __init__(self, input_size, output_vocab_size):
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
        self.decoder_cell = nn.GRUCell(PROMPT_SIZE + EMBED_SIZE, HIDDEN_SIZE)

        self.output = nn.Sequential(
            nn.LayerNorm(PROMPT_SIZE + HIDDEN_SIZE),
            nn.Dropout(DROPOUT),
            nn.Linear(PROMPT_SIZE + HIDDEN_SIZE, output_vocab_size),
        )

    def forward(self, x, max_len_tensor=None):
        batch_size = x.size(0)
        prompt_context = self.encoder(x)

        hidden = torch.zeros(batch_size, HIDDEN_SIZE, device=x.device)
        prev = torch.full(
            (batch_size,),
            BOS_ID,
            dtype=torch.long,
            device=x.device,
        )

        logits_steps = []

        for _ in range(MAX_OUTPUT_CHUNKS + 1):
            emb = self.embedding(prev)
            hidden = self.decoder_cell(torch.cat([emb, prompt_context], dim=-1), hidden)
            logits = self.output(torch.cat([prompt_context, hidden], dim=-1))
            logits_steps.append(logits.unsqueeze(1))
            prev = torch.argmax(logits, dim=-1)

        return torch.cat(logits_steps, dim=1)


class MathSeq2Seq(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        embed,
        hidden,
        dropout,
        max_output_len,
    ):
        super().__init__()

        self.max_output_len = int(max_output_len)

        self.input_emb = nn.Embedding(input_vocab_size, embed, padding_idx=PAD_ID)
        self.output_emb = nn.Embedding(output_vocab_size, embed, padding_idx=PAD_ID)

        self.encoder = nn.GRU(embed, hidden, batch_first=True, bidirectional=True)
        self.bridge = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.Tanh())

        self.decoder = nn.GRU(embed, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, output_vocab_size)

    def encode_context(self, x):
        emb = self.dropout(self.input_emb(x))
        _, h = self.encoder(emb)
        h = torch.cat([h[-2], h[-1]], dim=-1)
        return self.bridge(h).unsqueeze(0)

    def forward(self, x):
        batch = x.size(0)
        h = self.encode_context(x)

        prev = torch.full(
            (batch, 1),
            BOS_ID,
            dtype=torch.long,
            device=x.device,
        )

        logits_steps = []

        for _ in range(self.max_output_len):
            emb = self.dropout(self.output_emb(prev))
            dec_out, h = self.decoder(emb, h)
            logits = self.out(dec_out[:, -1])
            logits_steps.append(logits.unsqueeze(1))
            prev = torch.argmax(logits, dim=-1, keepdim=True)

        return torch.cat(logits_steps, dim=1)


def export_classifier(name, spec):
    model_dir = spec["dir"]
    pt_path = model_dir / spec["pt"]
    onnx_path = model_dir / spec["onnx"]

    if not pt_path.exists():
        print(f"[skip] missing {name}: {pt_path}")
        return

    input_vocab_path = model_dir / "input_vocab.json"

    if not input_vocab_path.exists():
        alt_vocab_path = model_dir / "vocab.json"
        if alt_vocab_path.exists():
            input_vocab_path = alt_vocab_path
        else:
            print(
                f"[skip] missing vocab for {name}: {model_dir / 'input_vocab.json'} or {alt_vocab_path}"
            )
            return

    config_path = model_dir / "config.json"
    config = load_json(config_path) if config_path.exists() else {}

    labels = resolve_labels(
        model_dir=model_dir,
        config=config,
        fallback_labels=spec.get("fallback_labels"),
    )

    hidden = int(config.get("hidden", spec["default_hidden"]))
    dropout = float(config.get("dropout", spec["default_dropout"]))

    vocab = load_json(input_vocab_path)
    ckpt = torch.load(pt_path, map_location="cpu")

    model = GenericClassifier(
        input_size=len(vocab),
        classes=len(labels),
        hidden=hidden,
        dropout=dropout,
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    dummy = torch.zeros(1, len(vocab), dtype=torch.float32)

    print(f"[export] {name}: {pt_path} -> {onnx_path}")

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )

    print(f"[ok] {onnx_path}")


def export_subject_inserter():
    model_dir = Path("assets/models/subject_inserter")
    pt_path = model_dir / "subject_inserter.pt"
    onnx_path = model_dir / "subject_inserter.onnx"
    vocab_path = model_dir / "vocab.json"
    labels_path = model_dir / "labels.json"
    config_path = model_dir / "config.json"

    if not pt_path.exists():
        print(f"[skip] missing subject_inserter: {pt_path}")
        return
    if not vocab_path.exists() or not labels_path.exists() or not config_path.exists():
        print(f"[skip] missing subject_inserter artifacts in {model_dir}")
        return

    vocab = load_json(vocab_path)
    labels = load_json(labels_path)
    config = load_json(config_path)
    state_dict = torch.load(pt_path, map_location="cpu")

    model = SubjectInserterNet(
        input_dim=int(config.get("input_dim", len(vocab))),
        hidden=int(config.get("hidden", 256)),
        num_labels=int(config.get("num_labels", len(labels))),
        dropout=float(config.get("dropout", 0.2)),
    ).to(DEVICE)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    dummy = torch.zeros(1, len(vocab), dtype=torch.float32)

    print(f"[export] subject_inserter: {pt_path} -> {onnx_path}")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"[ok] {onnx_path}")


def export_subject_finder():
    model_dir = Path("assets/models/subject_finder")
    pt_path = model_dir / "subject_finder.pt"
    onnx_path = model_dir / "subject_finder.onnx"
    vocab_path = model_dir / "vocab.json"
    config_path = model_dir / "config.json"

    if not pt_path.exists():
        print(f"[skip] missing subject_finder: {pt_path}")
        return
    if not vocab_path.exists() or not config_path.exists():
        print(f"[skip] missing subject_finder artifacts in {model_dir}")
        return

    vocab = load_json(vocab_path)
    config = load_json(config_path)
    ckpt = torch.load(pt_path, map_location="cpu")

    model = SubjectFinderNet(
        vocab_size=int(ckpt.get("vocab_size", len(vocab))),
        embed_size=int(ckpt.get("embed_size", config.get("embed_size", 128))),
        hidden_size=int(ckpt.get("hidden_size", config.get("hidden_size", 192))),
        dropout=float(ckpt.get("dropout", config.get("dropout", 0.25))),
        pad_id=int(config.get("pad_id", PAD_ID)),
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    max_len = int(ckpt.get("max_len", config.get("max_len", 96)))
    dummy_ids = torch.zeros(1, max_len, dtype=torch.long)
    dummy_mask = torch.ones(1, max_len, dtype=torch.float32)

    print(f"[export] subject_finder: {pt_path} -> {onnx_path}")
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["has_subject_logits", "start_logits", "end_logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "has_subject_logits": {0: "batch"},
            "start_logits": {0: "batch", 1: "seq"},
            "end_logits": {0: "batch", 1: "seq"},
        },
        opset_version=17,
    )
    print(f"[ok] {onnx_path}")


def export_generator(model_dir):
    model_dir = Path(model_dir)
    pt_path = model_dir / "model.pt"
    onnx_path = model_dir / "model.onnx"

    if not pt_path.exists():
        print(f"[skip] missing generator: {pt_path}")
        return

    input_vocab_path = model_dir / "input_vocab.json"
    output_chunks_path = model_dir / "output_chunks.json"

    if not input_vocab_path.exists() or not output_chunks_path.exists():
        print(f"[skip] missing generator vocab/chunks in {model_dir}")
        return

    input_vocab = load_json(input_vocab_path)
    output_chunks = load_json(output_chunks_path)
    ckpt = torch.load(pt_path, map_location="cpu")

    model = ChunkAnswerModel(
        input_size=len(input_vocab),
        output_vocab_size=len(output_chunks),
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    dummy = torch.zeros(1, len(input_vocab), dtype=torch.float32)

    print(f"[export] generator: {pt_path} -> {onnx_path}")

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )

    print(f"[ok] {onnx_path}")


def export_math_model(path):
    path = Path(path)

    if not path.exists():
        print(f"[skip] missing math seq2seq: {path}")
        return

    onnx_path = path.with_suffix(".onnx")

    ckpt = torch.load(path, map_location="cpu")

    cfg = ckpt["config"]
    input_vocab = ckpt["input_vocab"]
    output_vocab = ckpt["output_vocab"]

    model = MathSeq2Seq(
        input_vocab_size=len(input_vocab),
        output_vocab_size=len(output_vocab),
        embed=int(cfg["embed"]),
        hidden=int(cfg["hidden"]),
        dropout=float(cfg["dropout"]),
        max_output_len=int(cfg["max_output_len"]),
    ).to(DEVICE)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    dummy = torch.zeros(
        1,
        int(cfg["max_input_len"]),
        dtype=torch.long,
    )

    print(f"[export] math seq2seq: {path} -> {onnx_path}")

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )

    print(f"[ok] {onnx_path}")


def main():
    print("Exporting Meatball models to ONNX...")

    for name, spec in MODELS.items():
        try:
            export_classifier(name, spec)
        except Exception as e:
            print(f"[fail] {name}: {e}")

    try:
        export_generator(GENERATOR_DIR)
    except Exception as e:
        print(f"[fail] generator: {e}")

    try:
        export_subject_inserter()
    except Exception as e:
        print(f"[fail] subject_inserter: {e}")

    try:
        export_subject_finder()
    except Exception as e:
        print(f"[fail] subject_finder: {e}")

    try:
        export_math_model(MATH_MODEL_PATH)
    except Exception as e:
        print(f"[fail] math seq2seq: {e}")

    print("DONE")


if __name__ == "__main__":
    main()
