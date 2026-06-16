# tools/intent_nueral_net_train.py
#
# Tiny neural intent classifier for Smarter Meatball.
# Includes typo augmentation so the model learns common misspellings.
#
# Install:
#   python -m pip install torch numpy scikit-learn onnx
#
# Train:
#   python tools/intent_nueral_net_train.py assets/data/Smart-Meatball-Data.jsonl
#
# Output:
#   dist/meatball_intent.onnx
#   dist/meatball_metadata.json
#   dist/browser_inference.js

import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


CONFIG = {
    "feature_size": 4096,
    "ngram_min": 2,
    "ngram_max": 5,
    "hidden_units": 96,
    "dropout": 0.18,
    "epochs": 40,
    "batch_size": 32,
    "learning_rate": 0.001,
    "min_rows_per_intent": 2,
    "seed": 42,
    "output_dir": "/assets/models/intent-model/",
}

TYPO_AUGMENTATION = {
    "enabled": True,
    "copies_per_row": 2,
    "probability": 0.35,
    "max_changes": 3,
}

EARLY_STOPPING = {
    "enabled": True,
    "patience": 8,
    "min_delta": 0.0005,
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])


def read_jsonl(path: str) -> List[dict]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}: {exc}")

    return rows


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"[^a-z0-9?!.,'\"\s:_/\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fnv1a_hash(text: str) -> int:
    h = 2166136261

    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF

    return h


def text_to_vector(text: str) -> np.ndarray:
    vec = np.zeros(CONFIG["feature_size"], dtype=np.float32)
    clean = f" {normalize_text(text)} "

    for n in range(CONFIG["ngram_min"], CONFIG["ngram_max"] + 1):
        for i in range(0, len(clean) - n + 1):
            gram = clean[i : i + n]
            index = fnv1a_hash(gram) % CONFIG["feature_size"]
            vec[index] += 1.0

    for word in clean.strip().split():
        index = fnv1a_hash(f"word:{word}") % CONFIG["feature_size"]
        vec[index] += 1.5

    norm = np.linalg.norm(vec)

    if norm > 0:
        vec /= norm

    return vec


def random_typo(text: str) -> str:
    if not text or len(text) < 4:
        return text

    chars = list(text)
    changes = random.randint(1, TYPO_AUGMENTATION["max_changes"])

    nearby = {
        "a": "qwsz",
        "b": "vghn",
        "c": "xdfv",
        "d": "serfcx",
        "e": "wsdr",
        "f": "drtgvc",
        "g": "ftyhbv",
        "h": "gyujnb",
        "i": "ujko",
        "j": "huikmn",
        "k": "jiolm",
        "l": "kop",
        "m": "njk",
        "n": "bhjm",
        "o": "iklp",
        "p": "ol",
        "q": "wa",
        "r": "edft",
        "s": "awedxz",
        "t": "rfgy",
        "u": "yhji",
        "v": "cfgb",
        "w": "qase",
        "x": "zsdc",
        "y": "tghu",
        "z": "asx",
    }

    for _ in range(changes):
        if len(chars) < 2:
            break

        i = random.randint(0, len(chars) - 1)
        operation = random.choice(["delete", "swap", "replace", "nearby", "insert", "double"])

        if operation == "delete":
            if chars[i] != " ":
                chars.pop(i)

        elif operation == "swap":
            if i < len(chars) - 1 and chars[i] != " " and chars[i + 1] != " ":
                chars[i], chars[i + 1] = chars[i + 1], chars[i]

        elif operation == "replace":
            if chars[i].isalpha():
                chars[i] = random.choice("abcdefghijklmnopqrstuvwxyz")

        elif operation == "nearby":
            ch = chars[i].lower()
            if ch in nearby:
                replacement = random.choice(nearby[ch])
                chars[i] = replacement.upper() if chars[i].isupper() else replacement

        elif operation == "insert":
            if chars[i] != " ":
                chars.insert(i, random.choice("abcdefghijklmnopqrstuvwxyz"))

        elif operation == "double":
            if chars[i].isalpha():
                chars.insert(i, chars[i])

    return "".join(chars)


def augment_with_typos(rows: List[dict]) -> List[dict]:
    if not TYPO_AUGMENTATION["enabled"]:
        return rows

    augmented = list(rows)
    seen = {(normalize_text(row["question"]), row["intent"]) for row in rows}

    for row in rows:
        question = row["question"]

        for copy_index in range(TYPO_AUGMENTATION["copies_per_row"]):
            if random.random() > TYPO_AUGMENTATION["probability"]:
                continue

            typo_question = random_typo(question)
            key = (normalize_text(typo_question), row["intent"])

            if typo_question == question or key in seen:
                continue

            seen.add(key)

            augmented.append(
                {
                    **row,
                    "id": f"{row.get('id', 'row')}_typo_{copy_index + 1}",
                    "question": typo_question,
                    "tags": list(row.get("tags", [])) + ["typo_augmented"],
                    "source_note": str(row.get("source_note", "")) + "; typo augmented during training",
                }
            )

    return augmented


class MeatballIntentNet(nn.Module):
    def __init__(self, input_size: int, hidden_units: int, output_size: int, dropout: float):
        super().__init__()

        mid_units = max(32, hidden_units // 2)

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_units, mid_units),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(mid_units, output_size),
        )

    def forward(self, x):
        return self.net(x)


def batch_iter(x, y, batch_size: int):
    indices = np.arange(len(x))
    np.random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield x[batch_indices], y[batch_indices]


def accuracy(model, x, y, device):
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32, device=device))
        preds = torch.argmax(logits, dim=1)
        labels = torch.tensor(y, dtype=torch.long, device=device)
        return (preds == labels).float().mean().item()


def top_predictions(model, intents, question, device, k=5):
    model.eval()

    vec = text_to_vector(question)
    x = torch.tensor(np.array([vec]), dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    ranked = sorted(
        [{"intent": intents[i], "score": float(probs[i])} for i in range(len(intents))],
        key=lambda item: item["score"],
        reverse=True,
    )

    return ranked[:k]


def make_browser_inference_js(intents: List[str]) -> str:
    config_for_browser = {
        "feature_size": CONFIG["feature_size"],
        "ngram_min": CONFIG["ngram_min"],
        "ngram_max": CONFIG["ngram_max"],
    }

    return f"""
const MEATBALL_CONFIG = {json.dumps(config_for_browser, indent=2)};
const MEATBALL_INTENTS = {json.dumps(intents, indent=2)};

function normalizeText(text) {{
  return String(text || "")
    .toLowerCase()
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/[^a-z0-9?!.,'"\\s:_/-]/g, " ")
    .replace(/\\s+/g, " ")
    .trim();
}}

function fnv1aHash(text) {{
  let h = 2166136261;

  for (let i = 0; i < text.length; i++) {{
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }}

  return h >>> 0;
}}

function textToVector(text) {{
  const vec = new Float32Array(MEATBALL_CONFIG.feature_size);
  const clean = ` ${{normalizeText(text)}} `;

  for (let n = MEATBALL_CONFIG.ngram_min; n <= MEATBALL_CONFIG.ngram_max; n++) {{
    for (let i = 0; i <= clean.length - n; i++) {{
      const gram = clean.slice(i, i + n);
      const index = fnv1aHash(gram) % MEATBALL_CONFIG.feature_size;
      vec[index] += 1.0;
    }}
  }}

  const words = clean.trim().split(/\\s+/).filter(Boolean);

  for (const word of words) {{
    const index = fnv1aHash(`word:${{word}}`) % MEATBALL_CONFIG.feature_size;
    vec[index] += 1.5;
  }}

  let sum = 0;

  for (let i = 0; i < vec.length; i++) {{
    sum += vec[i] * vec[i];
  }}

  const norm = Math.sqrt(sum) || 1;

  for (let i = 0; i < vec.length; i++) {{
    vec[i] /= norm;
  }}

  return vec;
}}

async function loadMeatballIntentModel(modelUrl = "/models/meatball_intent.onnx") {{
  return await ort.InferenceSession.create(modelUrl, {{
    executionProviders: ["wasm"]
  }});
}}

function softmax(logits) {{
  const max = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}}

async function predictMeatballIntent(session, question) {{
  const inputVector = textToVector(question);

  const inputTensor = new ort.Tensor(
    "float32",
    inputVector,
    [1, MEATBALL_CONFIG.feature_size]
  );

  const outputs = await session.run({{
    input: inputTensor
  }});

  const logits = Array.from(outputs.logits.data);
  const scores = softmax(logits);

  const ranked = scores
    .map((score, index) => ({{
      intent: MEATBALL_INTENTS[index],
      score
    }}))
    .sort((a, b) => b.score - a.score);

  return {{
    intent: ranked[0].intent,
    confidence: ranked[0].score,
    top: ranked.slice(0, 5)
  }};
}}

window.loadMeatballIntentModel = loadMeatballIntentModel;
window.predictMeatballIntent = predictMeatballIntent;
""".strip()


def main():
    if len(sys.argv) < 2:
        print(
            """
Usage:
  python tools/intent_nueral_net_train.py dataset1.jsonl dataset2.jsonl

Example:
  python tools/intent_nueral_net_train.py assets/data/Smart-Meatball-Data.jsonl
""".strip()
        )
        sys.exit(1)

    input_files = sys.argv[1:]
    rows = []

    for file_path in input_files:
        loaded = read_jsonl(file_path)
        print(f"Loaded {len(loaded)} rows from {file_path}")
        rows.extend(loaded)

    cleaned_rows = []

    for row in rows:
        question = str(row.get("question") or row.get("text") or "").strip()
        intent = str(row.get("intent") or "").strip()

        if not question or not intent:
            continue

        cleaned_rows.append(
            {
                **row,
                "question": question,
                "intent": intent,
            }
        )

    seen = set()
    deduped_rows = []

    for row in cleaned_rows:
        key = (normalize_text(row["question"]), row["intent"])

        if key in seen:
            continue

        seen.add(key)
        deduped_rows.append(row)

    intent_counts: Dict[str, int] = {}

    for row in deduped_rows:
        intent_counts[row["intent"]] = intent_counts.get(row["intent"], 0) + 1

    usable_rows = [
        row
        for row in deduped_rows
        if intent_counts[row["intent"]] >= CONFIG["min_rows_per_intent"]
    ]

    if not usable_rows:
        raise RuntimeError("No usable training rows found.")

    original_usable_count = len(usable_rows)
    usable_rows = augment_with_typos(usable_rows)

    intents = sorted({row["intent"] for row in usable_rows})
    intent_to_index = {intent: index for index, intent in enumerate(intents)}

    print(f"Rows before typo augmentation: {original_usable_count}")
    print(f"Rows after typo augmentation: {len(usable_rows)}")
    print(f"Intents: {len(intents)}")

    x = np.stack([text_to_vector(row["question"]) for row in usable_rows])
    y = np.array([intent_to_index[row["intent"]] for row in usable_rows], dtype=np.int64)

    class_counts = np.bincount(y)
    stratify = y if len(class_counts) and min(class_counts) >= 2 else None

    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.16,
        random_state=CONFIG["seed"],
        stratify=stratify,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = MeatballIntentNet(
        input_size=CONFIG["feature_size"],
        hidden_units=CONFIG["hidden_units"],
        output_size=len(intents),
        dropout=CONFIG["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None
    patience_left = EARLY_STOPPING["patience"]

    for epoch in range(1, CONFIG["epochs"] + 1):
        model.train()

        total_loss = 0.0
        batch_count = 0

        for xb, yb in batch_iter(x_train, y_train, CONFIG["batch_size"]):
            xb_tensor = torch.tensor(xb, dtype=torch.float32, device=device)
            yb_tensor = torch.tensor(yb, dtype=torch.long, device=device)

            optimizer.zero_grad()

            logits = model(xb_tensor)
            loss = loss_fn(logits, yb_tensor)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        train_acc = accuracy(model, x_train, y_train, device)
        val_acc = accuracy(model, x_val, y_val, device)
        avg_loss = total_loss / max(batch_count, 1)

        if epoch == 1 or epoch % 10 == 0 or epoch == CONFIG["epochs"]:
            print(
                f"epoch {epoch:03d} | "
                f"loss={avg_loss:.4f} | "
                f"train_acc={train_acc:.4f} | "
                f"val_acc={val_acc:.4f}"
            )

        if EARLY_STOPPING["enabled"]:
            if val_acc > best_val_acc + EARLY_STOPPING["min_delta"]:
                best_val_acc = val_acc
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                patience_left = EARLY_STOPPING["patience"]
            else:
                patience_left -= 1

            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch:03d}. Best val_acc={best_val_acc:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\nSanity checks:")

    sanity_questions = [
        "what is timecat",
        "whta is timcat",
        "tell me about unlim8ted",
        "what is gravity",
        "how does javascript work",
        "diagnose my illness",
        "reset the conversation",
        "what is the meatball",
        "asdfghjkl sauce banana",
    ]

    for question in sanity_questions:
        print(f"\n{question}")

        for item in top_predictions(model, intents, question, device):
            print(f"  {item['intent']}: {item['score']:.4f}")

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy_input = torch.randn(1, CONFIG["feature_size"], dtype=torch.float32, device=device)

    onnx_path = output_dir / "meatball_intent.onnx"

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )

    metadata = {
        "created_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "config": CONFIG,
        "typo_augmentation": TYPO_AUGMENTATION,
        "early_stopping": EARLY_STOPPING,
        "input_files": input_files,
        "row_count_before_typo_augmentation": original_usable_count,
        "row_count_after_typo_augmentation": len(usable_rows),
        "intents": intents,
        "intent_counts": dict(sorted(intent_counts.items())),
    }

    metadata_path = output_dir / "meatball_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    browser_js_path = output_dir / "browser_inference.js"

    with open(browser_js_path, "w", encoding="utf-8") as f:
        f.write(make_browser_inference_js(intents))

    print(f"\nSaved ONNX model: {onnx_path}")
    print(f"Saved metadata: {metadata_path}")
    print(f"Saved browser inference helper: {browser_js_path}")


if __name__ == "__main__":
    main()