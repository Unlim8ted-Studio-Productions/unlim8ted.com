# tools/context_neural_net_train.py
#
# Train a tiny context-action model for Smarter Meatball.
#
# This model does NOT answer questions.
# It predicts how to frame the selected answer:
#   direct_answer
#   same_project_followup
#   expand_previous
#   topic_shift
#   correct_misunderstanding
#   clarify
#   soft_refusal
#   reset_needed
#   playful_bridge
#   continue_previous
#
# Install:
#   python -m pip install torch numpy scikit-learn onnx
#
# Train:
#   python tools/context_neural_net_train.py assets/data/Smart-Meatball-Context-Data.jsonl
#
# Output:
#   dist/context-model/meatball_context.onnx
#   dist/context-model/meatball_context_metadata.json
#   dist/context-model/browser_context_inference.js

import json
import random
import re
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


CONFIG = {
    "feature_size": 2048,
    "ngram_min": 2,
    "ngram_max": 5,
    "memory_feature_size": 256,
    "hidden_units": 64,
    "dropout": 0.15,
    "epochs": 60,
    "batch_size": 24,
    "learning_rate": 0.001,
    "seed": 42,
    "output_dir": "dist/context-model",
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


def memory_to_vector(row: dict) -> np.ndarray:
    vec = np.zeros(CONFIG["memory_feature_size"], dtype=np.float32)

    fields = [
        "last_intent",
        "last_project_key",
        "last_category",
        "current_intent",
        "current_project_key",
        "current_category",
    ]

    for field in fields:
        value = str(row.get(field, "none") or "none").lower()
        index = fnv1a_hash(f"{field}:{value}") % CONFIG["memory_feature_size"]
        vec[index] += 1.0

    message_count = float(row.get("message_count", 0) or 0)
    confusion_count = float(row.get("confusion_count", 0) or 0)
    topic_switch_count = float(row.get("topic_switch_count", 0) or 0)

    vec[0] = min(message_count / 20.0, 1.0)
    vec[1] = min(confusion_count / 8.0, 1.0)
    vec[2] = min(topic_switch_count / 8.0, 1.0)

    return vec


def row_to_vector(row: dict) -> np.ndarray:
    text_vec = text_to_vector(row.get("question", ""))
    memory_vec = memory_to_vector(row)
    return np.concatenate([text_vec, memory_vec]).astype(np.float32)


class ContextNet(nn.Module):
    def __init__(self, input_size: int, hidden_units: int, output_size: int, dropout: float):
        super().__init__()

        mid_units = max(24, hidden_units // 2)

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


def batch_iter(x, y, batch_size):
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


def make_browser_js(actions):
    browser_config = {
        "feature_size": CONFIG["feature_size"],
        "memory_feature_size": CONFIG["memory_feature_size"],
        "ngram_min": CONFIG["ngram_min"],
        "ngram_max": CONFIG["ngram_max"],
    }

    return f"""
const MEATBALL_CONTEXT_CONFIG = {json.dumps(browser_config, indent=2)};
const MEATBALL_CONTEXT_ACTIONS = {json.dumps(actions, indent=2)};

function mbNormalizeText(text) {{
  return String(text || "")
    .toLowerCase()
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/[^a-z0-9?!.,'"\\s:_/-]/g, " ")
    .replace(/\\s+/g, " ")
    .trim();
}}

function mbHash(text) {{
  let h = 2166136261;

  for (let i = 0; i < text.length; i++) {{
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }}

  return h >>> 0;
}}

function mbTextToVector(text) {{
  const vec = new Float32Array(MEATBALL_CONTEXT_CONFIG.feature_size);
  const clean = ` ${{mbNormalizeText(text)}} `;

  for (let n = MEATBALL_CONTEXT_CONFIG.ngram_min; n <= MEATBALL_CONTEXT_CONFIG.ngram_max; n++) {{
    for (let i = 0; i <= clean.length - n; i++) {{
      const gram = clean.slice(i, i + n);
      const index = mbHash(gram) % MEATBALL_CONTEXT_CONFIG.feature_size;
      vec[index] += 1.0;
    }}
  }}

  const words = clean.trim().split(/\\s+/).filter(Boolean);

  for (const word of words) {{
    const index = mbHash(`word:${{word}}`) % MEATBALL_CONTEXT_CONFIG.feature_size;
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

function mbMemoryToVector(memory) {{
  const vec = new Float32Array(MEATBALL_CONTEXT_CONFIG.memory_feature_size);

  const fields = [
    "last_intent",
    "last_project_key",
    "last_category",
    "current_intent",
    "current_project_key",
    "current_category"
  ];

  for (const field of fields) {{
    const value = String(memory[field] || "none").toLowerCase();
    const index = mbHash(`${{field}}:${{value}}`) % MEATBALL_CONTEXT_CONFIG.memory_feature_size;
    vec[index] += 1.0;
  }}

  vec[0] = Math.min((memory.message_count || 0) / 20.0, 1.0);
  vec[1] = Math.min((memory.confusion_count || 0) / 8.0, 1.0);
  vec[2] = Math.min((memory.topic_switch_count || 0) / 8.0, 1.0);

  return vec;
}}

function mbContextInputVector(question, memory) {{
  const textVec = mbTextToVector(question);
  const memoryVec = mbMemoryToVector(memory);

  const out = new Float32Array(
    MEATBALL_CONTEXT_CONFIG.feature_size +
    MEATBALL_CONTEXT_CONFIG.memory_feature_size
  );

  out.set(textVec, 0);
  out.set(memoryVec, MEATBALL_CONTEXT_CONFIG.feature_size);

  return out;
}}

function mbSoftmax(logits) {{
  const max = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}}

async function loadMeatballContextModel(modelUrl = "/models/context-model/meatball_context.onnx") {{
  return await ort.InferenceSession.create(modelUrl, {{
    executionProviders: ["wasm"]
  }});
}}

async function predictMeatballContextAction(session, question, memory) {{
  const inputVector = mbContextInputVector(question, memory);

  const tensor = new ort.Tensor(
    "float32",
    inputVector,
    [1, inputVector.length]
  );

  const outputs = await session.run({{
    input: tensor
  }});

  const logits = Array.from(outputs.logits.data);
  const scores = mbSoftmax(logits);

  const ranked = scores
    .map((score, index) => ({{
      action: MEATBALL_CONTEXT_ACTIONS[index],
      score
    }}))
    .sort((a, b) => b.score - a.score);

  return {{
    action: ranked[0].action,
    confidence: ranked[0].score,
    top: ranked.slice(0, 5)
  }};
}}

window.loadMeatballContextModel = loadMeatballContextModel;
window.predictMeatballContextAction = predictMeatballContextAction;
""".strip()


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/context_neural_net_train.py assets/data/Smart-Meatball-Context-Data.jsonl")
        sys.exit(1)

    rows = []

    for path in sys.argv[1:]:
        loaded = read_jsonl(path)
        print(f"Loaded {len(loaded)} rows from {path}")
        rows.extend(loaded)

    rows = [
        row
        for row in rows
        if row.get("question") and row.get("context_action")
    ]

    if not rows:
        raise RuntimeError("No usable context rows found.")

    actions = sorted({row["context_action"] for row in rows})
    action_to_index = {action: i for i, action in enumerate(actions)}

    print(f"Rows: {len(rows)}")
    print(f"Context actions: {len(actions)}")

    for action in actions:
        print(f"  {action}")

    x = np.stack([row_to_vector(row) for row in rows])
    y = np.array([action_to_index[row["context_action"]] for row in rows], dtype=np.int64)

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

    model = ContextNet(
        input_size=CONFIG["feature_size"] + CONFIG["memory_feature_size"],
        hidden_units=CONFIG["hidden_units"],
        output_size=len(actions),
        dropout=CONFIG["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = None
    patience = 8

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

        if epoch == 1 or epoch % 10 == 0 or epoch == CONFIG["epochs"]:
            print(
                f"epoch {epoch:03d} | "
                f"loss={total_loss / max(batch_count, 1):.4f} | "
                f"train_acc={train_acc:.4f} | "
                f"val_acc={val_acc:.4f}"
            )

        if val_acc > best_val_acc + 0.0005:
            best_val_acc = val_acc
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 8
        else:
            patience -= 1

        if patience <= 0:
            print(f"Early stopping at epoch {epoch:03d}. Best val_acc={best_val_acc:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    dummy_input = torch.randn(
        1,
        CONFIG["feature_size"] + CONFIG["memory_feature_size"],
        dtype=torch.float32,
        device=device,
    )

    onnx_path = output_dir / "meatball_context.onnx"

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
        "config": CONFIG,
        "actions": actions,
        "row_count": len(rows),
    }

    with open(output_dir / "meatball_context_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(output_dir / "browser_context_inference.js", "w", encoding="utf-8") as f:
        f.write(make_browser_js(actions))

    print(f"Saved: {onnx_path}")
    print(f"Saved: {output_dir / 'meatball_context_metadata.json'}")
    print(f"Saved: {output_dir / 'browser_context_inference.js'}")


if __name__ == "__main__":
    main()