# test_math_equation_translator.py

import argparse
import ast
import operator
import re
from pathlib import Path

import torch
import torch.nn as nn

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_MODEL = Path(
    "assets/models/math_equation_translator/math_equation_translator_final.pt"
)


def replace_math_symbols(text):
    text = str(text)
    replacements = {
        "×": "*",
        "✕": "*",
        "✖": "*",
        "∙": "*",
        "·": "*",
        "÷": "/",
        "∕": "/",
        "⁄": "/",
        "−": "-",
        "–": "-",
        "—": "-",
        "﹣": "-",
        "＋": "+",
        "＝": "=",
        "％": "%",
        "π": " pi ",
        "√": " sqrt ",
        "²": " ** 2 ",
        "³": " ** 3 ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def clean_text(x):
    x = replace_math_symbols(x)
    x = x.replace("\n", " ")
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def normalize_question(q):
    q = clean_text(q).lower()

    phrase_replacements = [
        (r"\bmultiplied\s+by\b", " times "),
        (r"\btimes\b", " * "),
        (r"\bplus\b", " + "),
        (r"\bminus\b", " - "),
        (r"\bdivided\s+by\b", " / "),
        (r"\bover\b", " / "),
        (r"\bto\s+the\s+power\s+of\b", " ^ "),
        (r"\bsquared\b", " ^ 2 "),
        (r"\bcubed\b", " ^ 3 "),
        (r"\bpercent\b", " % "),
    ]

    for pattern, repl in phrase_replacements:
        q = re.sub(pattern, repl, q)

    q = re.sub(r"[^a-z0-9_+\-*/^().,?:;$%=\s']", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def tokenize(text):
    text = str(text).lower()
    text = replace_math_symbols(text)
    return re.findall(
        r"\d+\.\d+|\d+|\*\*|sqrt|pi|[a-z_]+|[+\-*/^=().,?:;$%]",
        text,
    )


def detok(tokens):
    out = ""
    for t in tokens:
        t = str(t)
        if t in {".", ",", "?", "!", ":", ";", "%", ")", "]", "}"}:
            out = out.rstrip() + t
        elif t in {"(", "[", "{"}:
            if out and not out.endswith(" "):
                out += " "
            out += t
        elif t in {"+", "-", "*", "/", "**", "^", "="}:
            out += f" {t} "
        else:
            if out and not out.endswith((" ", "(", "[", "{")):
                out += " "
            out += t
    return re.sub(r"\s+", " ", out).strip()


def encode(tokens, vocab, max_len):
    ids = [BOS_ID]
    for tok in tokens:
        ids.append(vocab.get(tok, UNK_ID))
        if len(ids) >= max_len - 1:
            break
    ids.append(EOS_ID)
    ids = ids[:max_len]
    while len(ids) < max_len:
        ids.append(PAD_ID)
    return ids


def decode(ids, id_to_token):
    tokens = []
    for idx in ids:
        idx = int(idx)
        if idx == EOS_ID:
            break
        if idx in {PAD_ID, BOS_ID, UNK_ID}:
            continue
        tokens.append(id_to_token.get(idx, ""))
    return detok(tokens)


class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed, hidden, dropout):
        super().__init__()

        self.input_emb = nn.Embedding(input_vocab_size, embed, padding_idx=PAD_ID)
        self.output_emb = nn.Embedding(output_vocab_size, embed, padding_idx=PAD_ID)

        self.encoder = nn.GRU(embed, hidden, batch_first=True, bidirectional=True)

        self.bridge = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.Tanh(),
        )

        self.decoder = nn.GRU(embed, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, output_vocab_size)

    def encode_context(self, x):
        emb = self.dropout(self.input_emb(x))
        _, h = self.encoder(emb)
        h_fwd = h[-2]
        h_bwd = h[-1]
        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)
        return self.bridge(h_cat).unsqueeze(0)

    def forward(self, x, max_len):
        batch = x.size(0)
        h = self.encode_context(x)

        prev = torch.full((batch, 1), BOS_ID, dtype=torch.long, device=x.device)
        ids = []

        for _ in range(max_len):
            emb = self.dropout(self.output_emb(prev))
            dec_out, h = self.decoder(emb, h)
            logits = self.out(dec_out[:, -1])
            prev = torch.argmax(logits, dim=-1, keepdim=True)
            ids.append(prev)

        return torch.cat(ids, dim=1)


SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_eval_expr(expr):
    expr = expr.strip()
    expr = expr.replace("^", "**")

    if not expr:
        raise ValueError("empty expression")

    if not re.fullmatch(r"[0-9+\-*/().\s*]+", expr):
        raise ValueError("unsafe expression")

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in SAFE_OPS:
                raise ValueError("unsafe op")
            return SAFE_OPS[op_type](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in SAFE_OPS:
                raise ValueError("unsafe unary")
            return SAFE_OPS[op_type](_eval(node.operand))
        raise ValueError("unsafe expr")

    val = _eval(ast.parse(expr, mode="eval"))
    if isinstance(val, float) and abs(val - round(val)) < 1e-9:
        val = int(round(val))
    return val


def extract_equation_and_answer(text):
    equation = ""
    answer = ""

    m = re.search(r"equation\s*:\s*(.*?)(?:\s+answer\s*:|$)", text, flags=re.I)
    if m:
        equation = m.group(1).strip()

    m = re.search(r"answer\s*:\s*(.*)$", text, flags=re.I)
    if m:
        answer = m.group(1).strip()

    return equation, answer


def predict(
    question, model, input_vocab, output_id_to_token, max_input_len, max_output_len
):
    q = normalize_question(question)
    tokens = tokenize(q)

    x = torch.tensor(
        [encode(tokens, input_vocab, max_input_len)],
        dtype=torch.long,
        device=DEVICE,
    )

    with torch.no_grad():
        ids = model(x, max_output_len)[0].detach().cpu().tolist()

    decoded = decode(ids, output_id_to_token)
    equation, predicted_answer = extract_equation_and_answer(decoded)

    computed_answer = ""
    if equation:
        try:
            computed_answer = str(safe_eval_expr(equation))
        except Exception:
            computed_answer = ""

    return {
        "question": question,
        "normalized": q,
        "decoded": decoded,
        "equation": equation,
        "predicted_answer": predicted_answer,
        "computed_answer": computed_answer,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--question", default=None)
    args = parser.parse_args()

    checkpoint = torch.load(args.model, map_location=DEVICE)

    input_vocab = checkpoint["input_vocab"]
    output_vocab = checkpoint["output_vocab"]
    config = checkpoint["config"]

    output_id_to_token = {int(v): k for k, v in output_vocab.items()}

    model = Seq2Seq(
        input_vocab_size=len(input_vocab),
        output_vocab_size=len(output_vocab),
        embed=config["embed"],
        hidden=config["hidden"],
        dropout=config["dropout"],
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    max_input_len = config["max_input_len"]
    max_output_len = config["max_output_len"]

    print("Loaded:", args.model)
    print("Device:", DEVICE)
    print("Input vocab:", len(input_vocab))
    print("Output vocab:", len(output_vocab))

    if args.question:
        result = predict(
            args.question,
            model,
            input_vocab,
            output_id_to_token,
            max_input_len,
            max_output_len,
        )
        print()
        for k, v in result.items():
            print(f"{k}: {v}")
        return

    print()
    print("Interactive mode. Type quit / exit / stop.")
    while True:
        q = input("\nMath> ").strip()
        if q.lower() in {"quit", "exit", "stop"}:
            break
        if not q:
            continue

        result = predict(
            q,
            model,
            input_vocab,
            output_id_to_token,
            max_input_len,
            max_output_len,
        )

        print("normalized:", result["normalized"])
        print("decoded:", result["decoded"])
        print("equation:", result["equation"])
        print("predicted_answer:", result["predicted_answer"])
        print("computed_answer:", result["computed_answer"])


if __name__ == "__main__":
    main()
