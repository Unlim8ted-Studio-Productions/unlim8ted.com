# train_math_blended_phases.py
from datasets import load_dataset

import argparse
import ast
import json
import math
import operator
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DATA_DIR = Path("assets/data/math")
OUT_MODEL_DIR = Path("assets/models/math_equation_translator")
OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

MAX_INPUT_LEN = 96
MAX_OUTPUT_LEN = 64
MAX_INPUT_VOCAB = 24000
MAX_OUTPUT_VOCAB = 12000

BATCH_SIZE = 64
GRAD_CLIP = 1.0
VAL_SPLIT = 0.08

EMBED = 128
HIDDEN = 256
DROPOUT = 0.25

LR = 8e-4
WEIGHT_DECAY = 1e-3

PAD = "<PAD>"
BOS = "<BOS>"
EOS = "<EOS>"
UNK = "<UNK>"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

SPECIALS = [PAD, BOS, EOS, UNK]

PHASES = [
    {
        "name": "phase_1_arithmetic_heavy",
        "rows": 250000,
        "epochs": 4,
        "mix": {
            "synthetic": 0.85,
            "mathqa": 0.05,
            "gsm8k": 0.05,
            "mawps_asdiv_svamp": 0.05,
        },
    },
    {
        "name": "phase_2_program_heavy",
        "rows": 160000,
        "epochs": 4,
        "mix": {
            "synthetic": 0.45,
            "mathqa": 0.35,
            "gsm8k": 0.10,
            "mawps_asdiv_svamp": 0.10,
        },
    },
    {
        "name": "phase_3_word_problem_heavy",
        "rows": 120000,
        "epochs": 5,
        "mix": {
            "synthetic": 0.30,
            "mathqa": 0.20,
            "gsm8k": 0.35,
            "mawps_asdiv_svamp": 0.15,
        },
    },
    {
        "name": "phase_4_clean_word_problem_tune",
        "rows": 80000,
        "epochs": 4,
        "mix": {
            "synthetic": 0.25,
            "mathqa": 0.15,
            "gsm8k": 0.20,
            "mawps_asdiv_svamp": 0.40,
        },
    },
]

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def save_json(path, data):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def normalize_equation(eq):
    eq = clean_text(eq)
    eq = eq.replace("^", "**")
    eq = re.sub(r"\s+", " ", eq).strip()
    return eq


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


def get_field(row, names):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]

    lower_map = {str(k).lower(): k for k in row.keys()}

    for name in names:
        key = lower_map.get(str(name).lower())
        if key is not None and row[key] is not None:
            return row[key]

    return ""


def extract_equation_like(text):
    text = clean_text(text)

    m = re.search(r"<gadget[^>]*>\s*(.*?)\s*</gadget>", text, flags=re.I | re.S)
    if m:
        return clean_text(m.group(1))

    m = re.search(r"<<([^<>]+)>>", text)
    if m:
        eq = m.group(1)
        if "=" in eq:
            eq = eq.split("=")[0]
        return clean_text(eq)

    m = re.search(
        r"(?:equation|formula|expression|program)\s*[:=]\s*([^.;\n]+)",
        text,
        flags=re.I,
    )
    if m:
        return clean_text(m.group(1))

    m = re.search(
        r"([-+]?\d+(?:\.\d+)?\s*(?:[+\-*/]\s*[-+]?\d+(?:\.\d+)?\s*)+)\s*=\s*[-+]?\d+(?:\.\d+)?",
        text,
    )
    if m:
        return clean_text(m.group(1))

    m = re.search(
        r"([-+]?\d+(?:\.\d+)?\s*(?:[+\-*/]\s*[-+]?\d+(?:\.\d+)?\s*)+)",
        text,
    )
    if m:
        return clean_text(m.group(1))

    return ""


def extract_final_number(text):
    text = clean_text(text)

    m = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)

    m = re.search(r"<result>\s*([-+]?\d+(?:\.\d+)?)\s*</result>", text, flags=re.I)
    if m:
        return m.group(1)

    nums = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    return nums[-1] if nums else ""


SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_eval_expr(expr):
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

    tree = ast.parse(expr, mode="eval")
    val = _eval(tree)
    if isinstance(val, float) and abs(val - round(val)) < 1e-9:
        val = int(round(val))
    return val


def fmt_answer(x):
    if isinstance(x, float):
        return f"{x:.4f}".rstrip("0").rstrip(".")
    return str(x)


def synthetic_row():
    kind = random.choice(
        [
            "add",
            "sub",
            "mul",
            "div",
            "percent",
            "average",
            "square",
            "sqrt",
            "order",
            "linear_easy",
            "word_add",
            "word_sub",
            "word_mul",
            "word_div",
        ]
    )

    if kind == "add":
        a, b = random.randint(0, 500), random.randint(0, 500)
        eq = f"{a} + {b}"
        q = random.choice(
            [f"what is {a} plus {b}", f"calculate {a} + {b}", f"{a} + {b}"]
        )

    elif kind == "sub":
        a, b = random.randint(0, 500), random.randint(0, 500)
        eq = f"{a} - {b}"
        q = random.choice(
            [f"what is {a} minus {b}", f"calculate {a} - {b}", f"{a} - {b}"]
        )

    elif kind == "mul":
        a, b = random.randint(0, 100), random.randint(0, 100)
        eq = f"{a} * {b}"
        q = random.choice(
            [
                f"what is {a} times {b}",
                f"calculate {a} * {b}",
                f"{a} multiplied by {b}",
                f"what is {a} × {b}",
            ]
        )

    elif kind == "div":
        b = random.randint(1, 50)
        ans = random.randint(1, 50)
        a = b * ans
        eq = f"{a} / {b}"
        q = random.choice(
            [f"what is {a} divided by {b}", f"calculate {a} / {b}", f"{a} ÷ {b}"]
        )

    elif kind == "percent":
        a = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 75])
        b = random.randint(10, 500)
        eq = f"({a} / 100) * {b}"
        q = random.choice([f"what is {a} percent of {b}", f"find {a}% of {b}"])

    elif kind == "average":
        nums = [random.randint(0, 100) for _ in range(random.randint(2, 5))]
        eq = "(" + " + ".join(map(str, nums)) + f") / {len(nums)}"
        q = f"what is the average of {', '.join(map(str, nums))}"

    elif kind == "square":
        a = random.randint(0, 30)
        eq = f"{a} ** 2"
        q = random.choice(
            [
                f"what is {a} squared",
                f"what is {a} to the power of 2",
                f"what is {a}²",
            ]
        )

    elif kind == "sqrt":
        a = random.randint(0, 30)
        n = a * a
        eq = f"{a}"
        q = random.choice([f"what is the square root of {n}", f"what is √{n}"])

    elif kind == "order":
        a, b, c = random.randint(1, 30), random.randint(1, 30), random.randint(1, 30)
        eq = f"{a} + {b} * {c}"
        q = f"what is {a} plus {b} times {c}"

    elif kind == "linear_easy":
        x = random.randint(-20, 20)
        a = random.randint(1, 12)
        b = random.randint(-30, 30)
        c = a * x + b
        eq = f"({c} - {b}) / {a}"
        q = f"solve for x: {a}x + {b} = {c}"

    elif kind == "word_add":
        a, b = random.randint(1, 50), random.randint(1, 50)
        item = random.choice(["apples", "coins", "cards", "stickers"])
        eq = f"{a} + {b}"
        q = f"I have {a} {item} and get {b} more. How many {item} do I have?"

    elif kind == "word_sub":
        a = random.randint(5, 80)
        b = random.randint(1, a)
        item = random.choice(["apples", "coins", "cards", "stickers"])
        eq = f"{a} - {b}"
        q = f"I have {a} {item} and give away {b}. How many {item} are left?"

    elif kind == "word_mul":
        a, b = random.randint(2, 20), random.randint(2, 20)
        item = random.choice(["cookies", "fish", "books", "marbles"])
        group = random.choice(["boxes", "bags", "groups", "cats"])
        eq = f"{a} * {b}"
        q = f"There are {a} {group} with {b} {item} each. How many {item} are there?"

    else:
        b = random.randint(2, 20)
        ans = random.randint(2, 20)
        a = b * ans
        item = random.choice(["cookies", "fish", "books", "marbles"])
        eq = f"{a} / {b}"
        q = f"{a} {item} are split equally into {b} groups. How many {item} are in each group?"

    eq = normalize_equation(eq)

    try:
        answer = fmt_answer(safe_eval_expr(eq))
    except Exception:
        answer = ""

    return {
        "source": "synthetic",
        "question": clean_text(q),
        "equation": eq,
        "answer": answer,
        "template": kind,
    }


def make_synthetic_pool(n):
    rows = []
    seen = set()
    while len(rows) < n:
        row = synthetic_row()
        key = (row["question"].lower(), row["equation"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    return rows


def load_gsm8k_rows():
    rows = []
    try:
        ds = load_dataset("openai/gsm8k", "main", split="train")
    except Exception as e:
        print("[warn] failed GSM8K:", e, flush=True)
        return rows

    for r in ds:
        q = clean_text(r.get("question", ""))
        ans = clean_text(r.get("answer", ""))

        final_answer = extract_final_number(ans)

        equations = re.findall(r"<<([^<>]+)>>", ans)
        equation = equations[-1] if equations else ""

        if "=" in equation:
            equation = equation.split("=")[0].strip()

        if not equation:
            equation = extract_equation_like(ans)

        if not equation:
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?", q)
            equation = " ".join(nums[:4])

        if q and final_answer:
            rows.append(
                {
                    "source": "gsm8k",
                    "question": q,
                    "equation": normalize_equation(equation),
                    "answer": final_answer,
                    "template": "gsm8k",
                    "raw_solution": ans,
                }
            )

    return rows


def load_mathqa_rows():
    rows = []

    candidates = [
        ("allenai/math_qa", "train"),
        ("qwedsacf/grade-school-math-instructions", "train"),
    ]

    ds = None

    for name, split in candidates:
        try:
            print(f"[try] loading MathQA from {name}", flush=True)
            ds = load_dataset(name, split=split)
            print(f"[ok] loaded MathQA from {name}: {len(ds)}", flush=True)
            break
        except Exception as e:
            print(f"[warn] failed MathQA candidate {name}: {e}", flush=True)
            ds = None

    if ds is None:
        print("[warn] MathQA unavailable; continuing without it.", flush=True)
        return rows

    print("[debug] MathQA columns:", ds.column_names, flush=True)
    print("[debug] MathQA sample:", ds[0], flush=True)

    for r in ds:
        q = clean_text(
            get_field(
                r,
                [
                    "Problem",
                    "problem",
                    "question",
                    "Question",
                    "input",
                    "instruction",
                    "prompt",
                ],
            )
        )

        answer_text = clean_text(
            get_field(
                r,
                [
                    "answer",
                    "Answer",
                    "output",
                    "response",
                    "target",
                    "correct",
                    "Correct Answer",
                ],
            )
        )

        equation = clean_text(
            get_field(
                r,
                [
                    "annotated_formula",
                    "linear_formula",
                    "equation",
                    "Equation",
                    "formula",
                ],
            )
        )

        solution = clean_text(
            get_field(
                r,
                ["solution", "Solution", "rationale", "Rationale", "chain_of_thought"],
            )
        )

        combined = clean_text((solution + " " + answer_text).strip())

        if not equation:
            equation = extract_equation_like(combined)

        answer = extract_final_number(combined) or answer_text

        if q and (equation or answer):
            rows.append(
                {
                    "source": "mathqa",
                    "question": q,
                    "equation": normalize_equation(equation),
                    "answer": answer,
                    "template": "mathqa_program",
                }
            )

    print(f"[ok] parsed MathQA rows: {len(rows)}", flush=True)
    return rows


def load_svamp_rows():
    rows = []

    candidates = [("ChilleD/SVAMP", "train"), ("svamp", "train")]
    ds = None

    for name, split in candidates:
        try:
            ds = load_dataset(name, split=split)
            break
        except Exception:
            ds = None

    if ds is None:
        print("[warn] failed SVAMP", flush=True)
        return rows

    for r in ds:
        body = clean_text(get_field(r, ["Body", "body"]))
        question = clean_text(get_field(r, ["Question", "question"]))
        q = clean_text((body + " " + question).strip())
        eq = clean_text(get_field(r, ["Equation", "equation"]))
        ans = clean_text(get_field(r, ["Answer", "answer"]))

        if q and (eq or ans):
            rows.append(
                {
                    "source": "svamp",
                    "question": q,
                    "equation": normalize_equation(eq),
                    "answer": ans,
                    "template": "svamp",
                }
            )

    return rows


def load_asdiv_rows():
    rows = []

    candidates = [
        ("MU-NLPC/Calc-asdiv_a", "test"),
        ("EleutherAI/asdiv", "validation"),
    ]

    ds = None

    for name, split in candidates:
        try:
            print(f"[try] loading ASDiv from {name} split={split}", flush=True)
            ds = load_dataset(name, split=split)
            print(f"[ok] loaded ASDiv from {name}: {len(ds)}", flush=True)
            break
        except Exception as e:
            print(f"[warn] failed ASDiv candidate {name}: {e}", flush=True)
            ds = None

    if ds is None:
        print("[warn] ASDiv unavailable; continuing without it.", flush=True)
        return rows

    print("[debug] ASDiv columns:", ds.column_names, flush=True)
    print("[debug] ASDiv sample:", ds[0], flush=True)

    for r in ds:
        body = clean_text(get_field(r, ["body", "Body", "sBody", "context"]))
        question = clean_text(get_field(r, ["question", "Question", "sQuestion"]))
        q = clean_text((body + " " + question).strip())

        if not q:
            q = clean_text(get_field(r, ["problem", "Problem", "input", "prompt"]))

        eq_raw = clean_text(
            get_field(
                r,
                [
                    "formula",
                    "Formula",
                    "equation",
                    "Equation",
                    "lEquations",
                    "annotated_formula",
                    "chain",
                ],
            )
        )

        eq = extract_equation_like(eq_raw) or normalize_equation(eq_raw)

        ans = clean_text(
            get_field(
                r,
                [
                    "answer",
                    "Answer",
                    "result",
                    "result_float",
                    "lSolutions",
                    "solution",
                    "target",
                    "output",
                ],
            )
        )

        if not ans:
            ans = extract_final_number(eq_raw)

        if q and (eq or ans):
            rows.append(
                {
                    "source": "asdiv",
                    "question": q,
                    "equation": normalize_equation(eq),
                    "answer": ans,
                    "template": "asdiv",
                }
            )

    print(f"[ok] parsed ASDiv rows: {len(rows)}", flush=True)
    return rows


def load_mawps_rows():
    rows = []

    candidates = [("MU-NLPC/Calc-mawps", "train"), ("mawps", "train")]
    ds = None

    for name, split in candidates:
        try:
            ds = load_dataset(name, split=split)
            break
        except Exception:
            ds = None

    if ds is None:
        print("[warn] failed MAWPS", flush=True)
        return rows

    for r in ds:
        q = clean_text(get_field(r, ["question", "Question", "sQuestion"]))
        eq = clean_text(get_field(r, ["equation", "Equation", "lEquations"]))
        ans = clean_text(get_field(r, ["answer", "Answer", "lSolutions"]))

        if q and (eq or ans):
            rows.append(
                {
                    "source": "mawps",
                    "question": q,
                    "equation": normalize_equation(eq),
                    "answer": ans,
                    "template": "mawps",
                }
            )

    return rows


def dedupe_math_rows(rows):
    out = []
    seen = set()

    for row in rows:
        q = clean_text(row.get("question", ""))
        eq = clean_text(row.get("equation", ""))
        ans = clean_text(row.get("answer", ""))

        if not q:
            continue

        key = (q.lower(), eq.lower(), ans.lower())
        if key in seen:
            continue

        seen.add(key)
        row["question"] = q
        row["equation"] = normalize_equation(eq)
        row["answer"] = ans
        out.append(row)

    return out


def sample_with_replacement(pool, n):
    if not pool:
        return []
    return [random.choice(pool) for _ in range(n)]


def build_phase_rows(phase, pools):
    total = int(phase["rows"])
    rows = []

    for source_group, pct in phase["mix"].items():
        n = int(round(total * pct))

        if source_group == "synthetic":
            rows.extend(sample_with_replacement(pools["synthetic"], n))
        elif source_group == "mathqa":
            rows.extend(sample_with_replacement(pools["mathqa"], n))
        elif source_group == "gsm8k":
            rows.extend(sample_with_replacement(pools["gsm8k"], n))
        elif source_group == "mawps_asdiv_svamp":
            combined = pools["mawps"] + pools["asdiv"] + pools["svamp"]
            rows.extend(sample_with_replacement(combined, n))

    random.shuffle(rows)
    return dedupe_math_rows(rows)


def target_text(row):
    eq = clean_text(row.get("equation", ""))
    ans = clean_text(row.get("answer", ""))

    if eq and ans:
        return f"equation: {eq} answer: {ans}"
    if eq:
        return f"equation: {eq}"
    return f"answer: {ans}"


def build_vocabs(rows):
    input_counter = Counter()
    output_counter = Counter()

    for row in rows:
        input_counter.update(tokenize(normalize_question(row["question"])))
        output_counter.update(tokenize(target_text(row)))

    input_vocab = {s: i for i, s in enumerate(SPECIALS)}
    output_vocab = {s: i for i, s in enumerate(SPECIALS)}

    for tok, _ in input_counter.most_common(MAX_INPUT_VOCAB - len(input_vocab)):
        if tok not in input_vocab:
            input_vocab[tok] = len(input_vocab)

    for tok, _ in output_counter.most_common(MAX_OUTPUT_VOCAB - len(output_vocab)):
        if tok not in output_vocab:
            output_vocab[tok] = len(output_vocab)

    return input_vocab, output_vocab


def encode(tokens, vocab, max_len, add_bos_eos=False):
    ids = []

    if add_bos_eos:
        ids.append(BOS_ID)

    for tok in tokens:
        ids.append(vocab.get(tok, UNK_ID))
        if len(ids) >= max_len - (1 if add_bos_eos else 0):
            break

    if add_bos_eos:
        ids.append(EOS_ID)

    ids = ids[:max_len]

    while len(ids) < max_len:
        ids.append(PAD_ID)

    return ids


class MathDataset(Dataset):
    def __init__(self, rows, input_vocab, output_vocab):
        self.rows = rows
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        x_tokens = tokenize(normalize_question(row["question"]))
        y_tokens = tokenize(target_text(row))

        x = torch.tensor(
            encode(x_tokens, self.input_vocab, MAX_INPUT_LEN, add_bos_eos=True),
            dtype=torch.long,
        )

        y = torch.tensor(
            encode(y_tokens, self.output_vocab, MAX_OUTPUT_LEN, add_bos_eos=True),
            dtype=torch.long,
        )

        return x, y


class Seq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size):
        super().__init__()

        self.input_emb = nn.Embedding(input_vocab_size, EMBED, padding_idx=PAD_ID)
        self.output_emb = nn.Embedding(output_vocab_size, EMBED, padding_idx=PAD_ID)

        self.encoder = nn.GRU(EMBED, HIDDEN, batch_first=True, bidirectional=True)

        self.bridge = nn.Sequential(
            nn.Linear(HIDDEN * 2, HIDDEN),
            nn.Tanh(),
        )

        self.decoder = nn.GRU(EMBED, HIDDEN, batch_first=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.out = nn.Linear(HIDDEN, output_vocab_size)

    def encode(self, x):
        emb = self.dropout(self.input_emb(x))
        _, h = self.encoder(emb)

        h_fwd = h[-2]
        h_bwd = h[-1]

        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)
        h0 = self.bridge(h_cat).unsqueeze(0)

        return h0

    def forward(self, x, y=None, teacher_forcing=True, max_len=MAX_OUTPUT_LEN):
        batch = x.size(0)
        h = self.encode(x)

        prev = torch.full((batch, 1), BOS_ID, dtype=torch.long, device=x.device)
        logits_steps = []

        for t in range(max_len):
            emb = self.dropout(self.output_emb(prev))
            dec_out, h = self.decoder(emb, h)
            logits = self.out(dec_out[:, -1])
            logits_steps.append(logits.unsqueeze(1))

            if teacher_forcing and y is not None and t < y.size(1):
                prev = y[:, t].unsqueeze(1)
            else:
                prev = torch.argmax(logits, dim=-1, keepdim=True)

        return torch.cat(logits_steps, dim=1)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total = 0.0
    batches = 0
    token_correct = 0
    token_total = 0

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x, y, teacher_forcing=True)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        total += float(loss.item())
        batches += 1

        pred = torch.argmax(logits, dim=-1)
        mask = y != PAD_ID

        token_correct += int(((pred == y) & mask).sum().item())
        token_total += int(mask.sum().item())

    return {
        "loss": total / max(1, batches),
        "token_acc": token_correct / max(1, token_total),
    }


def split_train_val(rows):
    rows = rows[:]
    random.shuffle(rows)

    n_val = max(1, int(len(rows) * VAL_SPLIT))
    return rows[n_val:], rows[:n_val]


def save_checkpoint(path, model, input_vocab, output_vocab, phase_name, metrics):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_vocab": input_vocab,
            "output_vocab": output_vocab,
            "config": {
                "max_input_len": MAX_INPUT_LEN,
                "max_output_len": MAX_OUTPUT_LEN,
                "embed": EMBED,
                "hidden": HIDDEN,
                "dropout": DROPOUT,
                "model_type": "math_seq2seq_equation_translator",
                "pad_id": PAD_ID,
                "bos_id": BOS_ID,
                "eos_id": EOS_ID,
                "unk_id": UNK_ID,
            },
            "phase": phase_name,
            "metrics": metrics,
        },
        path,
    )


def train_phase(model, phase_name, rows, input_vocab, output_vocab, epochs):
    train_rows, val_rows = split_train_val(rows)

    train_ds = MathDataset(train_rows, input_vocab, output_vocab)
    val_ds = MathDataset(val_rows, input_vocab, output_vocab)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    best_loss = math.inf
    best_metrics = None

    for epoch in range(1, epochs + 1):
        model.train()

        total = 0.0
        batches = 0

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, y, teacher_forcing=True)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total += float(loss.item())
            batches += 1

        train_loss = total / max(1, batches)
        metrics = evaluate(model, val_loader, criterion)

        print(
            f"{phase_name} | epoch {epoch:03d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_loss {metrics['loss']:.4f} | "
            f"val_tok {metrics['token_acc']:.4f}",
            flush=True,
        )

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_metrics = metrics

            save_checkpoint(
                OUT_MODEL_DIR / f"{phase_name}_best.pt",
                model,
                input_vocab,
                output_vocab,
                phase_name,
                metrics,
            )

            print(f"[saved best] {OUT_MODEL_DIR / f'{phase_name}_best.pt'}", flush=True)

    save_checkpoint(
        OUT_MODEL_DIR / f"{phase_name}_last.pt",
        model,
        input_vocab,
        output_vocab,
        phase_name,
        best_metrics or {},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_pool", type=int, default=300000)
    parser.add_argument("--limit_public", type=int, default=0)
    parser.add_argument("--rebuild_data", action="store_true")
    args = parser.parse_args()

    print("SCRIPT STARTED", flush=True)
    print("device:", DEVICE, flush=True)

    print("Loading/building pools...", flush=True)

    synthetic_path = OUT_DATA_DIR / "synthetic_pool.jsonl"

    if synthetic_path.exists() and not args.rebuild_data:
        synthetic = load_jsonl(synthetic_path)
    else:
        synthetic = make_synthetic_pool(args.synthetic_pool)
        save_jsonl(synthetic_path, synthetic)

    mathqa = load_mathqa_rows()
    gsm8k = load_gsm8k_rows()
    mawps = load_mawps_rows()
    asdiv = load_asdiv_rows()
    svamp = load_svamp_rows()

    if args.limit_public:
        mathqa = mathqa[: args.limit_public]
        gsm8k = gsm8k[: args.limit_public]
        mawps = mawps[: args.limit_public]
        asdiv = asdiv[: args.limit_public]
        svamp = svamp[: args.limit_public]

    pools = {
        "synthetic": dedupe_math_rows(synthetic),
        "mathqa": dedupe_math_rows(mathqa),
        "gsm8k": dedupe_math_rows(gsm8k),
        "mawps": dedupe_math_rows(mawps),
        "asdiv": dedupe_math_rows(asdiv),
        "svamp": dedupe_math_rows(svamp),
    }

    for name, pool in pools.items():
        print(f"{name}: {len(pool)}", flush=True)
        save_jsonl(OUT_DATA_DIR / f"{name}.jsonl", pool)

    all_phase_rows = []
    phase_rows_by_name = {}

    print("Building blended phase files...", flush=True)

    for phase in PHASES:
        rows = build_phase_rows(phase, pools)
        phase_rows_by_name[phase["name"]] = rows
        all_phase_rows.extend(rows)

        path = OUT_DATA_DIR / f"{phase['name']}.jsonl"
        save_jsonl(path, rows)

        print(f"{phase['name']}: {len(rows)} saved to {path}", flush=True)

    print("Building vocab from all phase rows...", flush=True)

    input_vocab, output_vocab = build_vocabs(all_phase_rows)

    save_json(OUT_MODEL_DIR / "input_vocab.json", input_vocab)
    save_json(OUT_MODEL_DIR / "output_vocab.json", output_vocab)
    save_json(OUT_MODEL_DIR / "phases.json", PHASES)

    print("input vocab:", len(input_vocab), flush=True)
    print("output vocab:", len(output_vocab), flush=True)

    model = Seq2Seq(len(input_vocab), len(output_vocab)).to(DEVICE)

    for phase in PHASES:
        print("=" * 80, flush=True)
        print("TRAINING", phase["name"], flush=True)
        print("=" * 80, flush=True)

        train_phase(
            model=model,
            phase_name=phase["name"],
            rows=phase_rows_by_name[phase["name"]],
            input_vocab=input_vocab,
            output_vocab=output_vocab,
            epochs=phase["epochs"],
        )

    save_checkpoint(
        OUT_MODEL_DIR / "math_equation_translator_final.pt",
        model,
        input_vocab,
        output_vocab,
        "final",
        {},
    )

    print("DONE", flush=True)
    print("datasets saved in:", OUT_DATA_DIR, flush=True)
    print("checkpoints saved in:", OUT_MODEL_DIR, flush=True)


if __name__ == "__main__":
    main()
