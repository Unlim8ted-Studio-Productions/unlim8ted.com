import argparse, json, math, re, ast, operator
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GENERAL_MODEL_DIR = Path("assets/models/general_cover_chunks_noisy_continue")
MATH_MODEL_PATH = Path(
    "assets/models/math_equation_translator/math_equation_translator_final.pt"
)
MATH_CLASSIFIER_DIR = Path("assets/models/math_classifier")

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

KNOWN_SUBJECTS = {
    "unlim8ted": "Unlim8ted",
    "unlimited": "Unlim8ted",
    "unlimted": "Unlim8ted",
    "timecat": "TimeCat",
    "time cat": "TimeCat",
    "tmecat": "TimeCat",
    "cat game": "TimeCat",
    "the glitch": "The Glitch",
    "glitch": "The Glitch",
    "gltich": "The Glitch",
    "meatball ai": "Meatball AI",
    "meatball": "Meatball",
    "meat ball": "Meatball",
    "dogs": "dogs",
    "dog": "dogs",
    "cats": "cats",
    "cat": "cats",
}


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_for_planning(text):
    text = str(text).lower().replace("\n", " ")
    text = re.sub(r"[!?.,:;\"'`“”‘’()\[\]{}]", " ", text)
    text = re.sub(r"[^a-z0-9_+\-*/% -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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


def vectorize_general_question(question, input_vocab):
    feats = make_input_ngrams(input_tokenize(f"question: {question}"))
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


def decode_general_ids(ids, output_chunks):
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
        for w in self.parameters():
            nn.init.uniform_(w, -stdv, stdv)

    def forward(self, x, h):
        gi = torch.matmul(x, self.weight_ih.t()) + self.bias_ih
        gh = torch.matmul(h, self.weight_hh.t()) + self.bias_hh
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return n + z * (h - n)


class GeneralChunkModel(nn.Module):
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

    def forward(self, x, max_len=MAX_OUTPUT_CHUNKS + 1):
        batch = x.size(0)
        prompt = self.encoder(x)
        hidden = torch.zeros(batch, HIDDEN_SIZE, device=x.device)
        prev = torch.full((batch,), BOS_ID, dtype=torch.long, device=x.device)
        steps = []

        for _ in range(max_len):
            emb = self.embedding(prev)
            hidden = self.decoder_cell(torch.cat([emb, prompt], dim=-1), hidden)
            logits = self.output(torch.cat([prompt, hidden], dim=-1))
            steps.append(logits.unsqueeze(1))
            prev = torch.argmax(logits, dim=-1)

        return torch.cat(steps, dim=1)


def load_general(model_dir):
    model_dir = Path(model_dir)
    input_vocab = load_json(model_dir / "input_vocab.json")
    output_chunks = load_json(model_dir / "output_chunks.json")
    ckpt = torch.load(model_dir / "model.pt", map_location=DEVICE)

    model = GeneralChunkModel(len(input_vocab), len(output_chunks)).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, input_vocab, output_chunks


@torch.no_grad()
def generate_general(question, model, input_vocab, output_chunks):
    x = vectorize_general_question(question, input_vocab).to(DEVICE)
    logits = model(x)
    ids = torch.argmax(logits[0], dim=-1).detach().cpu().tolist()
    return decode_general_ids(ids, output_chunks)


def find_subjects(text):
    q = normalize_for_planning(text)
    found = []
    for alias, canonical in sorted(
        KNOWN_SUBJECTS.items(), key=lambda x: len(x[0]), reverse=True
    ):
        if re.search(rf"\b{re.escape(alias)}\b", q):
            if canonical not in found:
                found.append(canonical)
    return found


def subject_finder(text, memory_subject=None):
    subjects = find_subjects(text)
    return subjects[0] if subjects else memory_subject


def subject_inserter(text, subject):
    if not subject:
        return text

    if find_subjects(text):
        return text

    q = text.strip()
    q = re.sub(r"\bit\b", subject, q, flags=re.I)
    q = re.sub(r"\bits\b", f"{subject}'s", q, flags=re.I)
    q = re.sub(r"\bthey\b", subject, q, flags=re.I)
    q = re.sub(r"\bthem\b", subject, q, flags=re.I)
    q = re.sub(r"\btheir\b", f"{subject}'s", q, flags=re.I)

    low = normalize_for_planning(q)
    if low.startswith(
        (
            "who ",
            "what ",
            "where ",
            "when ",
            "why ",
            "how ",
            "does ",
            "do ",
            "is ",
            "are ",
            "can ",
        )
    ):
        return f"{q} about {subject}"

    return q


def is_compare_question(text):
    q = normalize_for_planning(text)
    return bool(
        re.search(r"\b(compare|contrast|vs|versus)\b", q)
        or "difference between" in q
        or "differences between" in q
    )


def extract_compare_subjects(text):
    q = normalize_for_planning(text)
    patterns = [
        r"(?:compare|contrast)\s+(.+?)\s+(?:and|with|to|vs|versus)\s+(.+)",
        r"(?:difference between|differences between)\s+(.+?)\s+and\s+(.+)",
        r"is there a difference between\s+(.+?)\s+and\s+(.+)",
        r"(.+?)\s+(?:vs|versus)\s+(.+)",
    ]

    for pat in patterns:
        m = re.search(pat, q)
        if m:
            a = re.sub(r"^(hi|hey|hello)\s+", "", m.group(1).strip())
            b = m.group(2).strip()
            if a and b:
                return [a, b]
    return []


def is_list_question(text):
    q = normalize_for_planning(text)
    if re.search(r"\bfacts about\b", q):
        return True
    if re.search(
        r"\b(list|give me|show me)\b.*\b(facts|examples|features|types|projects|things)\b",
        q,
    ):
        return True
    if re.search(r"\bwhat are some\b", q):
        return True
    if any(
        w in set(q.split())
        for w in {"facts", "examples", "features", "types", "projects"}
    ):
        return True
    return False


def split_multi_question(text, subject):
    if is_compare_question(text):
        return [text]

    parts = re.split(r"\s+\band\b\s+|\s+\balso\b\s+|\s+\bplus\b\s+|;", text, flags=re.I)
    parts = [p.strip(" ?.!,") for p in parts if p.strip(" ?.!,")]

    if len(parts) <= 1:
        return [text]

    return [subject_inserter(p, subject) for p in parts]


class MathClassifier(nn.Module):
    def __init__(self, input_size, hidden=384, dropout=0.25):
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
            nn.Linear(hidden // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


def classifier_features(text):
    text = normalize_for_planning(text)
    feats = []
    s = f"<{text}>"
    for n in (2, 3, 4):
        for i in range(len(s) - n + 1):
            feats.append("c:" + s[i : i + n])
    words = text.split()
    for n in (1, 2, 3):
        for i in range(len(words) - n + 1):
            feats.append("w:" + "_".join(words[i : i + n]))
    return feats


def vectorize_classifier(text, vocab):
    x = torch.zeros(1, len(vocab), dtype=torch.float32)
    counts = Counter(classifier_features(text))
    for feat, count in counts.items():
        idx = vocab.get(feat, 0)
        x[0, idx] = min(float(count), 5.0)
    return x


def load_math_classifier(model_dir):
    model_dir = Path(model_dir)
    vocab = load_json(model_dir / "input_vocab.json")
    ckpt = torch.load(model_dir / "math_classifier.pt", map_location=DEVICE)
    model = MathClassifier(len(vocab)).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, vocab


@torch.no_grad()
def classify_math(text, model, vocab):
    x = vectorize_classifier(text, vocab).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=-1)[0]
    return int(torch.argmax(probs).item()), float(probs[1].item())


# ---------- Math model ----------


def math_replace_symbols(text):
    text = str(text)
    for old, new in {
        "×": "*",
        "÷": "/",
        "−": "-",
        "²": " ** 2 ",
        "³": " ** 3 ",
        "√": " sqrt ",
    }.items():
        text = text.replace(old, new)
    return text


def math_normalize_question(q):
    q = math_replace_symbols(q).lower().replace("\n", " ")
    reps = [
        (r"\btimes\b", " * "),
        (r"\bplus\b", " + "),
        (r"\bminus\b", " - "),
        (r"\bdivided\s+by\b", " / "),
        (r"\bsquared\b", " ^ 2 "),
        (r"\bcubed\b", " ^ 3 "),
    ]
    for p, r in reps:
        q = re.sub(p, r, q)
    q = re.sub(r"[^a-z0-9_+\-*/^().,?:;$%=\s']", " ", q)
    return re.sub(r"\s+", " ", q).strip()


def math_tokenize(text):
    text = math_replace_symbols(str(text).lower())
    return re.findall(r"\d+\.\d+|\d+|\*\*|sqrt|pi|[a-z_]+|[+\-*/^=().,?:;$%]", text)


def math_detok(tokens):
    out = ""
    for t in tokens:
        if t in {".", ",", "?", "!", ":", ";", "%", ")"}:
            out = out.rstrip() + t
        elif t in {
            "(",
        }:
            if out and not out.endswith(" "):
                out += " "
            out += t
        elif t in {"+", "-", "*", "/", "**", "^", "="}:
            out += f" {t} "
        else:
            if out and not out.endswith((" ", "(")):
                out += " "
            out += t
    return re.sub(r"\s+", " ", out).strip()


class MathSeq2Seq(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embed, hidden, dropout):
        super().__init__()
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
        return self.bridge(torch.cat([h[-2], h[-1]], dim=-1)).unsqueeze(0)

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


def math_encode(tokens, vocab, max_len):
    ids = [BOS_ID]
    for t in tokens:
        ids.append(vocab.get(t, UNK_ID))
        if len(ids) >= max_len - 1:
            break
    ids.append(EOS_ID)
    ids = ids[:max_len]
    while len(ids) < max_len:
        ids.append(PAD_ID)
    return ids


def math_decode(ids, id_to_token):
    toks = []
    for idx in ids:
        idx = int(idx)
        if idx == EOS_ID:
            break
        if idx in {PAD_ID, BOS_ID, UNK_ID}:
            continue
        toks.append(id_to_token.get(idx, ""))
    return math_detok(toks)


def load_math_model(path):
    ckpt = torch.load(path, map_location=DEVICE)
    cfg = ckpt["config"]
    input_vocab = ckpt["input_vocab"]
    output_vocab = ckpt["output_vocab"]
    id_to = {int(v): k for k, v in output_vocab.items()}

    model = MathSeq2Seq(
        len(input_vocab), len(output_vocab), cfg["embed"], cfg["hidden"], cfg["dropout"]
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return model, input_vocab, id_to, cfg


def extract_equation_and_answer(text):
    eq = ""
    ans = ""
    m = re.search(r"equation\s*:\s*(.*?)(?:\s+answer\s*:|$)", text, flags=re.I)
    if m:
        eq = m.group(1).strip()
    m = re.search(r"answer\s*:\s*(.*)$", text, flags=re.I)
    if m:
        ans = m.group(1).strip()
    return eq, ans


SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def safe_eval_expr(expr):
    expr = expr.strip().replace("^", "**")
    if not expr or not re.fullmatch(r"[0-9+\-*/().\s*]+", expr):
        raise ValueError("unsafe")

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Num):
            return node.n
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            return SAFE_OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return SAFE_OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsafe")

    val = ev(ast.parse(expr, mode="eval"))
    if isinstance(val, float) and abs(val - round(val)) < 1e-9:
        val = int(round(val))
    return val


@torch.no_grad()
def answer_math(question, model, input_vocab, id_to, cfg):
    q = math_normalize_question(question)
    x = torch.tensor(
        [math_encode(math_tokenize(q), input_vocab, cfg["max_input_len"])],
        dtype=torch.long,
        device=DEVICE,
    )
    ids = model(x, cfg["max_output_len"])[0].detach().cpu().tolist()
    decoded = math_decode(ids, id_to)
    eq, pred_ans = extract_equation_and_answer(decoded)

    computed = ""
    if eq:
        try:
            computed = str(safe_eval_expr(eq))
        except Exception:
            computed = ""

    return {
        "normalized": q,
        "decoded": decoded,
        "equation": eq,
        "predicted_answer": pred_ans,
        "computed_answer": computed,
        "final": computed or pred_ans or decoded,
    }


class Memory:
    def __init__(self):
        self.subject = None


def print_step(name, value):
    print(f"\n[{name}]")
    if isinstance(value, (dict, list)):
        print(json.dumps(value, indent=2, ensure_ascii=False))
    else:
        print(value)


def answer(question, runtime, memory):
    general_model, general_vocab, general_chunks = runtime["general"]
    math_classifier, math_classifier_vocab = runtime["math_classifier"]
    math_model, math_vocab, math_id_to, math_cfg = runtime["math_model"]

    print_step("RAW_INPUT", question)

    math_label, math_prob = classify_math(
        question, math_classifier, math_classifier_vocab
    )
    print_step(
        "MATH_CLASSIFIER",
        {
            "label": "math" if math_label == 1 else "general",
            "math_probability": math_prob,
        },
    )

    if math_label == 1 and math_prob >= 0.55:
        m = answer_math(question, math_model, math_vocab, math_id_to, math_cfg)
        print_step("MATH_NORMALIZED", m["normalized"])
        print_step("MATH_DECODED", m["decoded"])
        print_step("MATH_EQUATION", m["equation"])
        print_step("MATH_COMPUTED", m["computed_answer"])
        return m["final"]

    subject = subject_finder(question, memory.subject)
    print_step(
        "SUBJECT_FINDER", {"found_subject": subject, "previous_subject": memory.subject}
    )

    if subject:
        memory.subject = subject

    if is_compare_question(question):
        comps = extract_compare_subjects(question)
        print_step("COMPARE_DETECTOR", {"is_compare": True, "subjects": comps})
        if len(comps) >= 2:
            return f"Comparing {comps[0]} and {comps[1]} would require thinking about two things at once, and this tiny meatball brain might explode."
        return "That is a compare question, but I am not sure what two things you want compared."

    list_mode = is_list_question(question)
    print_step("LIST_DETECTOR", {"is_list": list_mode})

    rewritten = subject_inserter(question, memory.subject)
    print_step("SUBJECT_INSERTER", rewritten)

    subquestions = split_multi_question(rewritten, memory.subject)
    print_step("MULTI_PART_SPLITTER", subquestions)

    answers = []
    for subq in subquestions:
        a = generate_general(subq, general_model, general_vocab, general_chunks)
        print_step("GENERATOR_CALL", {"subquestion": subq, "answer": a})
        answers.append(a)

    final = " ".join(a for a in answers if a).strip()

    if list_mode:
        parts = re.split(r"(?<=[.!?])\s+", final)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            final = "\n".join(f"- {p}" for p in parts[:8])

    print_step("FINAL_COMBINER", final)
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--general_model_dir", default=str(GENERAL_MODEL_DIR))
    parser.add_argument("--math_model", default=str(MATH_MODEL_PATH))
    parser.add_argument("--math_classifier_dir", default=str(MATH_CLASSIFIER_DIR))
    parser.add_argument("--question", default=None)
    args = parser.parse_args()

    runtime = {
        "general": load_general(args.general_model_dir),
        "math_classifier": load_math_classifier(args.math_classifier_dir),
        "math_model": load_math_model(args.math_model),
    }

    print("Loaded runtime.")
    print("Device:", DEVICE)

    memory = Memory()

    if args.question:
        final = answer(args.question, runtime, memory)
        print("\nANSWER:\n" + final)
        return

    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"quit", "exit", "stop"}:
            break
        if not q:
            continue
        final = answer(q, runtime, memory)
        print("\nANSWER:\n" + final)


if __name__ == "__main__":
    main()
