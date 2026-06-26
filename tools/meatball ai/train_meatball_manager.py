# train_meatball_manager.py
# Generates manager training data + trains manager model.
#
# Outputs:
# assets/data/manager/manager_train.jsonl
# assets/models/meatball_manager/manager.pt
# assets/models/meatball_manager/input_vocab.json
# assets/models/meatball_manager/labels.json
# assets/models/meatball_manager/config.json
#
# Run:
# python train_meatball_manager.py

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OUT_DATA_DIR = Path("assets/data/manager")
OUT_MODEL_DIR = Path("assets/models/meatball_manager")

SPECIALIZED_DIR = Path("assets/data/specialized_QA")
SMART_QA_PATH = Path("tools/SmartMeatballQA.jsonl")
MATH_DATA_DIR = Path("assets/data/math")

CHAR_NGRAMS = (2, 3, 4, 5)
WORD_NGRAMS = (1, 2, 3)
HISTORY_WORD_NGRAMS = (1, 2)

MAX_VOCAB = 24000
BATCH_SIZE = 128
EPOCHS = 45
PATIENCE = 9
VAL_SPLIT = 0.12

LR = 8e-4
WEIGHT_DECAY = 2e-3
GRAD_CLIP = 1.0
DROPOUT = 0.25
HIDDEN = 640

INTENTS = [
    "normal_qa",
    "math",
    "list",
    "compare",
    "multi_part",
    "smalltalk",
    "unknown",
]

SUBJECT_ACTIONS = [
    "keep",
    "update",
    "insert",
    "clear",
]

HISTORY_ACTIONS = [
    "ignore",
    "use",
]

EMOTIONS = [
    "neutral",
    "excited",
    "confused",
    "suspicious",
    "angry",
    "sad",
    "overwhelmed",
]

ANIMATIONS = [
    "neutral",
    "excited",
    "confused",
    "suspicious",
    "angry",
    "sad",
    "overwhelmed",
    "sauce_attack_cutscene",
]

ANSWER_STYLES = [
    "normal",
    "list",
    "short",
    "fallback",
]

KNOWN_SUBJECTS = {
    "NONE": ["none"],
    "Unlim8ted": [
        "unlim8ted",
        "unlimited",
        "unlimted",
        "unlim8ed",
        "unlim8ted studios",
        "unlimited studios",
    ],
    "TimeCat": [
        "timecat",
        "time cat",
        "time-cat",
        "tmecat",
        "cat game",
        "the cat game",
    ],
    "The Glitch": [
        "the glitch",
        "glitch",
        "gltich",
        "glotch",
        "glitc",
    ],
    "Meatball": [
        "meatball",
        "meat ball",
        "meetball",
        "meatbal",
    ],
    "Meatball AI": [
        "meatball ai",
        "meat ball ai",
        "meetball ai",
        "meatball bot",
        "meatball assistant",
    ],
    "dogs": ["dog", "dogs"],
    "cats": ["cat", "cats"],
}

SUBJECTS = list(KNOWN_SUBJECTS.keys())

SMALLTALK = [
    "hi",
    "hello",
    "hey",
    "how are you",
    "what are you",
    "are you awake",
    "good morning",
    "good night",
]

COMPARE_TEMPLATES = [
    "compare {a} and {b}",
    "{a} vs {b}",
    "what is the difference between {a} and {b}",
    "is there a difference between {a} and {b}",
]

LIST_TEMPLATES = [
    "facts about {s}",
    "list facts about {s}",
    "give me examples of {s}",
    "what are some facts about {s}",
    "show me features of {s}",
]

NORMAL_TEMPLATES = [
    "what is {s}",
    "tell me about {s}",
    "who made {s}",
    "does {s} have lore",
    "explain {s}",
]

FOLLOWUP_TEMPLATES = [
    "what is it",
    "who made it",
    "does it have clothes",
    "what about that",
    "tell me more",
    "why does it matter",
]

MULTI_TEMPLATES = [
    "what is {s} and who made it",
    "tell me about {s} and does it have lore",
    "what is {s} also who made it",
]

MATH_TEMPLATES = [
    "what is {a} + {b}",
    "calculate {a} times {b}",
    "what is {a} percent of {b}",
    "solve for x {a}x + {b} = {c}",
    "if I have {a} apples and get {b} more how many apples",
]

EMOTION_RULES_BY_SUBJECT = {
    "The Glitch": ("suspicious", "suspicious"),
    "Meatball": ("excited", "excited"),
    "Meatball AI": ("neutral", "neutral"),
    "TimeCat": ("excited", "excited"),
    "Unlim8ted": ("neutral", "neutral"),
}


random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize(text):
    text = str(text).lower().replace("\n", " ")
    text = re.sub(r"[^a-z0-9_+\-*/^().,?:;$%=\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_no_punc(text):
    text = normalize(text)
    text = re.sub(r"[!?.,:;\"'`()\[\]{}]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def find_subject(text):
    q = normalize_no_punc(text)

    best = "NONE"
    best_len = 0

    for canonical, aliases in KNOWN_SUBJECTS.items():
        if canonical == "NONE":
            continue

        terms = [canonical.lower()] + [a.lower() for a in aliases]

        for term in terms:
            if re.search(rf"\b{re.escape(term)}\b", q):
                if len(term) > best_len:
                    best = canonical
                    best_len = len(term)

    return best


def is_math_text(text):
    q = normalize(text)

    if re.search(r"\d+\s*[+\-*/]\s*\d+", q):
        return True

    if any(
        w in q.split()
        for w in ["calculate", "solve", "equation", "percent", "times", "minus", "plus"]
    ):
        return True

    if re.search(r"\b\d+\s*x\b|\bx\s*[+\-=]", q):
        return True

    return False


def is_compare_text(text):
    q = normalize_no_punc(text)
    return bool(
        re.search(r"\b(compare|contrast|vs|versus)\b", q)
        or "difference between" in q
        or "differences between" in q
    )


def is_list_text(text):
    q = normalize_no_punc(text)
    if "facts about" in q:
        return True
    if re.search(
        r"\b(list|show me|give me)\b.*\b(facts|examples|features|types|projects|things)\b",
        q,
    ):
        return True
    if "what are some" in q:
        return True
    return False


def load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_multi_text(text):
    q = normalize_no_punc(text)

    if is_compare_text(q):
        return False

    return bool(
        re.search(r"\b(and|also|plus)\b", q)
        and re.search(r"\b(what|who|does|tell|explain)\b", q)
    )


def is_smalltalk_text(text):
    q = normalize_no_punc(text)
    return q in SMALLTALK or q in {"yo", "sup"}


def choose_emotion(subject, intent):
    if intent == "math":
        return "confused", "confused"

    if intent == "compare":
        return "overwhelmed", "overwhelmed"

    if intent == "unknown":
        return "confused", "confused"

    if intent == "smalltalk":
        return "neutral", "neutral"

    if subject in EMOTION_RULES_BY_SUBJECT:
        return EMOTION_RULES_BY_SUBJECT[subject]

    return "neutral", "neutral"


def label_row(
    text,
    previous_subjects=None,
    history=None,
    previous_intent="normal_qa",
    previous_animation="neutral",
):
    previous_subjects = previous_subjects or []
    history = history or []

    subject = find_subject(text)
    prev_primary = previous_subjects[-1] if previous_subjects else "NONE"

    if is_math_text(text):
        intent = "math"
    elif is_compare_text(text):
        intent = "compare"
    elif is_list_text(text):
        intent = "list"
    elif is_multi_text(text):
        intent = "multi_part"
    elif is_smalltalk_text(text):
        intent = "smalltalk"
    elif not text.strip():
        intent = "unknown"
    else:
        intent = "normal_qa"

    if intent == "smalltalk":
        primary_subject = prev_primary
        secondary_subject = "NONE"
        subject_action = "keep"
        history_action = "ignore"

    elif subject != "NONE":
        primary_subject = subject
        secondary_subject = "NONE"
        subject_action = "update"
        history_action = "use" if history else "ignore"

    elif prev_primary != "NONE" and re.search(
        r"\b(it|its|they|them|that|this|more)\b", normalize_no_punc(text)
    ):
        primary_subject = prev_primary
        secondary_subject = "NONE"
        subject_action = "insert"
        history_action = "use"

    else:
        primary_subject = "NONE"
        secondary_subject = "NONE"
        subject_action = "keep"
        history_action = "ignore"

    if intent == "compare":
        # try rough second subject extraction
        subjects = []
        for s in SUBJECTS:
            if s != "NONE" and s != primary_subject:
                for alias in KNOWN_SUBJECTS[s]:
                    if re.search(
                        rf"\b{re.escape(alias.lower())}\b", normalize_no_punc(text)
                    ):
                        subjects.append(s)
                        break
        if primary_subject == "NONE" and subjects:
            primary_subject = subjects[0]
        if len(subjects) >= 2:
            secondary_subject = subjects[1]
        elif len(subjects) == 1 and subjects[0] != primary_subject:
            secondary_subject = subjects[0]
        else:
            secondary_subject = "NONE"

    emotion, animation = choose_emotion(primary_subject, intent)

    if "sauce attack" in normalize_no_punc(
        text
    ) or "forbidden pasta" in normalize_no_punc(text):
        animation = "sauce_attack_cutscene"
        emotion = "angry"

    if intent == "list":
        answer_style = "list"
    elif intent in {"math", "smalltalk"}:
        answer_style = "short"
    elif intent in {"unknown", "compare"}:
        answer_style = "fallback"
    else:
        answer_style = "normal"

    confidence = 0.92
    if intent == "unknown":
        confidence = 0.35
    elif subject_action == "insert":
        confidence = 0.78
    elif intent == "compare":
        confidence = 0.7

    return {
        "input": text,
        "history": history[-6:],
        "previous_subjects": previous_subjects[-5:],
        "previous_intent": previous_intent,
        "previous_animation": previous_animation,
        "intent": intent,
        "primary_subject": primary_subject,
        "secondary_subject": secondary_subject,
        "subject_action": subject_action,
        "history_action": history_action,
        "emotion": emotion,
        "animation": animation,
        "answer_style": answer_style,
        "confidence": confidence,
    }


def load_project_questions():
    questions = []

    for path in sorted(SPECIALIZED_DIR.glob("*.jsonl")):
        for r in load_jsonl(path):
            q = str(r.get("question", "")).strip()
            if q:
                questions.append(q)

    for r in load_jsonl(SMART_QA_PATH):
        q = str(r.get("question", "")).strip()
        if q:
            questions.append(q)

    return questions


def load_math_questions(max_rows=20000):
    questions = []
    seen = set()

    preferred_files = [
        "synthetic_pool.jsonl",
        "gsm8k.jsonl",
        "mathqa.jsonl",
        "mawps.jsonl",
        "asdiv.jsonl",
        "svamp.jsonl",
    ]

    for filename in preferred_files:
        path = MATH_DATA_DIR / filename
        if not path.exists():
            continue

        for r in load_jsonl(path):
            q = str(r.get("question", "")).strip()
            if not q:
                continue

            key = q.lower()
            if key in seen:
                continue

            seen.add(key)
            questions.append(q)

            if len(questions) >= max_rows:
                return questions

    return questions


def random_subject():
    return random.choice([s for s in SUBJECTS if s != "NONE"])


def generate_synthetic_rows(n=120000):
    rows = []

    for _ in range(n):
        kind = random.choice(
            [
                "normal",
                "list",
                "compare",
                "multi",
                "followup",
                "math",
                "smalltalk",
                "unknown",
                "animation",
            ]
        )

        previous_subjects = []
        history = []
        prev_intent = random.choice(INTENTS)
        prev_animation = random.choice(ANIMATIONS)

        if random.random() < 0.55:
            ps = random_subject()
            previous_subjects = [ps]
            history = [
                f"what is {ps}",
                f"{ps} is something the meatball brain remembers.",
            ]

        if kind == "normal":
            s = random_subject()
            text = random.choice(NORMAL_TEMPLATES).format(
                s=random.choice(KNOWN_SUBJECTS[s])
            )

        elif kind == "list":
            s = random_subject()
            text = random.choice(LIST_TEMPLATES).format(
                s=random.choice(KNOWN_SUBJECTS[s])
            )

        elif kind == "compare":
            a = random_subject()
            b = random_subject()
            while b == a:
                b = random_subject()
            text = random.choice(COMPARE_TEMPLATES).format(
                a=random.choice(KNOWN_SUBJECTS[a]),
                b=random.choice(KNOWN_SUBJECTS[b]),
            )

        elif kind == "multi":
            s = random_subject()
            text = random.choice(MULTI_TEMPLATES).format(
                s=random.choice(KNOWN_SUBJECTS[s])
            )

        elif kind == "followup":
            if not previous_subjects:
                previous_subjects = [random_subject()]
            text = random.choice(FOLLOWUP_TEMPLATES)

        elif kind == "math":
            a = random.randint(1, 200)
            b = random.randint(1, 200)
            c = random.randint(1, 500)
            text = random.choice(MATH_TEMPLATES).format(a=a, b=b, c=c)

        elif kind == "smalltalk":
            text = random.choice(SMALLTALK)

        elif kind == "animation":
            text = random.choice(
                [
                    "sauce attack",
                    "do the forbidden pasta technique",
                    "get angry",
                    "why is the sauce suspicious",
                    "too many questions",
                ]
            )

        else:
            text = random.choice(["", "???", "asdf qwer", "what"])

        rows.append(
            label_row(text, previous_subjects, history, prev_intent, prev_animation)
        )

    return rows


def generate_dataset(limit_project=0, synthetic_rows=120000):
    rows = []

    for q in load_project_questions():
        rows.append(label_row(q))
        if limit_project and len(rows) >= limit_project:
            break

    for q in load_math_questions():
        rows.append(label_row(q))

    rows.extend(generate_synthetic_rows(synthetic_rows))

    random.shuffle(rows)
    return rows


def make_features(row):
    feats = []

    current = normalize(row["input"])
    history = " ".join(normalize(x) for x in row.get("history", []))
    prev_subjects = row.get("previous_subjects", [])
    previous_intent = row.get("previous_intent", "normal_qa")
    previous_animation = row.get("previous_animation", "neutral")

    s = f"<{current}>"
    for n in CHAR_NGRAMS:
        for i in range(len(s) - n + 1):
            feats.append("cur_char:" + s[i : i + n])

    words = current.split()
    for n in WORD_NGRAMS:
        for i in range(len(words) - n + 1):
            feats.append("cur_word:" + "_".join(words[i : i + n]))

    hwords = history.split()
    for n in HISTORY_WORD_NGRAMS:
        for i in range(len(hwords) - n + 1):
            feats.append("hist_word:" + "_".join(hwords[i : i + n]))

    for ps in prev_subjects:
        feats.append("prev_subject:" + ps)

    feats.append("prev_intent:" + previous_intent)
    feats.append("prev_animation:" + previous_animation)

    if re.search(r"\d", current):
        feats.append("flag:has_number")
    if re.search(r"[+\-*/=]", current):
        feats.append("flag:has_operator")
    if "?" in row["input"]:
        feats.append("flag:question_mark")

    return feats


def build_vocab(rows):
    counter = Counter()
    for row in rows:
        counter.update(make_features(row))

    vocab = {"<UNK>": 0}
    for feat, _ in counter.most_common(MAX_VOCAB - 1):
        vocab[feat] = len(vocab)

    return vocab


def vectorize(row, vocab):
    x = torch.zeros(len(vocab), dtype=torch.float32)
    counts = Counter(make_features(row))

    for feat, count in counts.items():
        idx = vocab.get(feat, 0)
        x[idx] = min(float(count), 5.0)

    return x


class ManagerDataset(Dataset):
    def __init__(self, rows, vocab, maps):
        self.rows = rows
        self.vocab = vocab
        self.maps = maps

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        x = vectorize(row, self.vocab)

        return (
            x,
            torch.tensor(self.maps["intent"][row["intent"]], dtype=torch.long),
            torch.tensor(
                self.maps["primary_subject"][row["primary_subject"]], dtype=torch.long
            ),
            torch.tensor(
                self.maps["secondary_subject"][row["secondary_subject"]],
                dtype=torch.long,
            ),
            torch.tensor(
                self.maps["subject_action"][row["subject_action"]], dtype=torch.long
            ),
            torch.tensor(
                self.maps["history_action"][row["history_action"]], dtype=torch.long
            ),
            torch.tensor(self.maps["emotion"][row["emotion"]], dtype=torch.long),
            torch.tensor(self.maps["animation"][row["animation"]], dtype=torch.long),
            torch.tensor(
                self.maps["answer_style"][row["answer_style"]], dtype=torch.long
            ),
            torch.tensor(float(row["confidence"]), dtype=torch.float32),
        )


class ManagerModel(nn.Module):
    def __init__(self, input_size, sizes):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_size, HIDDEN),
            nn.LayerNorm(HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN // 2),
            nn.LayerNorm(HIDDEN // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        h = HIDDEN // 2

        self.intent = nn.Linear(h, sizes["intent"])
        self.primary_subject = nn.Linear(h, sizes["primary_subject"])
        self.secondary_subject = nn.Linear(h, sizes["secondary_subject"])
        self.subject_action = nn.Linear(h, sizes["subject_action"])
        self.history_action = nn.Linear(h, sizes["history_action"])
        self.emotion = nn.Linear(h, sizes["emotion"])
        self.animation = nn.Linear(h, sizes["animation"])
        self.answer_style = nn.Linear(h, sizes["answer_style"])
        self.confidence = nn.Linear(h, 1)

    def forward(self, x):
        h = self.backbone(x)
        return {
            "intent": self.intent(h),
            "primary_subject": self.primary_subject(h),
            "secondary_subject": self.secondary_subject(h),
            "subject_action": self.subject_action(h),
            "history_action": self.history_action(h),
            "emotion": self.emotion(h),
            "animation": self.animation(h),
            "answer_style": self.answer_style(h),
            "confidence": self.confidence(h).squeeze(-1),
        }


def make_label_maps():
    labels = {
        "intent": INTENTS,
        "primary_subject": SUBJECTS,
        "secondary_subject": SUBJECTS,
        "subject_action": SUBJECT_ACTIONS,
        "history_action": HISTORY_ACTIONS,
        "emotion": EMOTIONS,
        "animation": ANIMATIONS,
        "answer_style": ANSWER_STYLES,
    }

    maps = {k: {v: i for i, v in enumerate(vals)} for k, vals in labels.items()}
    return labels, maps


def compute_loss(outputs, batch, weights=None):
    weights = weights or {}

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    (
        x,
        y_intent,
        y_primary,
        y_secondary,
        y_subject_action,
        y_history_action,
        y_emotion,
        y_animation,
        y_answer_style,
        y_confidence,
    ) = batch

    loss = 0
    loss += weights.get("intent", 1.5) * ce(outputs["intent"], y_intent)
    loss += weights.get("primary_subject", 1.2) * ce(
        outputs["primary_subject"], y_primary
    )
    loss += weights.get("secondary_subject", 0.6) * ce(
        outputs["secondary_subject"], y_secondary
    )
    loss += weights.get("subject_action", 1.0) * ce(
        outputs["subject_action"], y_subject_action
    )
    loss += weights.get("history_action", 0.5) * ce(
        outputs["history_action"], y_history_action
    )
    loss += weights.get("emotion", 0.8) * ce(outputs["emotion"], y_emotion)
    loss += weights.get("animation", 0.8) * ce(outputs["animation"], y_animation)
    loss += weights.get("answer_style", 0.7) * ce(
        outputs["answer_style"], y_answer_style
    )
    loss += weights.get("confidence", 0.2) * mse(
        torch.sigmoid(outputs["confidence"]), y_confidence
    )

    return loss


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    batches = 0

    correct = Counter()
    total = Counter()

    for batch in loader:
        batch = tuple(t.to(DEVICE) for t in batch)
        x = batch[0]
        outputs = model(x)
        loss = compute_loss(outputs, batch)

        total_loss += float(loss.item())
        batches += 1

        names = [
            "intent",
            "primary_subject",
            "secondary_subject",
            "subject_action",
            "history_action",
            "emotion",
            "animation",
            "answer_style",
        ]

        targets = batch[1:9]

        for name, y in zip(names, targets):
            pred = torch.argmax(outputs[name], dim=-1)
            correct[name] += int((pred == y).sum().item())
            total[name] += int(y.numel())

    metrics = {"loss": total_loss / max(1, batches)}
    for name in correct:
        metrics[name + "_acc"] = correct[name] / max(1, total[name])

    return metrics


def move_batch(batch):
    return tuple(t.to(DEVICE) for t in batch)


def train(args):
    print("device:", DEVICE, flush=True)

    rows = generate_dataset(
        limit_project=args.limit_project,
        synthetic_rows=args.synthetic_rows,
    )

    OUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    save_jsonl(OUT_DATA_DIR / "manager_train.jsonl", rows)

    random.shuffle(rows)
    split = int(len(rows) * (1.0 - VAL_SPLIT))
    train_rows = rows[:split]
    val_rows = rows[split:] or rows[:]

    vocab = build_vocab(train_rows)
    labels, maps = make_label_maps()
    sizes = {k: len(v) for k, v in labels.items()}

    train_ds = ManagerDataset(train_rows, vocab, maps)
    val_ds = ManagerDataset(val_rows, vocab, maps)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = ManagerModel(len(vocab), sizes).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best = math.inf
    bad = 0

    print("rows:", len(rows), flush=True)
    print("train:", len(train_rows), flush=True)
    print("val:", len(val_rows), flush=True)
    print("vocab:", len(vocab), flush=True)
    print("labels:", labels, flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        batches = 0

        for batch in train_loader:
            batch = move_batch(batch)
            x = batch[0]

            opt.zero_grad(set_to_none=True)
            outputs = model(x)
            loss = compute_loss(outputs, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()

            total_loss += float(loss.item())
            batches += 1

        train_loss = total_loss / max(1, batches)
        metrics = evaluate(model, val_loader)

        print(
            f"epoch {epoch:03d} | "
            f"train {train_loss:.4f} | "
            f"val {metrics['loss']:.4f} | "
            f"intent {metrics['intent_acc']:.3f} | "
            f"subject {metrics['primary_subject_acc']:.3f} | "
            f"anim {metrics['animation_acc']:.3f}",
            flush=True,
        )

        if metrics["loss"] < best:
            best = metrics["loss"]
            bad = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": len(vocab),
                    "sizes": sizes,
                    "best_loss": best,
                },
                OUT_MODEL_DIR / "manager.pt",
            )

            save_json(OUT_MODEL_DIR / "input_vocab.json", vocab)
            save_json(OUT_MODEL_DIR / "labels.json", labels)
            save_json(
                OUT_MODEL_DIR / "config.json",
                {
                    "model_type": "meatball_manager_network",
                    "char_ngrams": list(CHAR_NGRAMS),
                    "word_ngrams": list(WORD_NGRAMS),
                    "history_word_ngrams": list(HISTORY_WORD_NGRAMS),
                    "hidden": HIDDEN,
                    "dropout": DROPOUT,
                    "animations_from_html": ANIMATIONS,
                    "emotions": EMOTIONS,
                    "known_subjects": KNOWN_SUBJECTS,
                },
            )

            print("[saved best]", flush=True)
        else:
            bad += 1
            if bad >= PATIENCE:
                print("[early stop]", flush=True)
                break

    print("DONE", flush=True)
    print("data:", OUT_DATA_DIR / "manager_train.jsonl", flush=True)
    print("model:", OUT_MODEL_DIR / "manager.pt", flush=True)


def load_runtime(model_dir):
    model_dir = Path(model_dir)
    vocab = load_json(model_dir / "input_vocab.json")
    labels = load_json(model_dir / "labels.json")
    ckpt = torch.load(model_dir / "manager.pt", map_location=DEVICE)

    sizes = {k: len(v) for k, v in labels.items()}

    model = ManagerModel(len(vocab), sizes).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    return model, vocab, labels


def vectorize_runtime(
    text, history, previous_subjects, previous_intent, previous_animation, vocab
):
    row = {
        "input": text,
        "history": history[-6:],
        "previous_subjects": previous_subjects[-5:],
        "previous_intent": previous_intent,
        "previous_animation": previous_animation,
    }
    return vectorize(row, vocab).unsqueeze(0)


@torch.no_grad()
def predict_manager(
    text,
    model,
    vocab,
    labels,
    history=None,
    previous_subjects=None,
    previous_intent="normal_qa",
    previous_animation="neutral",
):
    history = history or []
    previous_subjects = previous_subjects or []

    x = vectorize_runtime(
        text, history, previous_subjects, previous_intent, previous_animation, vocab
    ).to(DEVICE)
    outputs = model(x)

    result = {}

    for name, values in labels.items():
        probs = torch.softmax(outputs[name], dim=-1)[0]
        idx = int(torch.argmax(probs).item())
        result[name] = values[idx]
        result[name + "_confidence"] = float(probs[idx].item())

    result["confidence"] = float(torch.sigmoid(outputs["confidence"])[0].item())
    return result


def test(args):
    model, vocab, labels = load_runtime(args.model_dir)

    history = []
    subjects = []
    prev_intent = "normal_qa"
    prev_animation = "neutral"

    if args.text:
        result = predict_manager(
            args.text,
            model,
            vocab,
            labels,
            history,
            subjects,
            prev_intent,
            prev_animation,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    print("Interactive manager test. Type quit / exit / stop.")
    while True:
        text = input("\nInput> ").strip()
        if text.lower() in {"quit", "exit", "stop"}:
            break

        result = predict_manager(
            text, model, vocab, labels, history, subjects, prev_intent, prev_animation
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

        prev_intent = result["intent"]
        prev_animation = result["animation"]

        if result["primary_subject"] != "NONE":
            subjects.append(result["primary_subject"])

        history.append(text)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode")

    train_p = sub.add_parser("train")
    train_p.add_argument("--epochs", type=int, default=EPOCHS)
    train_p.add_argument("--synthetic_rows", type=int, default=125000)
    train_p.add_argument("--limit_project", type=int, default=0)

    test_p = sub.add_parser("test")
    test_p.add_argument("--model_dir", default=str(OUT_MODEL_DIR))
    test_p.add_argument("--text", default=None)

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
