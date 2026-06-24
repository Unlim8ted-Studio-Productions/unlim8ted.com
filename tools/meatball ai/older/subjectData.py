import json
import random
import re
from pathlib import Path

OUTPUT_PATH = Path("assets/data/subject-span-training.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)

TARGET_ROWS = 5000


SUBJECTS = [
    "walruses",
    "blue toaster robot",
    "the glitch",
    "the life of a meatball",
    "timecat",
    "star tracker",
    "unlim8ted",
    "smarter meatball",
    "banana car",
    "purple vacuum dragon",
    "quantum spoon",
    "electric jellyfish",
    "floating castle",
    "tiny moon machine",
    "red robot dog",
    "crystal submarine",
    "shadow pancake",
    "glass tiger",
    "rocket turtle",
    "silver mushroom",
    "zargle flomp",
    "mindle sprocket",
    "the sauce jar",
    "meatball merch",
    "the weird film",
    "my game idea",
    "the app project",
    "the music project",
    "the hardware thing",
    "Unlim8ted's robot",
    "Emily's drawing",
    "the black hoodie",
    "the glitch shirt",
    "the meatball shirt",
    "cosmic sandwich",
    "ancient frog computer",
    "neon fish helmet",
    "paper spaceship",
    "the orange button",
    "the secret project",
]


INTRO_QUESTIONS = [
    "what is {s}",
    "what are {s}",
    "what's {s}",
    "whats {s}",
    "tell me about {s}",
    "explain {s}",
    "summarize {s}",
    "who made {s}",
    "who created {s}",
    "is {s} real",
    "is {s} a project",
    "does {s} have merch",
    "can I buy {s}",
    "what does {s} do",
    "what is {s} about",
]


BOT_TEMPLATES = [
    "{S} is the thing we were talking about.",
    "{S} is a project or subject in this conversation.",
    "{S} is the current topic.",
    "{S} is what the user asked about.",
    "{S} is the subject being discussed.",
]


FOLLOWUPS = [
    ("what is it", "history"),
    ("what is that", "history"),
    ("what is this", "history"),
    ("what are they", "history"),
    ("what do they eat", "history"),
    ("what does it do", "history"),
    ("what is it about", "history"),
    ("tell me more", "history"),
    ("tell me more about it", "history"),
    ("more about that", "history"),
    ("who made it", "history"),
    ("who created it", "history"),
    ("is it real", "history"),
    ("is it a film", "history"),
    ("is it a game", "history"),
    ("is it a product", "history"),
    ("does it have merch", "history"),
    ("is there a shirt", "history"),
    ("can I buy it", "history"),
    ("where can I get it", "history"),
    ("which one is best", "history"),
    ("what's the best one", "history"),
]


SMALLTALK = [
    "hi",
    "hello",
    "hey",
    "yo",
    "how are you",
    "cool",
    "nice",
    "lol",
    "thanks",
    "ok",
    "okay",
    "awesome",
]


NONSENSE = [
    "asdf qwer zzz",
    "???",
    "what",
    "idk",
    "thing thing thing",
    "banana????",
    "florp meep",
]


def cap_first(s):
    return s[:1].upper() + s[1:] if s else s


def make_history(subject):
    q = random.choice(INTRO_QUESTIONS).format(s=subject)
    bot = random.choice(BOT_TEMPLATES).format(S=cap_first(subject))
    return [f"User: {q}", f"Bot: {bot}"]


def row(message, history, target_subject, source):
    return {
        "message": message,
        "history": history,
        "target_subject": target_subject,
        "subject_source": source,
    }


rows = []

# Explicit subject in current message.
for subject in SUBJECTS:
    for template in INTRO_QUESTIONS:
        msg = template.format(s=subject)
        rows.append(row(msg, [], subject, "message"))

# Followups where subject must be copied from history.
for subject in SUBJECTS:
    for msg, _ in FOLLOWUPS:
        rows.append(row(msg, make_history(subject), subject, "history"))

# Smalltalk should NOT inherit subject.
for subject in SUBJECTS:
    for msg in SMALLTALK:
        rows.append(row(msg, make_history(subject), "", "none"))

# No history vague followups -> no subject.
for msg, _ in FOLLOWUPS:
    rows.append(row(msg, [], "", "none"))

# Nonsense -> no subject.
for msg in NONSENSE:
    rows.append(row(msg, [], "", "none"))
    rows.append(row(msg, make_history(random.choice(SUBJECTS)), "", "none"))

# Extra randomized combinations.
while len(rows) < TARGET_ROWS:
    mode = random.choice(["explicit", "followup", "smalltalk", "none"])
    subject = random.choice(SUBJECTS)

    if mode == "explicit":
        msg = random.choice(INTRO_QUESTIONS).format(s=subject)
        hist = random.choice([[], make_history(random.choice(SUBJECTS))])
        rows.append(row(msg, hist, subject, "message"))

    elif mode == "followup":
        msg, _ = random.choice(FOLLOWUPS)
        rows.append(row(msg, make_history(subject), subject, "history"))

    elif mode == "smalltalk":
        msg = random.choice(SMALLTALK)
        rows.append(row(msg, make_history(subject), "", "none"))

    else:
        msg = random.choice(NONSENSE + [m for m, _ in FOLLOWUPS])
        rows.append(row(msg, [], "", "none"))

random.shuffle(rows)

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    for r in rows[:TARGET_ROWS]:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print("wrote", min(len(rows), TARGET_ROWS), "rows to", OUTPUT_PATH)
