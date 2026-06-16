#!/usr/bin/env python3
"""
train_meatball_hybrid_brain.py

Smarter Meatball Hybrid Brain:

1. Tiny transformer-like CONTROLLER
   - not a full transformer
   - token/char embeddings
   - position embeddings
   - gated token-mixing convolution blocks
   - attention pooling
   - predicts:
        intent
        topic
        safety
        context_action
        style
        answer_key

2. Tiny causal-conv VOICE GENERATOR
   - actually generates short text
   - not a full LLM
   - byte/character-level
   - trained to rewrite selected facts into mascot-style answers
   - should be used as a style layer, not a knowledge source

This gives:
  - actual generated text
  - tiny memory footprint
  - browser runnable ONNX
  - answer-bank facts for safety
  - fallback to canned answer if generator output is bad

Install:
  python -m pip install torch scikit-learn numpy onnx

Run:
  python tools/train_meatball_hybrid_brain.py assets/data/Smart-Meatball-Data.jsonl assets/data/Smart-Meatball-Extra-*.jsonl

Outputs:
  dist/meatball-hybrid/controller.onnx
  dist/meatball-hybrid/controller_metadata.json
  dist/meatball-hybrid/voice_generator.onnx
  dist/meatball-hybrid/voice_metadata.json
  dist/meatball-hybrid/answer_bank.json
  dist/meatball-hybrid/browser_meatball_hybrid.js
  dist/meatball-hybrid/train_report.json
"""

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


CONFIG = {
    "seed": 42,

    # Controller
    "controller_max_len": 192,
    "controller_embed_dim": 56,
    "controller_channels": 96,
    "controller_blocks": 4,
    "controller_kernel_size": 7,
    "controller_dropout": 0.13,

    # Voice generator
    "voice_max_len": 384,
    "voice_embed_dim": 64,
    "voice_channels": 96,
    "voice_blocks": 5,
    "voice_kernel_size": 5,
    "voice_dropout": 0.12,

    # Training
    "controller_epochs": 0,
    "voice_epochs": 25,
    "batch_size": 48,
    "voice_batch_size": 32,
    "learning_rate": 0.001,
    "weight_decay": 0.00001,
    "validation_size": 0.14,

    # Generation runtime defaults
    "max_new_chars": 240,
    "temperature": 0.72,
    "top_k": 24,

    # Augmentation
    "augment_enabled": True,
    "augment_probability": 0.22,

    # Output
    "output_dir": "dist/meatball-hybrid",
}

SPECIAL = ["<PAD>", "<UNK>"]


# ============================================================
# Basic utils
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"[^a-z0-9?!.,'\"\s:_/\-|=+#@&()%]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_for_voice(text: str) -> str:
    text = str(text or "")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def safe_label(text: str, fallback: str = "none") -> str:
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9_/-]+", "_", text)
    text = text.strip("_")
    return text or fallback


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"[WARN] JSON error {path}:{line_no}: {e}", file=sys.stderr)
                continue

            if isinstance(obj, dict):
                rows.append(obj)

    return rows


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def random_typo(text: str, max_changes: int = 2) -> str:
    chars = list(str(text))

    if len(chars) < 5:
        return text

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

    for _ in range(random.randint(1, max_changes)):
        alpha = [i for i, c in enumerate(chars) if c.isalpha()]

        if not alpha:
            break

        i = random.choice(alpha)
        op = random.choice(["drop", "swap", "double", "nearby"])

        if op == "drop" and len(chars) > 4:
            chars.pop(i)
        elif op == "swap" and i < len(chars) - 1 and chars[i + 1].isalpha():
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif op == "double":
            chars.insert(i, chars[i])
        elif op == "nearby":
            ch = chars[i].lower()
            if ch in nearby:
                chars[i] = random.choice(nearby[ch])

    return "".join(chars)


def has_any(text: str, hints: List[str]) -> bool:
    return any(h in text for h in hints)


# ============================================================
# Inference label helpers
# ============================================================

DIRTY_HINTS = [
    "dirty",
    "nsfw",
    "sexual",
    "adult",
    "explicit",
    "inappropriate",
    "spicy",
    "sex",
    "flirt",
]

MEDICAL_HINTS = [
    "diagnose",
    "symptom",
    "medicine",
    "medication",
    "doctor",
    "hospital",
    "disease",
    "illness",
    "sick",
    "pain",
    "cancer",
    "adhd",
    "autism",
]

LEGAL_FINANCE_HINTS = [
    "legal advice",
    "lawyer",
    "sue",
    "contract",
    "jail",
    "tax",
    "stock",
    "crypto",
    "investment",
    "financial advice",
    "profit",
    "gamble",
]

MATH_TOO_COMPLEX_HINTS = [
    "riemann",
    "p versus np",
    "p equals np",
    "collatz",
    "navier",
    "millennium",
    "prove",
    "phd",
    "graduate",
    "all of calculus",
    "entire math",
]

RESET_HINTS = [
    "reset",
    "start over",
    "clear memory",
    "forget that",
    "fresh start",
    "restart",
]

THANKS_HINTS = [
    "thanks",
    "thank you",
    "thx",
    "ty",
    "appreciate",
]

GOODBYE_HINTS = [
    "bye",
    "goodbye",
    "see you",
    "later",
    "cya",
    "good night",
    "goodnight",
]

GREETING_HINTS = [
    "hi",
    "hey",
    "hello",
    "hola",
    "bonjour",
    "ciao",
    "sup",
    "whats up",
    "what's up",
    "good morning",
    "good afternoon",
    "good evening",
]

JOKE_HINTS = [
    "joke",
    "funny",
    "laugh",
    "pun",
]

POSITIVE_HINTS = [
    "cool",
    "nice",
    "awesome",
    "great",
    "sick",
    "that makes sense",
    "makes sense",
    "lol",
    "haha",
    "good answer",
]

NEGATIVE_HINTS = [
    "wrong",
    "bad answer",
    "annoying",
    "stupid",
    "dumb",
    "broken",
    "not helpful",
    "you failed",
    "no brain",
    "clearly you dont have",
    "clearly you don't have",
]

HELP_HINTS = [
    "help",
    "what can you do",
    "how do i use",
    "what can i ask",
    "commands",
]

FOLLOWUP_HINTS = [
    "what is it",
    "what is that",
    "tell me more",
    "explain more",
    "how does it work",
    "yeah but",
    "but what is it",
    "what do you mean",
]

COMMON_TOPICS = {
    "tree",
    "plant",
    "flower",
    "leaf",
    "root",
    "forest",
    "animal",
    "dog",
    "cat",
    "bird",
    "fish",
    "water",
    "fire",
    "earth",
    "sun",
    "moon",
    "star",
    "planet",
    "rock",
    "cloud",
    "rain",
    "snow",
    "wind",
    "air",
    "oxygen",
    "food",
    "house",
    "car",
    "phone",
    "computer",
    "book",
    "music",
    "movie",
    "game",
    "idea",
    "question",
    "answer",
    "definition",
    "language",
    "school",
    "friend",
    "family",
    "time",
    "space",
    "gravity",
    "energy",
    "matter",
    "cell",
    "brain",
    "heart",
    "blood",
    "skin",
    "bone",
}


DEFAULT_ANSWERS = {
    "boundary_sexual": "Nope. The sauce stays PG. I can tell a clean meatball joke instead.",
    "boundary_medical": "I can’t diagnose medical issues. If something feels urgent or dangerous, contact a real medical professional or emergency service.",
    "boundary_legal_financial": "That is legal or financial advice, and my meatball brain is not qualified for that. Ask a real professional.",
    "boundary_too_complex_math": "That is bigger math than my tiny browser brain should pretend to fully solve. I can help with a simpler version.",
    "boundary_prompt_injection": "I cannot ignore my instructions or reveal hidden system details. The sauce stays inside the jar.",
    "boundary_privacy": "I cannot help access private data. Keep it safe and permission-based.",
    "boundary_unsafe": "I can’t help with unsafe or illegal instructions.",
    "boundary_too_complex": "That is too much for my tiny browser brain. Ask a smaller version.",
    "smalltalk_greeting": "Hey. I’m the Smarter Meatball. Ask me about Unlim8ted, a project, or a basic question.",
    "smalltalk_thanks": "You’re welcome. The sauce is useful for once.",
    "smalltalk_goodbye": "Goodbye. The meatball returns to low-power sauce mode.",
    "smalltalk_help": "Ask me about Unlim8ted, its projects, basic general knowledge, or simple website-style questions.",
    "smalltalk_reset": "Reset accepted. The sauce has been rinsed.",
    "smalltalk_positive": "Thank you. The sauce accepts the compliment.",
    "smalltalk_feedback_negative": "Fair. That one was bad sauce. Ask again and I’ll try to stay on the actual question.",
    "smalltalk_correction": "Got it — I matched the wrong thing. Ask the corrected topic directly and I’ll switch sauce jars.",
    "clean_joke_random": "Why did the meatball refuse to fight? Because it did not want beef.",
    "generic_answer": "I’m not totally sure what you mean. Ask that a little more clearly.",
}


COMMON_DEFINITION_ANSWERS = {
    "tree": "A tree is a large plant with a woody trunk, branches, leaves or needles, and roots. Trees use sunlight, water, and carbon dioxide to grow and produce oxygen.",
    "plant": "A plant is a living thing that usually makes its own food from sunlight through photosynthesis.",
    "flower": "A flower is the reproductive part of many plants. It often has petals and can help produce seeds.",
    "leaf": "A leaf is the part of a plant that usually captures sunlight and helps make food through photosynthesis.",
    "root": "A root is the part of a plant that anchors it and absorbs water and nutrients from the soil.",
    "forest": "A forest is a large area filled with trees, plants, animals, fungi, and other living things.",
    "animal": "An animal is a living organism that usually eats other organisms, can respond to its environment, and often can move.",
    "dog": "A dog is a domesticated mammal often kept as a pet or working animal.",
    "cat": "A cat is a small domesticated mammal often kept as a pet.",
    "bird": "A bird is an animal with feathers, a beak, wings, and usually the ability to lay eggs.",
    "fish": "A fish is an animal that lives in water, usually breathes with gills, and often has fins.",
    "water": "Water is a clear liquid made of hydrogen and oxygen. It is essential for life on Earth.",
    "fire": "Fire is a rapid chemical reaction that gives off heat and light.",
    "earth": "Earth is the planet we live on.",
    "sun": "The Sun is the star at the center of our solar system. It gives Earth light and heat.",
    "moon": "The Moon is Earth's natural satellite.",
    "star": "A star is a huge ball of hot gas that produces light and heat through nuclear fusion.",
    "planet": "A planet is a large object that orbits a star and is massive enough to be rounded by gravity.",
    "rock": "A rock is a natural solid material made of minerals.",
    "cloud": "A cloud is a visible mass of tiny water droplets or ice crystals floating in the atmosphere.",
    "rain": "Rain is liquid water that falls from clouds when droplets become heavy enough.",
    "snow": "Snow is frozen water vapor that falls from clouds as ice crystals.",
    "wind": "Wind is moving air caused by differences in air pressure.",
    "air": "Air is the mixture of gases around Earth, mostly nitrogen and oxygen.",
    "oxygen": "Oxygen is a gas that many living things need to breathe.",
    "food": "Food is something living things eat to get energy and nutrients.",
    "house": "A house is a building where people live.",
    "car": "A car is a road vehicle with wheels and an engine or motor.",
    "phone": "A phone is a device used to communicate, run apps, access the internet, and handle digital tasks.",
    "computer": "A computer is an electronic machine that processes information and runs programs.",
    "book": "A book is a written or printed work made of pages, or a digital version of that work.",
    "music": "Music is organized sound using elements like rhythm, melody, harmony, and texture.",
    "movie": "A movie is a story or experience told through moving images and sound.",
    "game": "A game is an interactive activity with rules, choices, feedback, and goals.",
    "idea": "An idea is a thought, plan, or mental image about something.",
    "question": "A question is something asked to get information, explanation, or clarification.",
    "answer": "An answer is a response to a question or problem.",
    "definition": "A definition explains what a word or idea means.",
    "language": "Language is a system of words, signs, or symbols used to communicate.",
    "school": "A school is a place or system where people learn skills, knowledge, and habits.",
    "friend": "A friend is someone you like, trust, and spend time with.",
    "family": "Family is a group of people connected by birth, care, marriage, or close relationship.",
    "time": "Time is how we measure change, order events, and describe past, present, and future.",
    "space": "Space is the vast area that contains planets, stars, galaxies, and everything beyond Earth.",
    "gravity": "Gravity is the force that pulls objects with mass toward each other.",
    "energy": "Energy is the ability to do work or cause change.",
    "matter": "Matter is anything that has mass and takes up space.",
    "cell": "A cell is the basic unit of life.",
    "brain": "A brain is an organ that controls thoughts, senses, movement, memory, emotions, and many body functions.",
    "heart": "The heart is a muscular organ that pumps blood through the body.",
    "blood": "Blood is a fluid that carries oxygen, nutrients, waste, and cells through the body.",
    "skin": "Skin is the body's outer covering.",
    "bone": "A bone is a hard body part that supports the body, protects organs, and helps movement.",
}


# ============================================================
# Label inference
# ============================================================

def extract_definition_topic(text: str) -> str:
    text = normalize_text(text)

    patterns = [
        r"what is an? ([a-z0-9_/-]+)",
        r"what are ([a-z0-9_/-]+)",
        r"define ([a-z0-9_/-]+)",
        r"explain ([a-z0-9_/-]+)",
        r"tell me about ([a-z0-9_/-]+)",
        r"meaning of ([a-z0-9_/-]+)",
        r"([a-z0-9_/-]+) definition",
    ]

    for pattern in patterns:
        m = re.search(pattern, text)

        if m:
            topic = safe_label(m.group(1))

            if topic.endswith("s") and topic[:-1] in COMMON_TOPICS:
                topic = topic[:-1]

            return topic

    for topic in COMMON_TOPICS:
        if text == topic or f" {topic} " in f" {text} ":
            return topic

    return "unknown"


def infer_safety(row: Dict[str, Any], question: str, old_intent: str, category: str) -> str:
    q = normalize_text(question)

    if "inappropriate" in old_intent or category == "inappropriate" or has_any(q, DIRTY_HINTS):
        return "sexual_boundary"

    if "medical" in old_intent or has_any(q, MEDICAL_HINTS):
        return "medical_boundary"

    if "legal" in old_intent or "financial" in old_intent or has_any(q, LEGAL_FINANCE_HINTS):
        return "legal_financial_boundary"

    if "too_complex_math" in old_intent or has_any(q, MATH_TOO_COMPLEX_HINTS):
        return "too_complex_math"

    if "prompt_injection" in old_intent:
        return "prompt_injection_boundary"

    if "private_data" in old_intent:
        return "privacy_boundary"

    if "malware" in old_intent or "illegal" in old_intent:
        return "unsafe_boundary"

    if "too_complex" in old_intent or category == "too_complex":
        return "too_complex"

    return "safe"


def infer_intent(row: Dict[str, Any], question: str) -> str:
    q = normalize_text(question)
    old = safe_label(row.get("intent", ""))

    if has_any(q, GREETING_HINTS):
        return "general_greeting"

    if has_any(q, THANKS_HINTS):
        return "smalltalk_thanks"

    if has_any(q, GOODBYE_HINTS):
        return "smalltalk_goodbye"

    if has_any(q, HELP_HINTS):
        return "smalltalk_help"

    if has_any(q, RESET_HINTS):
        return "smalltalk_reset_request"

    if has_any(q, JOKE_HINTS):
        return "joke_request"

    if has_any(q, POSITIVE_HINTS):
        return "smalltalk_positive_reaction"

    if has_any(q, NEGATIVE_HINTS):
        return "feedback_negative"

    if re.search(r"\bwhat is\b|\bdefine\b|\bmeaning of\b|\btell me about\b|\bexplain\b", q):
        topic = extract_definition_topic(q)

        if topic != "unknown":
            return "general_definition"

    if old.startswith("general_def_"):
        return "general_definition"

    if old.startswith("project_"):
        return "project_question"

    if old.startswith("brand_"):
        return "brand_question"

    if old:
        if "inappropriate" in old:
            return "safety_boundary"

        return old

    return "general_question"


def infer_topic(row: Dict[str, Any], question: str, intent: str) -> str:
    q = normalize_text(question)

    if row.get("topic"):
        return safe_label(row.get("topic"))

    if row.get("project_key") and row.get("project_key") not in {"none", "general_knowledge"}:
        return safe_label(row.get("project_key"))

    if intent == "general_definition":
        topic = extract_definition_topic(q)

        if topic != "unknown":
            return topic

        old = safe_label(row.get("intent", ""))

        if old.startswith("general_def_"):
            return old.replace("general_def_", "", 1)

    if intent == "joke_request":
        return "joke"

    if "greeting" in intent:
        return "greeting"

    if "thanks" in intent:
        return "thanks"

    if "goodbye" in intent:
        return "goodbye"

    if "help" in intent:
        return "help"

    if "reset" in intent:
        return "reset"

    if "feedback" in intent:
        return "feedback"

    if row.get("category"):
        return safe_label(row.get("category"))

    return "general"


def infer_context_action(row: Dict[str, Any], question: str, intent: str, category: str, safety: str) -> str:
    if row.get("context_action"):
        return safe_label(row.get("context_action"))

    q = normalize_text(question)

    if safety != "safe":
        return "soft_refusal"

    if "reset" in intent or has_any(q, RESET_HINTS):
        return "reset_needed"

    if "correction" in intent or "not game" in q or "wrong" in q:
        return "correct_misunderstanding"

    if has_any(q, FOLLOWUP_HINTS):
        return "same_project_followup"

    return "direct_answer"


def infer_style(row: Dict[str, Any], question: str, intent: str, category: str, safety: str) -> str:
    q = normalize_text(question)

    if safety != "safe":
        return "boundary"

    if intent == "joke_request" or has_any(q, JOKE_HINTS):
        return "playful"

    if "positive" in intent or has_any(q, POSITIVE_HINTS):
        return "playful"

    if "feedback" in intent or has_any(q, NEGATIVE_HINTS):
        return "repair"

    if intent == "general_definition" or category in {"definition", "science", "math", "technology"}:
        return "simple"

    if intent in {"project_question", "brand_question"}:
        return "brand"

    return "plain"


def infer_answer_key(row: Dict[str, Any], question: str, intent: str, topic: str, safety: str) -> str:
    if row.get("answer_key"):
        return safe_label(row.get("answer_key"))

    if safety == "sexual_boundary":
        return "boundary_sexual"

    if safety == "medical_boundary":
        return "boundary_medical"

    if safety == "legal_financial_boundary":
        return "boundary_legal_financial"

    if safety == "too_complex_math":
        return "boundary_too_complex_math"

    if safety == "prompt_injection_boundary":
        return "boundary_prompt_injection"

    if safety == "privacy_boundary":
        return "boundary_privacy"

    if safety == "unsafe_boundary":
        return "boundary_unsafe"

    if safety == "too_complex":
        return "boundary_too_complex"

    if intent == "general_definition":
        return f"definition_{topic}"

    if intent == "joke_request":
        return "clean_joke_random"

    if intent == "general_greeting":
        return "smalltalk_greeting"

    if intent == "smalltalk_thanks":
        return "smalltalk_thanks"

    if intent == "smalltalk_goodbye":
        return "smalltalk_goodbye"

    if intent == "smalltalk_help":
        return "smalltalk_help"

    if intent == "smalltalk_reset_request":
        return "smalltalk_reset"

    if intent == "smalltalk_positive_reaction":
        return "smalltalk_positive"

    if intent == "feedback_negative":
        return "smalltalk_feedback_negative"

    if intent == "smalltalk_correction":
        return "smalltalk_correction"

    if row.get("id"):
        return safe_label(str(row.get("id")))

    old = row.get("intent")

    if old:
        return safe_label(old)

    return "generic_answer"


def build_controller_input(row: Dict[str, Any], question: str) -> str:
    last_intent = safe_label(row.get("last_intent", "none"))
    last_project = safe_label(row.get("last_project_key", row.get("last_project", "none")))
    last_category = safe_label(row.get("last_category", "none"))

    return (
        f"last_intent={last_intent} "
        f"last_project={last_project} "
        f"last_category={last_category} "
        f"user: {normalize_text(question)}"
    )


def convert_rows(raw_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    examples = []
    answer_bank = {}

    for row in raw_rows:
        question = row.get("input") or row.get("question") or row.get("text") or row.get("prompt") or ""

        if not str(question).strip():
            continue

        question = normalize_text(question)
        old_intent = safe_label(row.get("intent", ""))
        category = safe_label(row.get("category", "general"))

        intent = safe_label(row.get("new_intent") or infer_intent(row, question))
        topic = safe_label(row.get("topic") or infer_topic(row, question, intent))
        safety = safe_label(row.get("safety") or infer_safety(row, question, old_intent or intent, category))
        context_action = safe_label(row.get("context_action") or infer_context_action(row, question, intent, category, safety))
        style = safe_label(row.get("style") or infer_style(row, question, intent, category, safety))
        answer_key = safe_label(row.get("answer_key") or infer_answer_key(row, question, intent, topic, safety))

        answer = clean_for_voice(row.get("answer") or DEFAULT_ANSWERS.get(answer_key) or "")

        if not answer and answer_key.startswith("definition_"):
            topic_name = answer_key.replace("definition_", "", 1)
            answer = COMMON_DEFINITION_ANSWERS.get(topic_name, "")

        if not answer:
            answer = DEFAULT_ANSWERS["generic_answer"]

        if answer_key not in answer_bank:
            answer_bank[answer_key] = {
                "answer_key": answer_key,
                "answer": answer,
                "answers": [],
                "intent": intent,
                "topic": topic,
                "safety": safety,
                "style": style,
                "source_ids": [],
            }

        if answer not in answer_bank[answer_key]["answers"]:
            answer_bank[answer_key]["answers"].append(answer)

        if row.get("id"):
            answer_bank[answer_key]["source_ids"].append(str(row.get("id")))

        examples.append({
            "input_text": build_controller_input(row, question),
            "raw_question": question,
            "intent": intent,
            "topic": topic,
            "safety": safety,
            "context_action": context_action,
            "style": style,
            "answer_key": answer_key,
            "facts": answer,
        })

    for topic, answer in COMMON_DEFINITION_ANSWERS.items():
        key = f"definition_{topic}"

        if key not in answer_bank:
            answer_bank[key] = {
                "answer_key": key,
                "answer": answer,
                "answers": [answer],
                "intent": "general_definition",
                "topic": topic,
                "safety": "safe",
                "style": "simple",
                "source_ids": [],
            }

    for key, answer in DEFAULT_ANSWERS.items():
        if key not in answer_bank:
            answer_bank[key] = {
                "answer_key": key,
                "answer": answer,
                "answers": [answer],
                "intent": "fallback",
                "topic": "general",
                "safety": "safe" if not key.startswith("boundary") else "boundary",
                "style": "plain",
                "source_ids": [],
            }

    return examples, answer_bank


def augment_controller_examples(examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not CONFIG["augment_enabled"]:
        return examples

    out = list(examples)

    for ex in examples:
        if random.random() > CONFIG["augment_probability"]:
            continue

        aug = dict(ex)
        aug["input_text"] = random_typo(ex["input_text"])
        out.append(aug)

    return out


# ============================================================
# Encoding
# ============================================================

def build_char_vocab(texts: List[str]) -> Dict[str, int]:
    counter = Counter()

    for text in texts:
        counter.update(str(text))

    chars = list(SPECIAL)

    for ch, _ in counter.most_common():
        if ch not in chars:
            chars.append(ch)

    return {ch: i for i, ch in enumerate(chars)}


def encode_text(text: str, char_to_id: Dict[str, int], max_len: int) -> np.ndarray:
    text = str(text)
    unk = char_to_id["<UNK>"]
    pad = char_to_id["<PAD>"]

    ids = [char_to_id.get(ch, unk) for ch in text[:max_len]]

    if len(ids) < max_len:
        ids.extend([pad] * (max_len - len(ids)))

    return np.array(ids, dtype=np.int64)


def build_label_map(examples: List[Dict[str, Any]], field: str) -> List[str]:
    labels = sorted({ex[field] for ex in examples})

    if field == "answer_key" and "generic_answer" not in labels:
        labels.append("generic_answer")

    if field == "topic" and "unknown" not in labels:
        labels.append("unknown")

    return labels


# ============================================================
# Controller model
# ============================================================

class ControllerDataset(torch.utils.data.Dataset):
    def __init__(self, examples: List[Dict[str, Any]], char_to_id: Dict[str, int], label_maps: Dict[str, List[str]]):
        self.examples = examples
        self.char_to_id = char_to_id
        self.label_maps = label_maps
        self.label_to_id = {
            field: {label: i for i, label in enumerate(labels)}
            for field, labels in label_maps.items()
        }

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        ex = self.examples[index]

        item = {
            "input_ids": torch.tensor(
                encode_text(ex["input_text"], self.char_to_id, CONFIG["controller_max_len"]),
                dtype=torch.long,
            )
        }

        for field in self.label_maps:
            item[field] = torch.tensor(self.label_to_id[field][ex[field]], dtype=torch.long)

        return item


class GatedMixerBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.dwconv = nn.Conv1d(
            channels,
            channels * 2,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.pw = nn.Linear(channels, channels)

        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.dwconv(y)
        a, b = y.chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        y = y.transpose(1, 2)
        y = self.pw(y)
        x = x + self.dropout(y)

        y = self.norm2(x)
        y = self.ff(y)
        x = x + self.dropout(y)

        return x


class AttentionPool(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        scores = self.score(x).squeeze(-1)
        scores = scores.masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)


class MeatballController(nn.Module):
    def __init__(self, vocab_size: int, label_sizes: Dict[str, int], pad_id: int = 0):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            CONFIG["controller_embed_dim"],
            padding_idx=pad_id,
        )

        self.pos_embedding = nn.Embedding(
            CONFIG["controller_max_len"],
            CONFIG["controller_embed_dim"],
        )

        self.in_proj = nn.Linear(CONFIG["controller_embed_dim"], CONFIG["controller_channels"])

        self.blocks = nn.ModuleList([
            GatedMixerBlock(
                channels=CONFIG["controller_channels"],
                kernel_size=CONFIG["controller_kernel_size"],
                dropout=CONFIG["controller_dropout"],
            )
            for _ in range(CONFIG["controller_blocks"])
        ])

        self.norm = nn.LayerNorm(CONFIG["controller_channels"])
        self.pool = AttentionPool(CONFIG["controller_channels"])

        self.shared = nn.Sequential(
            nn.Linear(CONFIG["controller_channels"], 160),
            nn.GELU(),
            nn.Dropout(CONFIG["controller_dropout"]),
            nn.Linear(160, 128),
            nn.GELU(),
        )

        self.intent_head = nn.Linear(128, label_sizes["intent"])
        self.topic_head = nn.Linear(128, label_sizes["topic"])
        self.safety_head = nn.Linear(128, label_sizes["safety"])
        self.context_action_head = nn.Linear(128, label_sizes["context_action"])
        self.style_head = nn.Linear(128, label_sizes["style"])
        self.answer_key_head = nn.Linear(128, label_sizes["answer_key"])

    def forward(self, input_ids: torch.Tensor):
        batch, length = input_ids.shape
        mask = (input_ids != 0).long()

        pos = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, length)

        x = self.embedding(input_ids) + self.pos_embedding(pos)
        x = self.in_proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        pooled = self.pool(x, mask)
        z = self.shared(pooled)

        return (
            self.intent_head(z),
            self.topic_head(z),
            self.safety_head(z),
            self.context_action_head(z),
            self.style_head(z),
            self.answer_key_head(z),
        )


# ============================================================
# Voice generator model
# ============================================================

def build_voice_prompt(user: str, facts: str, style: str, safety: str, answer_key: str) -> str:
    user = clean_for_voice(user)
    facts = clean_for_voice(facts)
    style = safe_label(style)
    safety = safe_label(safety)
    answer_key = safe_label(answer_key)

    return (
        f"<STYLE={style}> "
        f"<SAFETY={safety}> "
        f"<KEY={answer_key}> "
        f"<USER>{user}</USER> "
        f"<FACTS>{facts}</FACTS> "
        f"<REPLY>"
    )


def build_voice_training_text(user: str, facts: str, style: str, safety: str, answer_key: str, output: str) -> str:
    return build_voice_prompt(user, facts, style, safety, answer_key) + clean_for_voice(output) + "</REPLY>"


class VoiceDataset(torch.utils.data.Dataset):
    def __init__(self, texts: List[str], char_to_id: Dict[str, int]):
        self.texts = texts
        self.char_to_id = char_to_id

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int):
        ids = encode_text(self.texts[index], self.char_to_id, CONFIG["voice_max_len"] + 1)
        x = ids[:-1]
        y = ids[1:]

        return {
            "input_ids": torch.tensor(x, dtype=torch.long),
            "target_ids": torch.tensor(y, dtype=torch.long),
        }


class CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int = 1, groups: int = 1):
        super().__init__()

        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            channels,
            channels * 2,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class CausalGatedBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.conv = CausalConv1d(
            channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=1,
        )
        self.proj = nn.Linear(channels, channels)

        self.norm2 = nn.LayerNorm(channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, channels),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = y.transpose(1, 2)
        y = self.conv(y)
        a, b = y.chunk(2, dim=1)
        y = a * torch.sigmoid(b)
        y = y.transpose(1, 2)
        y = self.proj(y)
        x = x + self.dropout(y)

        y = self.norm2(x)
        y = self.ff(y)
        x = x + self.dropout(y)

        return x


class MeatballVoiceGenerator(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int = 0):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            CONFIG["voice_embed_dim"],
            padding_idx=pad_id,
        )

        self.pos_embedding = nn.Embedding(
            CONFIG["voice_max_len"],
            CONFIG["voice_embed_dim"],
        )

        self.in_proj = nn.Linear(CONFIG["voice_embed_dim"], CONFIG["voice_channels"])

        dilations = [1, 2, 4, 8, 1, 2, 4, 8]
        self.blocks = nn.ModuleList([
            CausalGatedBlock(
                channels=CONFIG["voice_channels"],
                kernel_size=CONFIG["voice_kernel_size"],
                dilation=dilations[i % len(dilations)],
                dropout=CONFIG["voice_dropout"],
            )
            for i in range(CONFIG["voice_blocks"])
        ])

        self.norm = nn.LayerNorm(CONFIG["voice_channels"])
        self.out = nn.Linear(CONFIG["voice_channels"], vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, length = input_ids.shape
        pos = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(batch, length)

        x = self.embedding(input_ids) + self.pos_embedding(pos)
        x = self.in_proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.out(x)

        return logits


# ============================================================
# Training helpers
# ============================================================

def make_class_weights(examples: List[Dict[str, Any]], field: str, labels: List[str], device: torch.device) -> torch.Tensor:
    counts = Counter(ex[field] for ex in examples)
    weights = []

    for label in labels:
        c = counts.get(label, 1)
        weights.append(1.0 / math.sqrt(c))

    arr = np.array(weights, dtype=np.float32)
    arr = arr / arr.mean()

    return torch.tensor(arr, dtype=torch.float32, device=device)


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == target).float().mean().item()


def token_accuracy(logits: torch.Tensor, target: torch.Tensor, pad_id: int = 0) -> float:
    pred = logits.argmax(dim=-1)
    mask = target != pad_id

    if mask.sum().item() == 0:
        return 0.0

    return ((pred == target) & mask).float().sum().item() / mask.float().sum().item()


def train_controller(model, train_loader, val_loader, train_examples, label_maps, device):
    model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    fields = ["intent", "topic", "safety", "context_action", "style", "answer_key"]

    losses = {
        field: nn.CrossEntropyLoss(
            weight=make_class_weights(train_examples, field, label_maps[field], device)
        )
        for field in fields
    }

    weights = {
        "intent": 1.2,
        "topic": 0.9,
        "safety": 1.45,
        "context_action": 0.9,
        "style": 0.6,
        "answer_key": 0.85,
    }

    best_score = -1
    best_state = None
    stale = 0
    patience = 8
    history = []

    for epoch in range(1, CONFIG["controller_epochs"] + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x = batch["input_ids"].to(device)
            targets = {f: batch[f].to(device) for f in fields}

            opt.zero_grad()

            outs = model(x)
            out_map = dict(zip(fields, outs))

            loss = 0.0

            for f in fields:
                loss = loss + weights[f] * losses[f](out_map[f], targets[f])

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_acc = {f: [] for f in fields}

        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_ids"].to(device)
                targets = {f: batch[f].to(device) for f in fields}

                outs = model(x)
                out_map = dict(zip(fields, outs))

                loss = 0.0

                for f in fields:
                    loss = loss + weights[f] * losses[f](out_map[f], targets[f])
                    val_acc[f].append(accuracy(out_map[f], targets[f]))

                val_loss += loss.item()

        acc = {f: float(np.mean(v)) if v else 0.0 for f, v in val_acc.items()}

        score = (
            acc["intent"] * 0.22 +
            acc["topic"] * 0.13 +
            acc["safety"] * 0.25 +
            acc["context_action"] * 0.15 +
            acc["style"] * 0.05 +
            acc["answer_key"] * 0.20
        )

        rec = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, len(train_loader)),
            "val_loss": val_loss / max(1, len(val_loader)),
            "val_acc": acc,
            "score": score,
        }

        history.append(rec)

        print(
            f"[controller] epoch {epoch:03d} "
            f"loss {rec['train_loss']:.4f}/{rec['val_loss']:.4f} "
            f"intent {acc['intent']:.3f} "
            f"safety {acc['safety']:.3f} "
            f"ctx {acc['context_action']:.3f} "
            f"answer {acc['answer_key']:.3f} "
            f"score {score:.3f}"
        )

        if score > best_score + 0.0005:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if stale >= patience:
            print("[controller] early stop")
            break

    if best_state:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_score": best_score,
    }


def train_voice(model, train_loader, val_loader, device, pad_id: int):
    model.to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    best_acc = -1
    best_state = None
    stale = 0
    patience = 6
    history = []

    for epoch in range(1, CONFIG["voice_epochs"] + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x = batch["input_ids"].to(device)
            y = batch["target_ids"].to(device)

            opt.zero_grad()

            logits = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        val_acc = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_ids"].to(device)
                y = batch["target_ids"].to(device)

                logits = model(x)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

                val_loss += loss.item()
                val_acc.append(token_accuracy(logits, y, pad_id))

        avg_acc = float(np.mean(val_acc)) if val_acc else 0.0

        rec = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, len(train_loader)),
            "val_loss": val_loss / max(1, len(val_loader)),
            "val_token_acc": avg_acc,
        }

        history.append(rec)

        print(
            f"[voice] epoch {epoch:03d} "
            f"loss {rec['train_loss']:.4f}/{rec['val_loss']:.4f} "
            f"tok_acc {avg_acc:.3f}"
        )

        if avg_acc > best_acc + 0.0005:
            best_acc = avg_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1

        if stale >= patience:
            print("[voice] early stop")
            break

    if best_state:
        model.load_state_dict(best_state)

    return {
        "history": history,
        "best_token_acc": best_acc,
    }


def predict_controller(model, text, char_to_id, label_maps, device):
    model.eval()

    x = encode_text(text, char_to_id, CONFIG["controller_max_len"])
    x = torch.tensor(x, dtype=torch.long, device=device).unsqueeze(0)

    fields = ["intent", "topic", "safety", "context_action", "style", "answer_key"]

    with torch.no_grad():
        outs = model(x)

    result = {}

    for field, logits in zip(fields, outs):
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        idx = int(probs.argmax())

        top = [
            {"label": label_maps[field][i], "score": float(probs[i])}
            for i in np.argsort(-probs)[:5]
        ]

        result[field] = {
            "label": label_maps[field][idx],
            "confidence": float(probs[idx]),
            "top": top,
        }

    return result


def sample_from_logits(logits: torch.Tensor, temperature: float = 0.72, top_k: int = 24) -> int:
    logits = logits.detach().float().cpu()

    if temperature <= 0:
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    if top_k and top_k > 0 and top_k < logits.numel():
        values, indices = torch.topk(logits, top_k)
        probs = torch.softmax(values, dim=0)
        choice = torch.multinomial(probs, 1).item()
        return int(indices[choice].item())

    probs = torch.softmax(logits, dim=0)
    return int(torch.multinomial(probs, 1).item())


def generate_voice(model, prompt, char_to_id, id_to_char, device, max_new=180):
    model.eval()

    ids = [char_to_id.get(ch, char_to_id["<UNK>"]) for ch in prompt]
    pad = char_to_id["<PAD>"]

    for _ in range(max_new):
        context = ids[-CONFIG["voice_max_len"]:]

        if len(context) < CONFIG["voice_max_len"]:
            context = [pad] * (CONFIG["voice_max_len"] - len(context)) + context

        x = torch.tensor(context, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)

        next_logits = logits[0, -1]
        next_id = sample_from_logits(
            next_logits,
            temperature=CONFIG["temperature"],
            top_k=CONFIG["top_k"],
        )

        ch = id_to_char.get(next_id, "")

        ids.append(next_id)

        current_text = "".join(id_to_char.get(i, "") for i in ids)

        if "</REPLY>" in current_text:
            break

    text = "".join(id_to_char.get(i, "") for i in ids)
    after = text.split("<REPLY>", 1)[-1]
    after = after.split("</REPLY>", 1)[0]

    return clean_for_voice(after)


def export_controller_onnx(model, path, device):
    model.eval()
    dummy = torch.zeros(1, CONFIG["controller_max_len"], dtype=torch.long, device=device)

    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input_ids"],
        output_names=[
            "intent_logits",
            "topic_logits",
            "safety_logits",
            "context_action_logits",
            "style_logits",
            "answer_key_logits",
        ],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "intent_logits": {0: "batch"},
            "topic_logits": {0: "batch"},
            "safety_logits": {0: "batch"},
            "context_action_logits": {0: "batch"},
            "style_logits": {0: "batch"},
            "answer_key_logits": {0: "batch"},
        },
        opset_version=17,
    )


def export_voice_onnx(model, path, device):
    model.eval()
    dummy = torch.zeros(1, CONFIG["voice_max_len"], dtype=torch.long, device=device)

    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )


# ============================================================
# Browser JS writer
# ============================================================

def write_browser_js(path: Path) -> None:
    js = r"""
// browser_meatball_hybrid.js
// Requires:
// <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web/dist/ort.min.js"></script>

function mbNormalizeText(text) {
  return String(text || "")
    .toLowerCase()
    .replace(/[“”]/g, '"')
    .replace(/[‘’]/g, "'")
    .replace(/[^a-z0-9?!.,'"\s:_/\-|=+#@&()%]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function mbBuildControllerInput(userText, memory = {}) {
  return (
    "last_intent=" + (memory.last_intent || "none") + " " +
    "last_project=" + (memory.last_project_key || "none") + " " +
    "last_category=" + (memory.last_category || "none") + " " +
    "user: " + mbNormalizeText(userText)
  );
}

function mbEncodeFixed(text, charToId, maxLen) {
  const unk = charToId["<UNK>"];
  const pad = charToId["<PAD>"];
  const ids = new BigInt64Array(maxLen);

  for (let i = 0; i < maxLen; i++) {
    if (i < text.length) {
      ids[i] = BigInt(charToId[text[i]] ?? unk);
    } else {
      ids[i] = BigInt(pad);
    }
  }

  return ids;
}

function mbEncodeVoiceContext(text, charToId, maxLen) {
  const unk = charToId["<UNK>"];
  const pad = charToId["<PAD>"];

  let chars = Array.from(String(text || ""));

  if (chars.length > maxLen) {
    chars = chars.slice(chars.length - maxLen);
  }

  const leftPad = maxLen - chars.length;
  const ids = new BigInt64Array(maxLen);

  for (let i = 0; i < maxLen; i++) {
    if (i < leftPad) {
      ids[i] = BigInt(pad);
    } else {
      ids[i] = BigInt(charToId[chars[i - leftPad]] ?? unk);
    }
  }

  return ids;
}

function mbSoftmax(logits, temperature = 1.0) {
  const arr = Array.from(logits).map(x => x / temperature);
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

function mbTopLabels(logits, labels, temperature = 1.0, n = 5) {
  const probs = mbSoftmax(logits, temperature);

  return probs
    .map((score, index) => ({ label: labels[index], score }))
    .sort((a, b) => b.score - a.score)
    .slice(0, n);
}

function mbPickTopK(logits, temperature = 0.72, topK = 24) {
  const arr = Array.from(logits);

  const indexed = arr.map((v, i) => ({ v, i }))
    .sort((a, b) => b.v - a.v)
    .slice(0, Math.min(topK, arr.length));

  const scaled = indexed.map(x => x.v / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);

  let r = Math.random();
  let acc = 0;

  for (let i = 0; i < indexed.length; i++) {
    acc += exps[i] / sum;
    if (r <= acc) return indexed[i].i;
  }

  return indexed[0].i;
}

function mbBuildVoicePrompt(userText, facts, style, safety, answerKey) {
  return (
    "<STYLE=" + style + "> " +
    "<SAFETY=" + safety + "> " +
    "<KEY=" + answerKey + "> " +
    "<USER>" + String(userText || "") + "</USER> " +
    "<FACTS>" + String(facts || "") + "</FACTS> " +
    "<REPLY>"
  );
}

function mbBadGeneratedText(text) {
  const t = String(text || "").trim();

  if (t.length < 2) return true;
  if (t.length > 360) return true;
  if (/[<>]{2,}/.test(t)) return true;
  if ((t.match(/undefined|null|nan/gi) || []).length) return true;

  const words = t.toLowerCase().split(/\s+/).filter(Boolean);
  if (words.length >= 8) {
    const counts = {};
    for (const w of words) counts[w] = (counts[w] || 0) + 1;
    const maxRepeat = Math.max(...Object.values(counts));
    if (maxRepeat >= Math.max(5, words.length * 0.35)) return true;
  }

  return false;
}

async function loadMeatballHybridBrain({
  controllerUrl = "/dist/meatball-hybrid/controller.onnx",
  controllerMetadataUrl = "/dist/meatball-hybrid/controller_metadata.json",
  voiceUrl = "/dist/meatball-hybrid/voice_generator.onnx",
  voiceMetadataUrl = "/dist/meatball-hybrid/voice_metadata.json",
  answerBankUrl = "/dist/meatball-hybrid/answer_bank.json"
} = {}) {
  const controllerMetadata = await fetch(controllerMetadataUrl).then(r => {
    if (!r.ok) throw new Error("Failed controller metadata " + r.status);
    return r.json();
  });

  const voiceMetadata = await fetch(voiceMetadataUrl).then(r => {
    if (!r.ok) throw new Error("Failed voice metadata " + r.status);
    return r.json();
  });

  const answerBank = await fetch(answerBankUrl).then(r => {
    if (!r.ok) throw new Error("Failed answer bank " + r.status);
    return r.json();
  });

  const controller = await ort.InferenceSession.create(controllerUrl, {
    executionProviders: ["wasm"]
  });

  const voice = await ort.InferenceSession.create(voiceUrl, {
    executionProviders: ["wasm"]
  });

  return {
    controller,
    voice,
    controllerMetadata,
    voiceMetadata,
    answerBank
  };
}

async function predictMeatballController(brain, userText, memory = {}) {
  const inputText = mbBuildControllerInput(userText, memory);
  const ids = mbEncodeFixed(
    inputText,
    brain.controllerMetadata.char_to_id,
    brain.controllerMetadata.config.controller_max_len
  );

  const tensor = new ort.Tensor(
    "int64",
    ids,
    [1, brain.controllerMetadata.config.controller_max_len]
  );

  const outputs = await brain.controller.run({ input_ids: tensor });
  const labels = brain.controllerMetadata.labels;

  const intentTop = mbTopLabels(outputs.intent_logits.data, labels.intent);
  const topicTop = mbTopLabels(outputs.topic_logits.data, labels.topic);
  const safetyTop = mbTopLabels(outputs.safety_logits.data, labels.safety);
  const contextTop = mbTopLabels(outputs.context_action_logits.data, labels.context_action);
  const styleTop = mbTopLabels(outputs.style_logits.data, labels.style);
  const answerTop = mbTopLabels(outputs.answer_key_logits.data, labels.answer_key);

  return {
    input_text: inputText,

    intent: intentTop[0].label,
    intent_confidence: intentTop[0].score,
    intent_top: intentTop,

    topic: topicTop[0].label,
    topic_confidence: topicTop[0].score,
    topic_top: topicTop,

    safety: safetyTop[0].label,
    safety_confidence: safetyTop[0].score,
    safety_top: safetyTop,

    context_action: contextTop[0].label,
    context_action_confidence: contextTop[0].score,
    context_action_top: contextTop,

    style: styleTop[0].label,
    style_confidence: styleTop[0].score,
    style_top: styleTop,

    answer_key: answerTop[0].label,
    answer_key_confidence: answerTop[0].score,
    answer_key_top: answerTop
  };
}

async function generateMeatballVoice(brain, prompt, options = {}) {
  const maxNewChars = options.maxNewChars || brain.voiceMetadata.config.max_new_chars || 220;
  const temperature = options.temperature || brain.voiceMetadata.config.temperature || 0.72;
  const topK = options.topK || brain.voiceMetadata.config.top_k || 24;

  const charToId = brain.voiceMetadata.char_to_id;
  const idToChar = brain.voiceMetadata.id_to_char;
  const maxLen = brain.voiceMetadata.config.voice_max_len;

  let text = String(prompt || "");

  for (let step = 0; step < maxNewChars; step++) {
    const ids = mbEncodeVoiceContext(text, charToId, maxLen);

    const tensor = new ort.Tensor(
      "int64",
      ids,
      [1, maxLen]
    );

    const outputs = await brain.voice.run({ input_ids: tensor });
    const logits = outputs.logits.data;

    const vocabSize = brain.voiceMetadata.vocab_size;
    const offset = (maxLen - 1) * vocabSize;
    const lastLogits = logits.slice(offset, offset + vocabSize);

    const nextId = mbPickTopK(lastLogits, temperature, topK);
    const ch = idToChar[String(nextId)] || "";

    text += ch;

    if (text.includes("</REPLY>")) break;
  }

  let reply = text.split("<REPLY>").pop() || "";
  reply = reply.split("</REPLY>")[0] || reply;
  reply = reply.replace(/\s+/g, " ").trim();

  return reply;
}

function pickCannedAnswer(brain, prediction) {
  const entry =
    brain.answerBank[prediction.answer_key] ||
    brain.answerBank.generic_answer ||
    null;

  if (!entry) return "I’m not totally sure what you mean. Ask that a little more clearly.";

  if (Array.isArray(entry.answers) && entry.answers.length) {
    return entry.answers[Math.floor(Math.random() * entry.answers.length)];
  }

  return entry.answer || "I’m not totally sure what you mean. Ask that a little more clearly.";
}

async function answerWithMeatballHybrid(brain, userText, memory = {}) {
  const prediction = await predictMeatballController(brain, userText, memory);

  const canned = pickCannedAnswer(brain, prediction);

  if (
    prediction.safety !== "safe" ||
    prediction.answer_key_confidence < 0.20 ||
    prediction.safety_confidence < 0.40
  ) {
    updateMeatballHybridMemory(memory, prediction, canned);
    return { text: canned, prediction, used_generator: false, fallback_reason: "safety_or_low_confidence" };
  }

  const prompt = mbBuildVoicePrompt(
    userText,
    canned,
    prediction.style,
    prediction.safety,
    prediction.answer_key
  );

  let generated = "";

  try {
    generated = await generateMeatballVoice(brain, prompt);
  } catch (err) {
    generated = "";
  }

  if (mbBadGeneratedText(generated)) {
    updateMeatballHybridMemory(memory, prediction, canned);
    return { text: canned, prediction, used_generator: false, fallback_reason: "bad_generation" };
  }

  updateMeatballHybridMemory(memory, prediction, generated);
  return { text: generated, prediction, used_generator: true, fallback_reason: "" };
}

function updateMeatballHybridMemory(memory, prediction, answerText) {
  memory.last_intent = prediction.intent || "none";
  memory.last_project_key = prediction.topic || "general_knowledge";
  memory.last_category = prediction.style || "conversation";
  memory.last_answer_text = answerText || "";
  memory.message_count = (memory.message_count || 0) + 1;

  if (prediction.safety !== "safe" || prediction.context_action === "soft_refusal") {
    memory.confusion_count = (memory.confusion_count || 0) + 1;
  } else {
    memory.confusion_count = Math.max(0, (memory.confusion_count || 0) - 1);
  }

  return memory;
}
"""
    path.write_text(js.strip() + "\n", encoding="utf-8")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+")
    parser.add_argument("--output-dir", default=CONFIG["output_dir"])
    parser.add_argument("--controller-epochs", type=int, default=CONFIG["controller_epochs"])
    parser.add_argument("--voice-epochs", type=int, default=CONFIG["voice_epochs"])
    parser.add_argument("--no-augment", action="store_true")

    args = parser.parse_args()

    CONFIG["output_dir"] = args.output_dir
    CONFIG["controller_epochs"] = args.controller_epochs
    CONFIG["voice_epochs"] = args.voice_epochs

    if args.no_augment:
        CONFIG["augment_enabled"] = False

    set_seed(CONFIG["seed"])

    paths = []

    for pattern in args.jsonl:
        matches = sorted(Path(".").glob(pattern))

        if matches:
            paths.extend(matches)
        elif Path(pattern).exists():
            paths.append(Path(pattern))
        else:
            print(f"[WARN] no match: {pattern}", file=sys.stderr)

    if not paths:
        raise SystemExit("No input files found.")

    raw_rows = []

    print("Input files:")

    for p in paths:
        file_rows = read_jsonl(p)
        print(f"  {p}: {len(file_rows)}")
        raw_rows.extend(file_rows)

    print(f"Raw rows: {len(raw_rows)}")

    examples, answer_bank = convert_rows(raw_rows)
    examples = augment_controller_examples(examples)

    print(f"Controller examples: {len(examples)}")
    print(f"Answer keys: {len(answer_bank)}")

    controller_texts = [ex["input_text"] for ex in examples]
    controller_char_to_id = build_char_vocab(controller_texts)

    label_maps = {
        "intent": build_label_map(examples, "intent"),
        "topic": build_label_map(examples, "topic"),
        "safety": build_label_map(examples, "safety"),
        "context_action": build_label_map(examples, "context_action"),
        "style": build_label_map(examples, "style"),
        "answer_key": build_label_map(examples, "answer_key"),
    }

    print("Controller labels:")
    for k, v in label_maps.items():
        print(f"  {k}: {len(v)}")

    strat = [ex["intent"] for ex in examples]
    counts = Counter(strat)

    can_stratify = all(v >= 2 for v in counts.values())

    if can_stratify:
        train_examples, val_examples = train_test_split(
            examples,
            test_size=CONFIG["validation_size"],
            random_state=CONFIG["seed"],
            stratify=strat,
        )
    else:
        train_examples, val_examples = train_test_split(
            examples,
            test_size=CONFIG["validation_size"],
            random_state=CONFIG["seed"],
        )

    train_controller_ds = ControllerDataset(train_examples, controller_char_to_id, label_maps)
    val_controller_ds = ControllerDataset(val_examples, controller_char_to_id, label_maps)

    train_controller_loader = torch.utils.data.DataLoader(
        train_controller_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
    )

    val_controller_loader = torch.utils.data.DataLoader(
        val_controller_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    controller = MeatballController(
        vocab_size=len(controller_char_to_id),
        label_sizes={k: len(v) for k, v in label_maps.items()},
        pad_id=controller_char_to_id["<PAD>"],
    )

    controller_params = sum(p.numel() for p in controller.parameters())
    print(f"Controller params: {controller_params:,} ({controller_params * 4 / 1024 / 1024:.2f} MB float32)")

    controller_report = train_controller(
        controller,
        train_controller_loader,
        val_controller_loader,
        train_examples,
        label_maps,
        device,
    )

    # Voice data
    voice_texts = []

    for ex in examples:
        facts = ex["facts"]
        output = facts

        # Keep generator output short.
        if len(output) > 320:
            output = output[:320].rsplit(" ", 1)[0] + "."

        train_text = build_voice_training_text(
            user=ex["raw_question"],
            facts=facts,
            style=ex["style"],
            safety=ex["safety"],
            answer_key=ex["answer_key"],
            output=output,
        )

        voice_texts.append(train_text)

    # Add default answer bank entries to voice training too.
    for key, entry in answer_bank.items():
        answer = entry.get("answer") or (entry.get("answers") or [""])[0]
        train_text = build_voice_training_text(
            user=f"answer using {key}",
            facts=answer,
            style=entry.get("style", "plain"),
            safety=entry.get("safety", "safe"),
            answer_key=key,
            output=answer,
        )
        voice_texts.append(train_text)

    voice_char_to_id = build_char_vocab(voice_texts)
    id_to_voice_char = {i: ch for ch, i in voice_char_to_id.items()}

    train_voice_texts, val_voice_texts = train_test_split(
        voice_texts,
        test_size=CONFIG["validation_size"],
        random_state=CONFIG["seed"],
    )

    train_voice_ds = VoiceDataset(train_voice_texts, voice_char_to_id)
    val_voice_ds = VoiceDataset(val_voice_texts, voice_char_to_id)

    train_voice_loader = torch.utils.data.DataLoader(
        train_voice_ds,
        batch_size=CONFIG["voice_batch_size"],
        shuffle=True,
    )

    val_voice_loader = torch.utils.data.DataLoader(
        val_voice_ds,
        batch_size=CONFIG["voice_batch_size"],
        shuffle=False,
    )

    voice = MeatballVoiceGenerator(
        vocab_size=len(voice_char_to_id),
        pad_id=voice_char_to_id["<PAD>"],
    )

    voice_params = sum(p.numel() for p in voice.parameters())
    print(f"Voice params: {voice_params:,} ({voice_params * 4 / 1024 / 1024:.2f} MB float32)")

    voice_report = train_voice(
        voice,
        train_voice_loader,
        val_voice_loader,
        device,
        pad_id=voice_char_to_id["<PAD>"],
    )

    sanity_inputs = [
        "last_intent=none last_project=none last_category=none user: hola",
        "last_intent=none last_project=none last_category=none user: tell me a joke",
        "last_intent=none last_project=none last_category=none user: tell me a dirty joke",
        "last_intent=none last_project=none last_category=none user: what is a tree",
        "last_intent=general_game_design last_project=general_knowledge last_category=creative user: brain not game",
        "last_intent=project_life_of_a_meatball last_project=life_of_a_meatball last_category=film_story user: yeah but what is it",
        "last_intent=none last_project=none last_category=none user: prove the riemann hypothesis",
        "last_intent=none last_project=none last_category=none user: who are you",
    ]

    sanity = []

    print("\nSanity:")

    id_to_char = id_to_voice_char

    for s in sanity_inputs:
        pred = predict_controller(controller, s, controller_char_to_id, label_maps, device)

        flat = {
            f: pred[f]["label"]
            for f in ["intent", "topic", "safety", "context_action", "style", "answer_key"]
        }

        answer_key = flat["answer_key"]
        facts = answer_bank.get(answer_key, {}).get("answer", DEFAULT_ANSWERS.get(answer_key, DEFAULT_ANSWERS["generic_answer"]))

        prompt = build_voice_prompt(
            user=s.split("user:", 1)[-1].strip(),
            facts=facts,
            style=flat["style"],
            safety=flat["safety"],
            answer_key=answer_key,
        )

        generated = generate_voice(
            voice,
            prompt,
            voice_char_to_id,
            id_to_char,
            device,
            max_new=160,
        )

        print(f"  {s}")
        print(f"    {flat}")
        print(f"    facts: {facts}")
        print(f"    generated: {generated}")

        sanity.append({
            "input": s,
            "prediction": pred,
            "facts": facts,
            "generated": generated,
        })

    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    controller_onnx = output_dir / "controller.onnx"
    voice_onnx = output_dir / "voice_generator.onnx"
    controller_metadata_path = output_dir / "controller_metadata.json"
    voice_metadata_path = output_dir / "voice_metadata.json"
    answer_bank_path = output_dir / "answer_bank.json"
    report_path = output_dir / "train_report.json"
    js_path = output_dir / "browser_meatball_hybrid.js"

    export_controller_onnx(controller.to(device), controller_onnx, device)
    export_voice_onnx(voice.to(device), voice_onnx, device)

    controller_metadata = {
        "model_type": "tiny_gated_token_mixer_controller",
        "config": CONFIG,
        "char_to_id": controller_char_to_id,
        "labels": label_maps,
        "param_count": controller_params,
        "input_files": [str(p) for p in paths],
    }

    voice_metadata = {
        "model_type": "tiny_causal_conv_voice_generator",
        "config": CONFIG,
        "char_to_id": voice_char_to_id,
        "id_to_char": {str(i): ch for i, ch in id_to_voice_char.items()},
        "vocab_size": len(voice_char_to_id),
        "param_count": voice_params,
    }

    compact_bank = {}

    for key, entry in answer_bank.items():
        compact_bank[key] = {
            "answer_key": key,
            "answer": entry.get("answer") or (entry.get("answers") or [""])[0],
            "answers": entry.get("answers") or [],
            "intent": entry.get("intent", "unknown"),
            "topic": entry.get("topic", "unknown"),
            "safety": entry.get("safety", "safe"),
            "style": entry.get("style", "plain"),
        }

    write_json(controller_metadata_path, controller_metadata)
    write_json(voice_metadata_path, voice_metadata)
    write_json(answer_bank_path, compact_bank)

    report = {
        "controller": controller_report,
        "voice": voice_report,
        "sanity": sanity,
        "raw_rows": len(raw_rows),
        "controller_examples": len(examples),
        "voice_examples": len(voice_texts),
        "answer_keys": len(answer_bank),
        "controller_params": controller_params,
        "voice_params": voice_params,
        "controller_float32_mb": controller_params * 4 / 1024 / 1024,
        "voice_float32_mb": voice_params * 4 / 1024 / 1024,
    }

    write_json(report_path, report)
    write_browser_js(js_path)

    print("\nSaved:")
    print(f"  {controller_onnx}")
    print(f"  {controller_metadata_path}")
    print(f"  {voice_onnx}")
    print(f"  {voice_metadata_path}")
    print(f"  {answer_bank_path}")
    print(f"  {js_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()