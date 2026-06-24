import argparse
import json
import multiprocessing as mp
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DEFAULT_FRAGMENTS_DIR = Path("assets/data/fragments")
DEFAULT_OUTPUT_FILE = Path("assets/data/categories.jsonl")

COMMON_CATEGORY = "common"
UNKNOWN_CATEGORY = "i_cant_i_dont_know"

DEFAULT_TARGET_ROWS = 3000

SINGLE_CATEGORY_RATE = 0.70
MULTI_CATEGORY_RATE = 0.20
UNKNOWN_RATE = 0.10

MIN_MULTI_CATEGORIES = 2
MAX_MULTI_CATEGORIES = 3

FACTS_PER_CATEGORY = 6

MAX_NEW_TOKENS_QUESTION = 120
MAX_NEW_TOKENS_CHECKER = 80

MAX_GENERATION_TIME_SECONDS = 25

QUESTION_TEMPERATURE = 0.75
CHECKER_TEMPERATURE = 0.05
TOP_P = 0.95

GENERATOR_RETRIES_PER_TASK = 3

DEFAULT_WORKERS = 2


# ============================================================
# GLOBALS INSIDE EACH WORKER
# ============================================================

G_TOKENIZER = None
G_MODEL = None
G_CATEGORIES = None
G_ROUTER_CATEGORIES = None
G_RELATED_MAP = None
G_HAS_UNKNOWN = False
G_DEVICE = "cpu"


# ============================================================
# RELATED CATEGORY HINTS
# ============================================================

RELATED_GROUPS = {
    "unlim8ted": [
        "unlim8ted",
        "unlim8ted_games",
        "unlim8ted_films",
        "unlim8ted_music",
        "unlim8ted_products",
        "unlim8ted_hardware",
        "unlim8ted_phone",
        "life_of_a_meatball",
        "meatball",
        "glitch",
        "the_glitch",
        "timecat",
        "star_tracker",
        "square_pixels",
        "wise_size",
        "no_escape_for_you",
        "quiet_defiance",
        "website",
        "chatbot",
        "assistant",
        "project",
        "film",
        "game",
        "music",
    ],
    "animals": [
        "animal",
        "cat",
        "dog",
        "bird",
        "fish",
        "shark",
        "octopus",
        "tiger",
        "lion",
        "wolf",
        "bear",
        "snake",
        "frog",
        "bee",
        "ant",
        "spider",
        "whale",
        "dolphin",
        "chicken",
        "horse",
        "cow",
        "rabbit",
        "fox",
        "elephant",
        "turtle",
        "snail",
    ],
    "space": [
        "space",
        "sun",
        "moon",
        "star",
        "planet",
        "galaxy",
        "orbit",
        "comet",
        "meteor",
        "astronomy",
        "telescope",
        "rocket",
    ],
    "food": [
        "food",
        "pizza",
        "cheese",
        "meatball",
        "sauce",
        "pasta",
        "bread",
        "apple",
        "egg",
        "chicken",
        "cake",
        "cookie",
        "banana",
        "ice_cream",
        "coffee",
    ],
    "objects": [
        "tool",
        "knife",
        "brush",
        "spoon",
        "fork",
        "cup",
        "table",
        "chair",
        "box",
        "rope",
        "paper",
        "pen",
        "pencil",
        "book",
        "backpack",
        "umbrella",
        "glasses",
        "watch",
        "bicycle",
        "train",
        "car",
        "bus",
        "airplane",
        "shirt",
        "sock",
        "shoe",
        "broom",
        "bucket",
        "plate",
        "bottle",
        "key",
        "lock",
        "scissors",
        "screwdriver",
        "hammer",
        "pillow",
        "blanket",
        "towel",
        "coin",
    ],
    "home": [
        "toaster",
        "refrigerator",
        "vacuum_cleaner",
        "lightbulb",
        "battery",
        "wire",
        "computer",
        "camera",
        "phone",
        "keyboard",
        "speaker",
        "television",
        "lamp",
        "microwave",
        "sink",
        "door",
        "window",
        "toilet",
        "screen",
        "remote_control",
        "headphones",
        "mirror",
        "mouse",
    ],
    "nature": [
        "fire",
        "water",
        "rock",
        "tree",
        "flower",
        "rain",
        "cloud",
        "wind",
        "river",
        "ocean",
        "mountain",
        "forest",
        "grass",
        "snow",
        "ecology",
        "geology",
        "meteorology",
    ],
    "science": [
        "biology",
        "chemistry",
        "physics",
        "anatomy",
        "genetics",
        "evolution",
        "ecology",
        "geology",
        "meteorology",
        "astronomy",
        "neuroscience",
        "plate_tectonics",
    ],
    "technology": [
        "computer",
        "camera",
        "phone",
        "robot",
        "ai",
        "model",
        "neural",
        "dataset",
        "code",
        "software",
        "app",
        "website",
        "screen",
        "internet",
        "keyboard",
        "electronics",
        "headphones",
        "mouse",
    ],
}


# ============================================================
# BASIC HELPERS
# ============================================================


def log(msg: str) -> None:
    print(msg, flush=True)


def normalize_category(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


def category_to_words(category: str) -> str:
    return category.replace("_", " ")


def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def safe_sample(items: List[Any], count: int) -> List[Any]:
    if not items:
        return []
    return random.sample(items, min(count, len(items)))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []

    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                log(f"[WARN] Skipping bad JSON: {path} line {line_num}")
                continue

            if isinstance(row, dict):
                rows.append(row)

    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0

    count = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1

    return count


def load_seen_inputs(path: Path) -> set[str]:
    seen = set()

    if not path.exists():
        return seen

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception:
                continue

            value = row.get("input", row.get("question", ""))

            if isinstance(value, str) and value.strip():
                seen.add(value.strip().lower())

    return seen


# ============================================================
# LOAD FRAGMENT CATEGORIES
# ============================================================


def load_fragment_categories(fragments_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    if not fragments_dir.exists():
        raise FileNotFoundError(f"Fragments directory not found: {fragments_dir}")

    files = sorted(fragments_dir.glob("*.jsonl"))

    if not files:
        raise FileNotFoundError(f"No .jsonl files found in: {fragments_dir}")

    categories: Dict[str, List[Dict[str, Any]]] = {}

    for path in files:
        category = normalize_category(path.stem)
        rows = load_jsonl(path)

        fragments = []

        for row in rows:
            text = clean_text(row.get("text", row.get("content", "")))

            if not text:
                continue

            fragments.append(
                {
                    "id": str(row.get("id", "")).strip(),
                    "text": text,
                    "type": str(row.get("type", "")).strip(),
                    "category": category,
                }
            )

        if fragments:
            categories[category] = fragments

    return categories


def get_router_categories(categories: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    result = []

    for category in sorted(categories.keys()):
        if category == COMMON_CATEGORY:
            continue

        if category == UNKNOWN_CATEGORY:
            continue

        result.append(category)

    return result


def usable_fragments_for_category(
    categories: Dict[str, List[Dict[str, Any]]],
    category: str,
) -> List[Dict[str, Any]]:
    fragments = categories.get(category, [])

    usable = [
        f
        for f in fragments
        if f.get("text")
        and str(f.get("type", "")).lower() not in {"word", "punctuation"}
    ]

    return usable if usable else fragments


def sample_facts_for_target(
    categories: Dict[str, List[Dict[str, Any]]],
    target_categories: List[str],
) -> str:
    lines = []

    for category in target_categories:
        if category == UNKNOWN_CATEGORY:
            continue

        usable = usable_fragments_for_category(categories, category)
        sampled = safe_sample(usable, FACTS_PER_CATEGORY)

        lines.append(f"CATEGORY: {category}")

        for fragment in sampled:
            text = clean_text(fragment.get("text", ""))
            if text:
                lines.append(f"- {text[:240]}")

    return "\n".join(lines).strip()


# ============================================================
# RELATED CATEGORY TARGET SELECTION
# ============================================================


def get_group_for_category(category: str) -> Optional[str]:
    words = set(category.split("_"))

    for group_name, group_items in RELATED_GROUPS.items():
        normalized_items = {normalize_category(x) for x in group_items}

        if category in normalized_items:
            return group_name

        if words.intersection(normalized_items):
            return group_name

        if any(item in category for item in normalized_items):
            return group_name

    return None


def build_related_map(router_categories: List[str]) -> Dict[str, List[str]]:
    related: Dict[str, List[str]] = {}

    for category in router_categories:
        group = get_group_for_category(category)
        pool = set()

        if group:
            group_items = {normalize_category(x) for x in RELATED_GROUPS[group]}

            for other in router_categories:
                other_words = set(other.split("_"))

                if other == category:
                    pool.add(other)
                elif other in group_items:
                    pool.add(other)
                elif other_words.intersection(group_items):
                    pool.add(other)
                elif any(item in other for item in group_items):
                    pool.add(other)

        category_words = set(category.split("_"))

        for other in router_categories:
            other_words = set(other.split("_"))

            if other == category:
                pool.add(other)
            elif category_words.intersection(other_words):
                pool.add(other)

        related[category] = sorted(pool)

    return related


def choose_row_kind() -> str:
    r = random.random()

    if r < UNKNOWN_RATE:
        return "unknown"

    if r < UNKNOWN_RATE + MULTI_CATEGORY_RATE:
        return "multi"

    return "single"


def choose_single_target(router_categories: List[str]) -> List[str]:
    return [random.choice(router_categories)]


def choose_multi_target(
    router_categories: List[str],
    related_map: Dict[str, List[str]],
) -> Optional[List[str]]:
    anchors = router_categories[:]
    random.shuffle(anchors)

    for anchor in anchors:
        pool = [c for c in related_map.get(anchor, []) if c != anchor]

        if not pool:
            continue

        max_count = min(MAX_MULTI_CATEGORIES, len(pool) + 1)
        count = random.randint(MIN_MULTI_CATEGORIES, max_count)

        chosen = [anchor]
        chosen.extend(safe_sample(pool, count - 1))
        chosen = list(dict.fromkeys(chosen))

        if len(chosen) >= MIN_MULTI_CATEGORIES:
            return chosen

    return None


def choose_target_categories() -> Tuple[str, List[str]]:
    assert G_ROUTER_CATEGORIES is not None
    assert G_RELATED_MAP is not None

    kind = choose_row_kind()

    if kind == "unknown" and G_HAS_UNKNOWN:
        return "unknown", [UNKNOWN_CATEGORY]

    if kind == "multi":
        target = choose_multi_target(G_ROUTER_CATEGORIES, G_RELATED_MAP)
        if target:
            return "multi", target

    return "single", choose_single_target(G_ROUTER_CATEGORIES)


# ============================================================
# MODEL GENERATION
# ============================================================


def load_model_for_worker(model_name: str, device: str) -> Tuple[Any, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    dtype = torch.float16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=None,
        trust_remote_code=True,
    )

    model.to(device)
    model.eval()

    return tokenizer, model


def generate_chat(
    tokenizer: Any,
    model: Any,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=TOP_P,
            max_time=MAX_GENERATION_TIME_SECONDS,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================
# QUESTION EXTRACTION
# ============================================================


def extract_one_question(raw: str) -> str:
    text = clean_text(raw)

    text = re.sub(r"^```(?:text)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    # If model used labels.
    text = re.sub(
        r"^(question|input|user|prompt)\s*:\s*", "", text, flags=re.IGNORECASE
    ).strip()

    # Keep first line only.
    if "\n" in text:
        text = clean_text(text.splitlines()[0])

    text = text.strip().strip('"').strip("'").strip()

    # Remove accidental leading bullet or number.
    text = re.sub(r"^\s*[-*\d.)]+\s*", "", text).strip()

    return clean_text(text)


def extract_checker_verdict(raw: str) -> Tuple[bool, str]:
    text = clean_text(raw).lower()

    # Strong fail markers first.
    if re.search(r"\b(fail|reject|no|bad)\b", text):
        return False, raw[:120]

    if re.search(r"\b(pass|approve|yes|good)\b", text):
        return True, raw[:120]

    return False, f"unclear checker output: {raw[:120]}"


# ============================================================
# QUESTION GENERATOR / CHECKER
# ============================================================


def generate_question_for_target(
    target_kind: str,
    target_categories: List[str],
) -> str:
    assert G_TOKENIZER is not None
    assert G_MODEL is not None
    assert G_CATEGORIES is not None

    if target_categories == [UNKNOWN_CATEGORY]:
        system_prompt = """
You write one natural user question for a category router dataset.

The correct category is:
i_cant_i_dont_know

Write a question that should NOT use any normal knowledge category.

Good fallback questions are:
- impossible
- unknowable
- too vague
- private
- asking for unsupported hidden facts
- asking a weird impossible relationship

Rules:
- Output only the user question.
- Do not answer it.
- Do not explain.
- Do not mention category, dataset, fragments, or router.
"""

        user_prompt = """
Write ONE fallback user question.

Examples:
What secret thing did I forget to tell you?
Can you prove what I dreamed last night?
What is the password hidden inside the meatball's brain?
How are fire and cameras secretly the same animal?

Output only the question.
"""

    else:
        facts = sample_facts_for_target(G_CATEGORIES, target_categories)
        readable = ", ".join(category_to_words(c) for c in target_categories)
        category_list = ", ".join(target_categories)

        system_prompt = """
You write one natural user question for a category router dataset.

The router will read the question and select category files.

Critical rule:
Write a question that is within the target categories and nothing else.

Rules:
- Output only the user question.
- Do not answer it.
- Do not explain.
- Do not mention category, dataset, fragments, facts, or router.
- Do not write a statement.
- Do not add outside topics.
- If there is one target category, the question must clearly be about that category.
- If there are multiple target categories, the question must naturally need ALL target categories.
"""

        user_prompt = f"""
Target categories:
{category_list}

Readable target names:
{readable}

Allowed information:
{facts}

Write ONE natural user question that should route to exactly those target categories.

Bad:
Fire can be used for cooking?
That is a statement.

Bad:
What is shown in this image?
That adds an outside image topic.

Bad:
What is the latest phone?
That asks for current outside facts.

Output only the question.
"""

    raw = generate_chat(
        tokenizer=G_TOKENIZER,
        model=G_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=MAX_NEW_TOKENS_QUESTION,
        temperature=QUESTION_TEMPERATURE,
    )

    return extract_one_question(raw)


def check_question_for_target(
    question: str,
    target_categories: List[str],
) -> Tuple[bool, str]:
    assert G_TOKENIZER is not None
    assert G_MODEL is not None
    assert G_CATEGORIES is not None

    if target_categories == [UNKNOWN_CATEGORY]:
        evidence = "Fallback category. Question should not be answerable using normal categories."
    else:
        evidence = sample_facts_for_target(G_CATEGORIES, target_categories)

    system_prompt = """
You check one generated user question for a category router dataset.

Reply with exactly one word:
PASS
or
FAIL

PASS only if:
- the text is a natural user question or natural user command
- it is not a placeholder
- it is not a statement pretending to be a question
- it stays within the target categories and nothing else
- for multiple target categories, it needs all of them
- for i_cant_i_dont_know, it should not use normal categories

FAIL otherwise.

Do not explain unless you write FAIL reason after the word.
"""

    user_prompt = f"""
Target categories:
{json.dumps(target_categories, ensure_ascii=False)}

Question:
{question}

Allowed evidence:
{evidence}

Does the question correctly route to exactly the target categories?

Reply:
PASS
or
FAIL reason
"""

    raw = generate_chat(
        tokenizer=G_TOKENIZER,
        model=G_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_new_tokens=MAX_NEW_TOKENS_CHECKER,
        temperature=CHECKER_TEMPERATURE,
    )

    return extract_checker_verdict(raw)


# ============================================================
# LOCAL VALIDATION
# ============================================================


def is_probably_statement_with_question_mark(text: str) -> bool:
    text = clean_text(text)

    if not text.endswith("?"):
        return False

    no_mark = text[:-1].strip().lower()

    question_starts = (
        "what ",
        "why ",
        "how ",
        "when ",
        "where ",
        "who ",
        "which ",
        "can ",
        "could ",
        "would ",
        "should ",
        "is ",
        "are ",
        "do ",
        "does ",
        "did ",
        "tell me",
        "explain",
        "describe",
        "give me",
        "show me",
        "help me",
    )

    if no_mark.startswith(question_starts):
        return False

    return True


def local_input_ok(question: str) -> Tuple[bool, str]:
    question = clean_text(question)

    if not question:
        return False, "empty"

    if len(question) < 6:
        return False, "too short"

    if len(question) > 280:
        return False, "too long"

    lower = question.lower()

    banned = [
        "natural user prompt",
        "question here",
        "provided facts",
        "given facts",
        "allowed information",
        "target categories",
        "category",
        "dataset",
        "fragments",
        "router",
        "this image",
        "the image",
        "shown in the image",
        "latest",
        "current",
        "newest",
    ]

    for phrase in banned:
        if phrase in lower:
            return False, f"banned phrase: {phrase}"

    if not question.endswith("?"):
        command_starts = (
            "tell me",
            "explain",
            "describe",
            "give me",
            "show me",
            "help me",
        )

        if not lower.startswith(command_starts):
            return False, "not question or command"

    if is_probably_statement_with_question_mark(question):
        return False, "statement with question mark"

    return True, "ok"


def categories_ok(categories: List[str]) -> Tuple[bool, str]:
    assert G_ROUTER_CATEGORIES is not None

    if not categories:
        return False, "empty categories"

    if UNKNOWN_CATEGORY in categories:
        if categories == [UNKNOWN_CATEGORY] and G_HAS_UNKNOWN:
            return True, "ok"
        return False, "bad unknown category use"

    if len(categories) > MAX_MULTI_CATEGORIES:
        return False, "too many categories"

    for c in categories:
        if c == COMMON_CATEGORY:
            return False, "common selected"
        if c not in G_ROUTER_CATEGORIES:
            return False, f"not selectable category: {c}"

    return True, "ok"


def make_row(question: str, categories: List[str]) -> Dict[str, Any]:
    return {
        "input": clean_text(question),
        "history": [],
        "categories": categories,
    }


# ============================================================
# WORKER TASK
# ============================================================


def worker_init(
    model_name: str,
    fragments_dir_str: str,
    device: str,
    seed_base: int,
) -> None:
    global G_TOKENIZER
    global G_MODEL
    global G_CATEGORIES
    global G_ROUTER_CATEGORIES
    global G_RELATED_MAP
    global G_HAS_UNKNOWN
    global G_DEVICE

    worker_name = mp.current_process().name
    seed = seed_base + abs(hash(worker_name)) % 1000000
    random.seed(seed)
    torch.manual_seed(seed)

    G_DEVICE = device

    fragments_dir = Path(fragments_dir_str)

    G_CATEGORIES = load_fragment_categories(fragments_dir)
    G_ROUTER_CATEGORIES = get_router_categories(G_CATEGORIES)
    G_RELATED_MAP = build_related_map(G_ROUTER_CATEGORIES)
    G_HAS_UNKNOWN = UNKNOWN_CATEGORY in G_CATEGORIES

    G_TOKENIZER, G_MODEL = load_model_for_worker(model_name, device)


def worker_generate_one(task_id: int) -> Dict[str, Any]:
    try:
        for attempt in range(1, GENERATOR_RETRIES_PER_ROW + 1):
            target_kind, target_categories = choose_target_categories()

            ok, reason = categories_ok(target_categories)
            if not ok:
                continue

            question = generate_question_for_target(target_kind, target_categories)

            local_ok, local_reason = local_input_ok(question)
            if not local_ok:
                continue

            approved, check_reason = check_question_for_target(
                question, target_categories
            )

            if not approved:
                continue

            return {
                "status": "accepted",
                "row": make_row(question, target_categories),
                "kind": target_kind,
                "attempts": attempt,
            }

        return {
            "status": "rejected",
            "reason": "all retries failed",
        }

    except RuntimeError as e:
        if "out of memory" in str(e).lower() and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "error",
            "reason": f"RuntimeError: {e}",
        }

    except Exception as e:
        return {
            "status": "error",
            "reason": f"{type(e).__name__}: {e}",
        }


# ============================================================
# MAIN
# ============================================================


def run(args: argparse.Namespace) -> None:
    random.seed()

    fragments_dir = Path(args.fragments_dir)
    output_file = Path(args.output)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    log("============================================================")
    log("MULTI-WORKER single-row router dataset generator")
    log("============================================================")
    log(f"Model: {args.model}")
    log(f"Device per worker: {device}")
    log(f"Workers: {args.workers}")
    log(f"Fragments dir: {fragments_dir}")
    log(f"Output file: {output_file}")
    log(f"Target rows: {args.target}")
    log("============================================================")

    categories = load_fragment_categories(fragments_dir)
    router_categories = get_router_categories(categories)

    log(f"Actual .jsonl category files loaded: {len(categories)}")
    log(f"Router-selectable categories: {len(router_categories)}")
    log(f"Has {COMMON_CATEGORY}.jsonl: {COMMON_CATEGORY in categories}")
    log(f"Has {UNKNOWN_CATEGORY}.jsonl: {UNKNOWN_CATEGORY in categories}")

    if not router_categories:
        raise RuntimeError("No router-selectable categories found.")

    if args.resume:
        seen_inputs = load_seen_inputs(output_file)
        accepted = count_jsonl_rows(output_file)
    else:
        seen_inputs = set()
        accepted = 0

        if output_file.exists():
            backup = output_file.with_suffix(".jsonl.bak")
            output_file.rename(backup)
            log(f"Old output backed up to: {backup}")

    submitted = 0
    rejected = 0
    errors = 0

    single_count = 0
    multi_count = 0
    unknown_count = 0
    duplicate_rejects = 0

    started = time.time()

    log(f"Starting accepted rows: {accepted}")
    log("Loading one model inside each worker. This may take a minute.")
    log("============================================================")

    max_in_flight = max(args.workers * 2, args.workers)

    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=ctx,
        initializer=worker_init,
        initargs=(
            args.model,
            str(fragments_dir),
            device,
            int(time.time()),
        ),
    ) as executor:
        futures = set()

        while accepted < args.target:
            while (
                len(futures) < max_in_flight
                and accepted + len(futures) < args.target + max_in_flight
            ):
                submitted += 1
                futures.add(executor.submit(worker_generate_one, submitted))

            done, futures = wait(futures, return_when=FIRST_COMPLETED)

            for future in done:
                result = future.result()

                status = result.get("status")

                if status == "accepted":
                    row = result["row"]
                    input_key = row["input"].lower().strip()

                    if input_key in seen_inputs:
                        duplicate_rejects += 1
                        rejected += 1
                        continue

                    append_jsonl(output_file, row)
                    seen_inputs.add(input_key)

                    accepted += 1

                    cats = row["categories"]

                    if cats == [UNKNOWN_CATEGORY]:
                        unknown_count += 1
                    elif len(cats) == 1:
                        single_count += 1
                    else:
                        multi_count += 1

                    log(
                        f"[ACCEPT {accepted}/{args.target}] {json.dumps(row, ensure_ascii=False)}"
                    )

                elif status == "rejected":
                    rejected += 1

                else:
                    errors += 1
                    rejected += 1
                    log(f"[worker error] {result.get('reason', '')}")

                if accepted >= args.target:
                    break

            elapsed = time.time() - started
            rate = accepted / max(elapsed, 1)
            remaining = max(args.target - accepted, 0)
            eta_minutes = remaining / max(rate, 0.0001) / 60

            log("------------------------------------------------------------")
            log(f"Accepted: {accepted}/{args.target}")
            log(
                f"Single: {single_count} | Multi: {multi_count} | Unknown: {unknown_count}"
            )
            log(
                f"Rejected: {rejected} | Duplicates: {duplicate_rejects} | Errors: {errors}"
            )
            log(f"Rate: {rate:.4f} rows/sec")
            log(f"ETA: {eta_minutes:.1f} minutes")
            log("------------------------------------------------------------")

            if accepted >= args.target:
                break

    elapsed = time.time() - started

    log("\n============================================================")
    log("DONE")
    log("============================================================")
    log(f"Output: {output_file}")
    log(f"Accepted: {accepted}")
    log(f"Single-category rows: {single_count}")
    log(f"Multi-category rows: {multi_count}")
    log(f"Unknown rows: {unknown_count}")
    log(f"Rejected: {rejected}")
    log(f"Duplicate rejects: {duplicate_rejects}")
    log(f"Errors: {errors}")
    log(f"Elapsed minutes: {elapsed / 60:.2f}")
    log("============================================================")


# ============================================================
# ARGS
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default=MODEL_NAME)

    parser.add_argument("--fragments-dir", default=str(DEFAULT_FRAGMENTS_DIR))

    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_FILE))

    parser.add_argument("--target", type=int, default=DEFAULT_TARGET_ROWS)

    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)

    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing output file.",
    )

    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Back up old output and start fresh.",
    )

    parser.set_defaults(resume=True)

    return parser.parse_args()


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        log("\nStopped.")
        sys.exit(0)
