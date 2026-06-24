import json
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

CATEGORIES_DATASET = Path("assets/data/categories.jsonl")
FRAGMENTS_DIR = Path("assets/data/fragments")

OUTPUT_DIR = Path("assets/data/assembly_prompts")

QUESTIONS_PER_PROMPT = 100

INCLUDE_COMMON = True
COMMON_TOPIC = "common"
UNKNOWN_TOPIC = "i_cant_i_dont_know"

# 0 = include every fragment from each topic.
MAX_FRAGMENTS_PER_TOPIC = 0


# ============================================================
# HELPERS
# ============================================================


def normalize_topic(topic: str) -> str:
    return str(topic).strip().lower().replace(" ", "_").replace("-", "_")


def load_jsonl(path: Path):
    rows = []

    if not path.exists():
        return rows

    with path.open("r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[WARN] Bad JSON skipped: {path} line {line_num}")

    return rows


def get_question(row: dict) -> str:
    return str(row.get("question", row.get("input", ""))).strip()


def get_history(row: dict):
    history = row.get("history", [])

    if not isinstance(history, list):
        return []

    return history


def get_topics(row: dict):
    topics = row.get("topics", row.get("categories", []))

    if not isinstance(topics, list):
        return []

    cleaned = []

    for topic in topics:
        topic = normalize_topic(topic)

        if topic and topic not in cleaned:
            cleaned.append(topic)

    return cleaned


def load_fragments_for_topic(topic: str):
    path = FRAGMENTS_DIR / f"{topic}.jsonl"

    rows = load_jsonl(path)

    fragments = []

    for i, row in enumerate(rows):
        text = str(row.get("text", row.get("content", ""))).strip()

        if not text:
            continue

        frag_id = str(row.get("id", f"{topic}_{i:04d}")).strip()

        fragments.append(
            {
                "id": frag_id,
                "text": text,
            }
        )

    if MAX_FRAGMENTS_PER_TOPIC > 0:
        fragments = fragments[:MAX_FRAGMENTS_PER_TOPIC]

    return fragments


def load_all_fragments():
    topic_to_fragments = {}

    for path in sorted(FRAGMENTS_DIR.glob("*.jsonl")):
        topic = normalize_topic(path.stem)
        topic_to_fragments[topic] = load_fragments_for_topic(topic)

    return topic_to_fragments


def build_fragments_text(topics, topic_to_fragments):
    lines = []

    use_topics = []


    for topic in topics:
        if topic not in use_topics:
            use_topics.append(topic)

    for topic in use_topics:
        fragments = topic_to_fragments.get(topic, [])

        lines.append("")
        lines.append(f"### TOPIC: {topic}")

        if not fragments:
            lines.append("NO_FRAGMENTS_FOUND")
            continue

        for frag in fragments:
            lines.append(f"[{frag['id']}] {frag['text']}")

    return "\n".join(lines).strip()


def build_question_block(row, index, topic_to_fragments):
    question = get_question(row)
    history = get_history(row)
    topics = get_topics(row)

    fragments_text = build_fragments_text(topics, topic_to_fragments)

    history_text = "[]"

    if history:
        history_text = json.dumps(history, ensure_ascii=False)

    return f"""
============================================================
ITEM {index}
============================================================

QUESTION:
{question}

HISTORY:
{history_text}

TOPICS:
{", ".join(topics)}

AVAILABLE FRAGMENTS:
{fragments_text}

"""


def build_prompt(batch_rows, batch_index, topic_to_fragments):
    lines = []

    lines.append(f"ASSEMBLY PROMPT BATCH {batch_index}")
    lines.append("=" * 80)
    lines.append("")
    lines.append("You are creating answers for a fragment-based chatbot dataset.")
    lines.append("")
    lines.append("For EACH item below:")
    lines.append("- Use ONLY the AVAILABLE FRAGMENTS shown for that item.")
    lines.append("- Assemble a natural answer by combining the needed fragments.")
    lines.append("- Do NOT use outside knowledge.")
    lines.append("- Do NOT invent facts.")
    lines.append("- If the fragments cannot answer the question, only put this fragment in the answer:")
    lines.append(f"  {UNKNOWN_TOPIC}")
    lines.append("- You may also use any of these fragments on any of the questions:")
    lines.append('  "w_i":"i","w_yes":"yes","w_no":"no","w_know":"know","w_maybe":"maybe","w_not":"not","w_unknown":"unknown","w_public":"public","w_details":"details","w_limited":"limited","w_a":"a","w_an":"an","w_the":"the","w_this":"this","w_that":"that","w_it":"it","w_they":"they","w_there":"there","w_one":"one","w_thing":"thing","w_part":"part","w_is":"is","w_are":"are","w_was":"was","w_were":"were","w_be":"be","w_being":"being","w_means":"means","w_means_that":"means that","w_has":"has","w_have":"have","w_uses":"uses","w_involves":"involves","w_includes":"includes","w_explains":"explains","w_describes":"describes","w_studies":"studies","w_asks":"asks","w_helps":"helps","w_shows":"shows","w_connects":"connects","w_focuses":"focuses","w_depends":"depends","w_can":"can","w_may":"may","w_might":"might","w_should":"should","w_does":"does","w_do":"do","w_did":"did","w_about":"about","w_around":"around","w_with":"with","w_without":"without","w_from":"from","w_for":"for","w_to":"to","w_into":"into","w_through":"through","w_because":"because","w_if":"if","w_so":"so","w_and":"and","w_or":"or","w_but":"but","w_also":"also","w_while":"while","w_when":"when","w_where":"where","w_why":"why","w_how":"how","w_as":"as","w_by":"by","w_of":"of","w_on":"on","w_in":"in","w_at":"at","w_basically":"basically","w_actually":"actually","w_mostly":"mostly","w_probably":"probably","w_usually":"usually","w_simply":"simply","w_officially":"officially","w_safely":"safely","w_roughly":"roughly","w_clearly":"clearly","w_mainly":"mainly","w_generally":"generally","p_period":".","p_comma":",","p_question":"?","p_bang":"!","p_colon":":","p_semicolon":";","transition_basic":"Basically","transition_short_answer":"Short answer","transition_simple":"Simply put","transition_safe":"The safe answer","transition_from_known":"From what is known","transition_good_way":"A good way to say it","transition_main_idea":"The main idea","transition_one_thing":"One useful way to frame it","transition_in_unlim8ted_terms":"In Unlim8ted terms","transition_not_exact":"Not exactly","transition_yes":"Yes","transition_no":"No","transition_maybe":"Maybe","transition_about_that":"About that","transition_here":"Here is the clean version","joiner_and":"and","joiner_but":"but","joiner_so":"so","joiner_because", "type": "joiner", "text": "because","joiner_also","also",')
    lines.append("")
    lines.append("Return answers in this format for every item:")
    lines.append("")
    lines.append('{"answer":["fragment_id_1","fragment_id_2"]}')
    lines.append("")
    lines.append("=" * 80)
    lines.append("ITEMS")
    lines.append("=" * 80)

    for i, row in enumerate(batch_rows, start=1):
        lines.append(build_question_block(row, i, topic_to_fragments))

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(CATEGORIES_DATASET)
    topic_to_fragments = load_all_fragments()

    print(f"Loaded questions: {len(rows)}")
    print(f"Loaded fragment topics: {len(topic_to_fragments)}")

    batch_count = 0

    for start in range(0, len(rows), QUESTIONS_PER_PROMPT):
        batch_count += 1

        batch_rows = rows[start : start + QUESTIONS_PER_PROMPT]

        prompt_text = build_prompt(
            batch_rows=batch_rows,
            batch_index=batch_count,
            topic_to_fragments=topic_to_fragments,
        )

        out_path = OUTPUT_DIR / f"assembly_prompt_{batch_count:04d}.txt"

        with out_path.open("w", encoding="utf-8") as f:
            f.write(prompt_text)

        print(f"Saved {out_path} with {len(batch_rows)} questions")

    print(f"Done. Created {batch_count} txt prompt files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
