# build_fragment_embeddings.py
import argparse
import json
import re
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def load_jsonl(path: Path):
    rows = []
    bad_lines = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
                else:
                    bad_lines += 1
                    print(f"Skipped non-object JSON at {path.name}:{line_num}")
            except Exception as e:
                bad_lines += 1
                print(f"Bad JSON skipped at {path.name}:{line_num} -> {e}")

    print(f"{path.name}: loaded {len(rows)} rows, skipped {bad_lines} bad lines")
    return rows


def clean_string(value):
    value = str(value or "")
    value = re.sub(r"\s+", " ", value).strip()
    return value


def normalize_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return [clean_string(x) for x in value if clean_string(x)]

    if isinstance(value, str):
        value = clean_string(value)
        if not value:
            return []
        return [value]

    return []


def require_id_text(row, source_name, index):
    frag_id = clean_string(row.get("id"))
    text = clean_string(row.get("text"))

    if not frag_id:
        raise ValueError(f"Missing id in {source_name} row {index}")

    if not text:
        raise ValueError(f"Missing text in {source_name} row {index}")

    return frag_id, text


def check_duplicate_ids(rows, source_name):
    seen = set()
    duplicates = []

    for i, row in enumerate(rows, 1):
        row_id = clean_string(row.get("id"))
        if not row_id:
            continue

        if row_id in seen:
            duplicates.append(row_id)

        seen.add(row_id)

    if duplicates:
        sample = duplicates[:20]
        raise ValueError(f"{source_name} has duplicate IDs. First duplicates: {sample}")


def normalize_fragment(row, index):
    frag_id, text = require_id_text(row, "fragments", index)

    topic = clean_string(row.get("topic", "general")).lower() or "general"
    role = clean_string(row.get("role", "fact")).lower() or "fact"
    tone = clean_string(row.get("tone", "clear")).lower() or "clear"
    tags = normalize_list(row.get("tags"))

    starts_as = (
        clean_string(row.get("starts_as", "complete_clause")) or "complete_clause"
    )
    ends_as = clean_string(row.get("ends_as", "complete_clause")) or "complete_clause"

    # Important:
    # This is what gets embedded.
    # It should describe the meaning of the fragment, not just the raw text.
    search_text_parts = [
        f"topic: {topic}",
        f"role: {role}",
        f"tone: {tone}",
    ]

    if tags:
        search_text_parts.append("tags: " + ", ".join(tags))

    search_text_parts.append("text: " + text)

    search_text = " | ".join(search_text_parts)

    return {
        "id": frag_id,
        "topic": topic,
        "role": role,
        "tone": tone,
        "tags": tags,
        "text": text,
        "starts_as": starts_as,
        "ends_as": ends_as,
        "search_text": search_text,
        "source_row": index,
    }


def normalize_bridge(row, index):
    bridge_id, text = require_id_text(row, "bridges", index)

    topic = clean_string(row.get("topic", "general")).lower() or "general"
    role = clean_string(row.get("role", "bridge")).lower() or "bridge"
    tone = clean_string(row.get("tone", "clear")).lower() or "clear"
    tags = normalize_list(row.get("tags"))

    requires_previous = clean_string(row.get("requires_previous", "complete_clause"))
    requires_next = clean_string(row.get("requires_next", "complete_clause"))

    return {
        "id": bridge_id,
        "topic": topic,
        "role": role,
        "tone": tone,
        "tags": tags,
        "text": text,
        "requires_previous": requires_previous,
        "requires_next": requires_next,
        "source_row": index,
    }


def l2_normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def make_bridge_groups(bridges):
    groups = {
        "all_bridge_ids": [],
        "by_topic": {},
        "by_role": {},
        "personality_ids": [],
        "question_ids": [],
        "uncertainty_ids": [],
        "emotion_ids": [],
        "memory_ids": [],
        "reasoning_ids": [],
        "relationship_ids": [],
        "explanation_ids": [],
    }

    for b in bridges:
        bid = b["id"]
        topic = b["topic"]
        role = b["role"]
        tone = b["tone"]
        tags = b["tags"]

        groups["all_bridge_ids"].append(bid)

        groups["by_topic"].setdefault(topic, []).append(bid)
        groups["by_role"].setdefault(role, []).append(bid)

        searchable = " ".join([topic, role, tone, " ".join(tags), b["text"]]).lower()

        if (
            "meatball" in searchable
            or "sauce" in searchable
            or "personality" in searchable
        ):
            groups["personality_ids"].append(bid)

        if topic == "question" or role == "question":
            groups["question_ids"].append(bid)

        if (
            topic == "uncertainty"
            or "uncertain" in searchable
            or "not sure" in searchable
        ):
            groups["uncertainty_ids"].append(bid)

        if topic == "emotion":
            groups["emotion_ids"].append(bid)

        if topic == "memory":
            groups["memory_ids"].append(bid)

        if topic == "reasoning":
            groups["reasoning_ids"].append(bid)

        if topic == "relationship":
            groups["relationship_ids"].append(bid)

        if topic == "explanation":
            groups["explanation_ids"].append(bid)

    return groups


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fragments", default="fragments.jsonl")
    parser.add_argument("--bridges", default="fragment_bridges.jsonl")

    parser.add_argument("--out-index", default="fragment_index.json")
    parser.add_argument("--out-embeddings", default="fragment_embeddings.npy")

    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model. all-MiniLM-L6-v2 is small and phone-friendly.",
    )

    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Save embeddings as int8 for smaller browser/mobile files.",
    )

    args = parser.parse_args()

    fragments_path = Path(args.fragments)
    bridges_path = Path(args.bridges)

    if not fragments_path.exists():
        raise FileNotFoundError(f"Could not find fragments file: {fragments_path}")

    if not bridges_path.exists():
        raise FileNotFoundError(f"Could not find bridges file: {bridges_path}")

    raw_fragments = load_jsonl(fragments_path)
    raw_bridges = load_jsonl(bridges_path)

    check_duplicate_ids(raw_fragments, fragments_path.name)
    check_duplicate_ids(raw_bridges, bridges_path.name)

    fragments = [normalize_fragment(row, i) for i, row in enumerate(raw_fragments, 1)]

    bridges = [normalize_bridge(row, i) for i, row in enumerate(raw_bridges, 1)]

    print()
    print(f"Normalized fragments for embedding: {len(fragments)}")
    print(f"Normalized bridges, not embedded: {len(bridges)}")

    if not fragments:
        raise ValueError("No fragments found to embed.")

    model = SentenceTransformer(args.model)

    texts = [f["search_text"] for f in fragments]

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    embeddings = l2_normalize(embeddings).astype("float32")

    if args.quantize:
        embeddings_to_save = np.clip(embeddings * 127, -127, 127).astype("int8")
        quantized = True
    else:
        embeddings_to_save = embeddings
        quantized = False

    bridge_groups = make_bridge_groups(bridges)

    index = {
        "version": 1,
        "embedding_model": args.model,
        "embedding_dim": int(embeddings.shape[1]),
        "quantized_int8": quantized,
        "notes": {
            "fragments_file": str(fragments_path),
            "bridges_file": str(bridges_path),
            "important": "Only fragments are embedded. Bridges are loaded separately and added after semantic retrieval.",
        },
        "counts": {
            "fragments": len(fragments),
            "bridges": len(bridges),
        },
        # Each embedding row corresponds to the fragment at the same index here.
        "fragments": [
            {
                "embedding_row": i,
                "id": f["id"],
                "topic": f["topic"],
                "role": f["role"],
                "tone": f["tone"],
                "tags": f["tags"],
                "text": f["text"],
                "starts_as": f["starts_as"],
                "ends_as": f["ends_as"],
            }
            for i, f in enumerate(fragments)
        ],
        # These are not embedded.
        "bridges": [
            {
                "id": b["id"],
                "topic": b["topic"],
                "role": b["role"],
                "tone": b["tone"],
                "tags": b["tags"],
                "text": b["text"],
                "requires_previous": b["requires_previous"],
                "requires_next": b["requires_next"],
            }
            for b in bridges
        ],
        "bridge_groups": bridge_groups,
    }

    Path(args.out_index).write_text(
        json.dumps(index, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    np.save(args.out_embeddings, embeddings_to_save)

    print()
    print("Done.")
    print(f"Wrote index: {args.out_index}")
    print(f"Wrote embeddings: {args.out_embeddings}")
    print(f"Embedding shape: {embeddings_to_save.shape}")
    print(f"Quantized: {quantized}")


if __name__ == "__main__":
    main()
