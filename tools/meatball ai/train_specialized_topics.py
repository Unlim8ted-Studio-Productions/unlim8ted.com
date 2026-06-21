import argparse
import importlib.util
import json
from pathlib import Path
import random


def load_trainer_module():
    module_path = Path(__file__).with_name("trained_slector_and_special.py")
    spec = importlib.util.spec_from_file_location(
        "trained_slector_and_special",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_clean_topic_rows(trainer, data_dir: Path, only_topic=None, limit_per_topic=0):
    topic_files = sorted(data_dir.glob("*.jsonl"))

    if not topic_files:
        raise RuntimeError(f"No .jsonl topic files found in {data_dir}")

    topic_rows = {}

    print()
    print("Loading topic datasets for specialist training...")

    for path in topic_files:
        topic = path.stem

        if only_topic and topic != only_topic:
            continue

        rows = trainer.load_jsonl(path)

        cleaned = []
        seen_questions = set()

        for row in rows:
            q = row["question"].strip()
            a = row["answer"].strip()

            if not q or not a:
                continue

            key = q.lower()
            if key in seen_questions:
                continue

            seen_questions.add(key)
            cleaned.append(
                {
                    "question": q,
                    "answer": a,
                }
            )

        if limit_per_topic and len(cleaned) > limit_per_topic:
            cleaned = random.sample(cleaned, limit_per_topic)

        if len(cleaned) < 10:
            print(f"[skip] {topic}: only {len(cleaned)} usable rows")
            continue

        topic_rows[topic] = cleaned
        print(f"{topic}: {len(cleaned)} rows")

    topics = sorted(topic_rows.keys())

    if not topics:
        raise RuntimeError("No usable topics loaded for specialist training.")

    return topic_rows, topics


def load_existing_manifest(out_dir: Path):
    manifest_path = out_dir / "manifest.json"

    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    return {
        "architecture": "MeatballAI selector plus topic-specialized original chunk models",
        "topics": {},
        "notes": [
            "Each topic has its own input_vocab.json and output_chunks.json.",
            "Specialized models use original mined chunk logic.",
            "Do not use one shared chunk vocabulary across topics.",
            "History is intentionally excluded from specialized model inputs.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default="assets/data/specialized_QA",
        help="Folder containing topic JSONL files used to train specialist models.",
    )

    parser.add_argument(
        "--out_dir",
        default="assets/models/specialized_meatball_chunks",
        help="Output folder for topic specialist files.",
    )

    parser.add_argument(
        "--limit_per_topic",
        type=int,
        default=0,
        help="Optional row limit per topic for quick tests.",
    )

    parser.add_argument(
        "--only_topic",
        default=None,
        help="Train only one topic, e.g. --only_topic art.",
    )

    parser.add_argument(
        "--general_only",
        action="store_true",
        help="Shortcut for training only the general topic.",
    )

    parser.add_argument(
        "--export_pt",
        action="store_true",
        help="Also export optional .pt checkpoint files alongside ONNX.",
    )

    args = parser.parse_args()

    if args.general_only:
        if args.only_topic and args.only_topic != "general":
            raise ValueError("--general_only conflicts with --only_topic unless --only_topic general is used.")
        args.only_topic = "general"

    trainer = load_trainer_module()

    print(f"device: {trainer.DEVICE}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    topics_dir = out_dir / "topics"

    out_dir.mkdir(parents=True, exist_ok=True)
    topics_dir.mkdir(parents=True, exist_ok=True)

    topic_rows, topics = load_clean_topic_rows(
        trainer,
        data_dir,
        only_topic=args.only_topic,
        limit_per_topic=args.limit_per_topic,
    )

    print()
    print(f"usable topics: {topics}")

    manifest = load_existing_manifest(out_dir)
    manifest["topics"] = manifest.get("topics", {})

    for topic in topics:
        topic_dir = topics_dir / topic

        trainer.train_topic_model(
            topic,
            topic_rows[topic],
            topic_dir,
            export_pt=args.export_pt,
        )

        manifest["topics"][topic] = {
            "dir": f"topics/{topic}",
            "model": f"topics/{topic}.onnx",
            "config": f"topics/{topic}/config.json",
            "input_vocab": f"topics/{topic}/input_vocab.json",
            "output_chunks": f"topics/{topic}/output_chunks.json",
        }

        if args.export_pt:
            manifest["topics"][topic]["model_pt"] = f"topics/{topic}/model.pt"

        trainer.save_json(out_dir / "manifest.json", manifest)

    trainer.save_json(out_dir / "manifest.json", manifest)

    print()
    print("DONE")
    print(f"topic output: {out_dir}")


if __name__ == "__main__":
    main()
