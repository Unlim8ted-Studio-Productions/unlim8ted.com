import argparse
import importlib.util
from pathlib import Path


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


def load_general_rows(trainer, data_dir: Path, limit_per_topic=0):
    general_path = data_dir / "general.jsonl"

    if not general_path.exists():
        raise FileNotFoundError(f"Missing general dataset: {general_path}")

    rows = trainer.load_jsonl(general_path)

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
        cleaned = cleaned[:limit_per_topic]

    if len(cleaned) < 10:
        raise RuntimeError(f"general has only {len(cleaned)} usable rows")

    return cleaned


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default="assets/data/specialized_QA",
        help="Folder containing specialized QA jsonl files.",
    )

    parser.add_argument(
        "--out_dir",
        default="assets/models/specialized_meatball_chunks",
        help="Output folder for specialized topic models.",
    )

    parser.add_argument(
        "--limit_per_topic",
        type=int,
        default=0,
        help="Optional row limit for quick tests.",
    )

    parser.add_argument(
        "--export_pt",
        action="store_true",
        help="Also export optional .pt checkpoint files alongside ONNX.",
    )

    parser.add_argument(
        "--disable_question_augmentation",
        action="store_true",
        help="Disable training-time question augmentation.",
    )

    parser.add_argument(
        "--max_question_augmentations",
        type=int,
        default=4,
        help="Maximum number of augmented question variants to add per training row.",
    )

    args = parser.parse_args()

    trainer = load_trainer_module()

    print(f"device: {trainer.DEVICE}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    topic_dir = out_dir / "topics" / "general"

    out_dir.mkdir(parents=True, exist_ok=True)
    topic_dir.mkdir(parents=True, exist_ok=True)

    rows = load_general_rows(
        trainer,
        data_dir,
        limit_per_topic=args.limit_per_topic,
    )

    augment_training_questions = not args.disable_question_augmentation
    max_question_augmentations = max(0, int(args.max_question_augmentations))

    print()
    print(f"general rows: {len(rows)}")
    print(f"question augmentation enabled: {augment_training_questions}")
    print(f"max question augmentations:   {max_question_augmentations}")

    trainer.train_topic_model(
        "general",
        rows,
        topic_dir,
        export_pt=args.export_pt,
        augment_training_questions=augment_training_questions,
        max_augmentations_per_question=max_question_augmentations,
    )

    print()
    print("DONE")
    print(f"general topic output: {topic_dir}")


if __name__ == "__main__":
    main()
