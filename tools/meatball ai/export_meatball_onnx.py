import argparse
from pathlib import Path

import torch

from train_input_and_alignment_models import (
    ALIGN_OUT_DIR,
    BOS,
    CORRECTOR_MAX_LEN,
    GreedyInputCorrector,
    INPUT_OUT_DIR,
    LABELS,
    TinyAlignmentClassifier,
    export_alignment_onnx,
    export_input_corrector_onnx,
    load_json,
)


def load_input_corrector(checkpoint_path, src_vocab_path, tgt_vocab_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    src_vocab = load_json(src_vocab_path, default={})
    tgt_vocab = load_json(tgt_vocab_path, default={})
    if not src_vocab or not tgt_vocab:
        raise RuntimeError(
            "Missing input/output vocab JSON for input corrector export."
        )

    src_vocab_size = int(checkpoint.get("src_vocab_size") or len(src_vocab))
    tgt_vocab_size = int(checkpoint.get("tgt_vocab_size") or len(tgt_vocab))
    max_len = int(checkpoint.get("max_len") or CORRECTOR_MAX_LEN)
    bos_id = int(checkpoint.get("bos_id", tgt_vocab[BOS]))

    model = GreedyInputCorrector(src_vocab_size, tgt_vocab_size, max_len, bos_id)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, max_len


def load_alignment_model(checkpoint_path, vocab_path, labels_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    vocab = load_json(vocab_path, default={})
    labels = load_json(labels_path, default=LABELS)
    if not vocab:
        raise RuntimeError("Missing input_vocab.json for alignment model export.")

    input_size = int(checkpoint.get("input_size") or len(vocab))
    num_classes = int(checkpoint.get("num_classes") or len(labels))
    state_dict = checkpoint["model_state_dict"]
    first_weight = state_dict.get("net.0.weight")
    hidden_size = int(first_weight.shape[0]) if first_weight is not None else 320

    model = TinyAlignmentClassifier(input_size, num_classes, hidden_size=hidden_size)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, input_size


def export_corrector(args):
    checkpoint_path = Path(args.corrector_pt)
    src_vocab_path = Path(args.corrector_src_vocab)
    tgt_vocab_path = Path(args.corrector_tgt_vocab)
    out_path = Path(args.corrector_onnx)

    model, max_len = load_input_corrector(
        checkpoint_path, src_vocab_path, tgt_vocab_path
    )
    export_input_corrector_onnx(model, out_path, max_len)
    print(out_path)


def export_alignment(args):
    checkpoint_path = Path(args.alignment_pt)
    vocab_path = Path(args.alignment_vocab)
    labels_path = Path(args.alignment_labels)
    out_path = Path(args.alignment_onnx)

    model, input_size = load_alignment_model(checkpoint_path, vocab_path, labels_path)
    export_alignment_onnx(model, out_path, input_size)
    print(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Export Meatball AI checkpoints to ONNX without retraining."
    )
    parser.add_argument(
        "--model",
        choices=["corrector", "alignment", "both"],
        default="both",
        help="Which model export to run.",
    )

    parser.add_argument(
        "--corrector-pt",
        default=str(INPUT_OUT_DIR / "input_text_corrector.pt"),
    )
    parser.add_argument(
        "--corrector-src-vocab",
        default=str(INPUT_OUT_DIR / "input_vocab.json"),
    )
    parser.add_argument(
        "--corrector-tgt-vocab",
        default=str(INPUT_OUT_DIR / "output_vocab.json"),
    )
    parser.add_argument(
        "--corrector-onnx",
        default=str(INPUT_OUT_DIR / "input_text_corrector.onnx"),
    )

    parser.add_argument(
        "--alignment-pt",
        default=str(ALIGN_OUT_DIR / "output_sanity_checker.pt"),
    )
    parser.add_argument(
        "--alignment-vocab",
        default=str(ALIGN_OUT_DIR / "input_vocab.json"),
    )
    parser.add_argument(
        "--alignment-labels",
        default=str(ALIGN_OUT_DIR / "labels.json"),
    )
    parser.add_argument(
        "--alignment-onnx",
        default=str(ALIGN_OUT_DIR / "output_sanity_checker.onnx"),
    )

    args = parser.parse_args()

    if args.model in ("corrector", "both"):
        export_corrector(args)
    if args.model in ("alignment", "both"):
        export_alignment(args)


if __name__ == "__main__":
    main()
