import argparse
import importlib.util
import math
import random
import statistics
import time
from pathlib import Path


def load_cover_module():
    module_path = Path(__file__).with_name("train_general_cover_chunks.py")
    spec = importlib.util.spec_from_file_location(
        "train_general_cover_chunks",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def format_seconds(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60.0:.1f}m"
    return f"{seconds / 3600.0:.2f}h"


def percentile(sorted_values, p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = (len(sorted_values) - 1) * p
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(sorted_values[lo])
    frac = idx - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specialized_dir", default="assets/data/specialized_QA")
    parser.add_argument("--smart_qa_path", default="tools/SmartMeatballQA.jsonl")
    parser.add_argument("--sample_rows", type=int, default=5000)
    parser.add_argument("--target_rows", type=int, default=200000)
    parser.add_argument("--max_chunk_words", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--superlinear_power",
        type=float,
        default=1.2,
        help="Exponent used for a pessimistic non-linear estimate.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    mod = load_cover_module()

    print("Loading combined rows...")
    t0 = time.perf_counter()
    rows = mod.load_combined_rows(Path(args.specialized_dir), Path(args.smart_qa_path))
    load_seconds = time.perf_counter() - t0

    total_rows = len(rows)
    sample_rows = min(args.sample_rows, total_rows)
    if sample_rows < 100:
        raise RuntimeError(f"Need at least 100 rows to estimate reliably, found {sample_rows}")

    if sample_rows < total_rows:
        sample = random.sample(rows, sample_rows)
    else:
        sample = list(rows)

    answer_lengths = sorted(len(mod.answer_tokenize(row["answer"])) for row in sample)
    avg_answer_len = statistics.mean(answer_lengths) if answer_lengths else 0.0
    p95_answer_len = percentile(answer_lengths, 0.95)

    print()
    print(f"load time:         {format_seconds(load_seconds)}")
    print(f"available rows:    {total_rows}")
    print(f"sample rows:       {sample_rows}")
    print(f"target rows:       {args.target_rows}")
    print(f"avg answer tokens: {avg_answer_len:.2f}")
    print(f"p95 answer tokens: {p95_answer_len:.2f}")

    print()
    print("Benchmarking expensive preprocessing stages...")

    t1 = time.perf_counter()
    tokenized_answers, occurrences, readable = mod.build_candidate_occurrences(
        sample,
        args.max_chunk_words,
    )
    candidate_seconds = time.perf_counter() - t1

    occurrence_count = sum(len(v) for v in occurrences.values())
    unique_chunks = len(occurrences)

    t2 = time.perf_counter()
    output_chunks = mod.build_cover_chunks(sample, max_chunk_words=args.max_chunk_words)
    cover_seconds = time.perf_counter() - t2

    key_to_id, max_chunk_words_used = mod.build_chunk_lookup(output_chunks)

    t3 = time.perf_counter()
    encoded_lengths = []
    unk_uses = 0
    for row in sample:
        ids = mod.encode_answer_to_chunks(row["answer"], key_to_id, max_chunk_words_used)
        stripped = mod.strip_special(ids)
        encoded_lengths.append(len(stripped))
        unk_uses += sum(1 for idx in stripped if idx == mod.UNK_ID)
    encode_seconds = time.perf_counter() - t3

    total_stage_seconds = candidate_seconds + cover_seconds + encode_seconds

    scale = max(args.target_rows / max(sample_rows, 1), 1.0)
    linear_estimate = total_stage_seconds * scale
    pessimistic_estimate = total_stage_seconds * (scale ** args.superlinear_power)

    print()
    print("Sample benchmark:")
    print(f"candidate mining:  {format_seconds(candidate_seconds)}")
    print(f"cover building:    {format_seconds(cover_seconds)}")
    print(f"encoding:          {format_seconds(encode_seconds)}")
    print(f"stage total:       {format_seconds(total_stage_seconds)}")

    print()
    print("Sample stats:")
    print(f"unique chunk candidates: {unique_chunks}")
    print(f"total occurrences:       {occurrence_count}")
    print(f"final output chunks:     {len(output_chunks)}")
    print(f"avg encoded chunks:      {statistics.mean(encoded_lengths):.2f}")
    print(f"encoded UNK uses:        {unk_uses}")

    print()
    print("Estimated preprocessing time for target rows:")
    print(f"linear estimate:         {format_seconds(linear_estimate)}")
    print(f"superlinear estimate:    {format_seconds(pessimistic_estimate)}")
    print(f"superlinear power:       {args.superlinear_power:.2f}")

    print()
    print("Notes:")
    print("- This estimates only chunk mining, cover building, and answer encoding.")
    print("- Training epochs are not included.")
    print("- Real scaling is often worse than linear because candidate and occurrence counts grow with corpus diversity.")


if __name__ == "__main__":
    main()
