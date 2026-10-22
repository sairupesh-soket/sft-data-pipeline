#!/usr/bin/env python3
"""
Analyze token lengths of converted SWE-agent trajectories.

Reads a JSONL file in batches, tokenizes each trajectory, and computes
token length statistics including mean, median, std, min, max, and percentiles.

Splits output into multiple parquet files to avoid "List index overflow" errors.
"""

import json
import argparse
import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Iterator, List, Tuple, Optional, TextIO
from transformers import AutoTokenizer


def read_jsonl_streaming(filepath: str) -> Iterator[Tuple[str, dict]]:
    """Read JSONL file line by line, yielding (raw_line, parsed_data)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield line, data
            except json.JSONDecodeError:
                continue


def batch_items(items: Iterator, batch_size: int) -> Iterator[List]:
    """Batch items from an iterator."""
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def format_messages_for_tokenization(messages: List[dict]) -> str:
    """Format messages into a single string for tokenization."""
    formatted_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        formatted_parts.append(f"<|{role}|>\n{content}")
    return "\n".join(formatted_parts)


def tokenize_batch(tokenizer, trajectories: List[dict]) -> Tuple[List[int], List[List[int]]]:
    """
    Tokenize a batch of trajectories.
    Linearized mid-train rows use a `text` field; SFT rows use `messages` + chat template.
    """
    if trajectories and trajectories[0].get("text") and not trajectories[0].get("messages"):
        texts = [t.get("text") or "" for t in trajectories]
        encodings = tokenizer(texts, add_special_tokens=False)["input_ids"]
        return [len(ids) for ids in encodings], encodings

    input_ids_list = []
    lengths = []

    # Extract just the messages lists
    batch_messages = [traj.get("messages", []) for traj in trajectories]

    # Use apply_chat_template.
    # Modern 'transformers' versions support passing a batch (List[List[dict]]).
    # If your transformers version is older, you might need a list comprehension here.
    try:
        # Attempt batch processing (faster)
        encodings = tokenizer.apply_chat_template(
            batch_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False # We just want the IDs
        )
        # encodings is now a List[List[int]]
        input_ids_list = encodings
        lengths = [len(ids) for ids in encodings]

    except (TypeError, ValueError) as e:
        raise e
        # Fallback for older transformers or if batching is not supported for this tokenizer
        for messages in batch_messages:
            ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False
            )
            input_ids_list.append(ids)
            lengths.append(len(ids))

    return lengths, input_ids_list


def get_part_filename(base_path: str, part_index: int) -> str:
    """Converts 'data.parquet' -> 'data_part_000.parquet'"""
    path = Path(base_path)
    stem = path.stem
    suffix = path.suffix
    new_name = f"{stem}_part_{part_index:03d}{suffix}"
    return str(path.with_name(new_name))


def process_and_filter(
    tokenizer,
    input_path: str,
    batch_size: int,
    max_tokens: Optional[int] = None,
    output_file: Optional[TextIO] = None,
    parquet_output: Optional[str] = None,
    rows_per_file: int = 10000
) -> List[int]:
    """Process trajectories, filtering and writing to multiple parquet files."""
    all_lengths = []
    filtered_count = 0
    total_count = 0
    batch_num = 0

    # Buffer for parquet rows
    parquet_buffer = []
    parquet_file_index = 0

    # Stream through file
    stream = read_jsonl_streaming(input_path)

    for batch in batch_items(stream, batch_size):
        batch_num += 1
        raw_lines = [item[0] for item in batch]
        trajectories = [item[1] for item in batch]

        # Tokenize batch
        lengths, token_ids_list = tokenize_batch(tokenizer, trajectories)
        all_lengths.extend(lengths)
        total_count += len(lengths)

        # Filter logic
        if max_tokens is not None:
            for i, (raw_line, length, token_ids) in enumerate(zip(raw_lines, lengths, token_ids_list)):
                if length <= max_tokens:
                    # Write to JSONL if requested
                    if output_file is not None:
                        output_file.write(raw_line + '\n')

                    # Add to Parquet buffer if requested
                    if parquet_output is not None:
                        messages = trajectories[i].get("messages", [])
                        text = trajectories[i].get("text") or format_messages_for_tokenization(messages)
                        byte_size = len(text.encode('utf-8'))

                        parquet_buffer.append({
                            'repo_id': 0,
                            'repo_name': '',
                            'pr_id': 0,
                            'token_ids': [int(t) for t in token_ids],
                            'token_count': int(length),
                            'byte_size': int(byte_size)
                        })

                    filtered_count += 1

        print(f"Processed batch {batch_num}: {len(batch)} items, total: {total_count}", end="\r")

        # Check if we need to flush parquet buffer
        if parquet_output and len(parquet_buffer) >= rows_per_file:
            current_filename = get_part_filename(parquet_output, parquet_file_index)
            write_parquet(parquet_buffer, current_filename)
            print(f"\nFlushed {len(parquet_buffer)} records to {current_filename}")
            parquet_buffer = []  # Clear buffer
            parquet_file_index += 1

    print() # Newline after progress

    if max_tokens is not None:
        print(f"\nFiltering complete: {filtered_count}/{total_count} trajectories <= {max_tokens} tokens")

    # Flush remaining parquet data
    if parquet_output and parquet_buffer:
        current_filename = get_part_filename(parquet_output, parquet_file_index)
        write_parquet(parquet_buffer, current_filename)
        print(f"Flushed remaining {len(parquet_buffer)} records to {current_filename}")

    return all_lengths


def write_parquet(data: List[dict], output_path: str):
    """Write data to parquet file with TokenizedPRData schema."""
    if not data:
        return

    # Define schema
    schema = pa.schema([
        ('repo_id', pa.int64()),
        ('repo_name', pa.string()),
        ('pr_id', pa.int64()),
        ('token_ids', pa.list_(pa.int32())),
        ('token_count', pa.int32()),
        ('byte_size', pa.int32()),
    ])

    # Build arrays
    try:
        repo_ids = pa.array([d['repo_id'] for d in data], type=pa.int64())
        repo_names = pa.array([d['repo_name'] for d in data], type=pa.string())
        pr_ids = pa.array([d['pr_id'] for d in data], type=pa.int64())
        token_ids = pa.array([d['token_ids'] for d in data], type=pa.list_(pa.int32()))
        token_counts = pa.array([d['token_count'] for d in data], type=pa.int32())
        byte_sizes = pa.array([d['byte_size'] for d in data], type=pa.int32())

        table = pa.Table.from_arrays(
            [repo_ids, repo_names, pr_ids, token_ids, token_counts, byte_sizes],
            schema=schema
        )

        # Use compression to help with size
        pq.write_table(table, output_path, compression='SNAPPY')

    except Exception as e:
        print(f"\nError writing parquet file {output_path}: {e}")
        # Hint about memory if it fails here
        print("Tip: Try reducing --rows-per-file if you encounter memory issues during write.")
        raise e


def compute_statistics(lengths: List[int]) -> dict:
    """Compute statistics for token lengths."""
    arr = np.array(lengths)
    if len(arr) == 0:
        return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0,
                "median": 0, "p5": 0, "p25": 0, "p75": 0, "p90": 0, "p95": 0, "p99": 0}

    stats = {
        "count": len(arr),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }
    return stats


def print_statistics(stats: dict):
    """Print statistics in a formatted way."""
    if stats['count'] == 0:
        print("\nNo data to compute statistics.")
        return

    print("\n" + "=" * 50)
    print("TOKEN LENGTH STATISTICS")
    print("=" * 50)
    print(f"{'Count:':<20} {stats['count']:>15,}")
    print(f"{'Mean:':<20} {stats['mean']:>15,.2f}")
    print(f"{'Std:':<20} {stats['std']:>15,.2f}")
    print(f"{'Min:':<20} {stats['min']:>15,}")
    print(f"{'Max:':<20} {stats['max']:>15,}")
    print("-" * 50)
    print("PERCENTILES")
    print("-" * 50)
    print(f"{'5%:':<20} {stats['p5']:>15,.2f}")
    print(f"{'25% (Q1):':<20} {stats['p25']:>15,.2f}")
    print(f"{'50% (Median):':<20} {stats['median']:>15,.2f}")
    print(f"{'75% (Q3):':<20} {stats['p75']:>15,.2f}")
    print(f"{'90%:':<20} {stats['p90']:>15,.2f}")
    print(f"{'95%:':<20} {stats['p95']:>15,.2f}")
    print(f"{'99%:':<20} {stats['p99']:>15,.2f}")
    print("=" * 50)


def print_retention_analysis(lengths: List[int]):
    """Print retention analysis by token length thresholds."""
    arr = np.array(lengths)
    total_records = len(arr)
    if total_records == 0:
        return

    total_tokens = int(np.sum(arr))
    thresholds = [8192, 16384, 32768, 65536, 131072]

    print("\n" + "=" * 70)
    print("RETENTION ANALYSIS BY TOKEN LENGTH")
    print("=" * 70)

    for threshold in thresholds:
        mask = arr <= threshold
        records_retained = int(np.sum(mask))
        tokens_retained = int(np.sum(arr[mask]))

        records_pct = 100.0 * records_retained / total_records
        tokens_pct = 100.0 * tokens_retained / total_tokens if total_tokens > 0 else 0

        print(f"  <= {threshold:>6} tokens: {records_retained:>10} records ({records_pct:>5.1f}%), "
              f"{tokens_retained:>15} tokens ({tokens_pct:>5.1f}%)")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Analyze token lengths and save to chunked Parquet files")
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument("-m", "--model", default="Qwen/Qwen2.5-72B", help="Tokenizer model name")
    parser.add_argument("-b", "--batch-size", type=int, default=1024, help="Batch size for reading")
    parser.add_argument("-o", "--output", help="Save statistics to JSON file")
    parser.add_argument("--max-tokens", type=int, default=None, help="Filter max tokens")
    parser.add_argument("--filtered-output", help="Output JSONL file for filtered trajectories")
    parser.add_argument("--parquet-output", help="Output parquet file base name (e.g., data.parquet)")
    parser.add_argument("--rows-per-file", type=int, default=20000,
                       help="Number of rows per Parquet file to avoid list index overflow (default: 20000)")

    args = parser.parse_args()

    if (args.filtered_output or args.parquet_output) and args.max_tokens is None:
        parser.error("--filtered-output and --parquet-output require --max-tokens")

    print(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"Processing file: {args.input}")

    output_file = None
    if args.filtered_output:
        output_file = open(args.filtered_output, 'w', encoding='utf-8')

    try:
        all_lengths = process_and_filter(
            tokenizer,
            args.input,
            args.batch_size,
            args.max_tokens,
            output_file,
            args.parquet_output,
            args.rows_per_file
        )
    finally:
        if output_file:
            output_file.close()

    if not all_lengths:
        print("No trajectories found in input file!")
        return

    stats = compute_statistics(all_lengths)
    print_statistics(stats)
    print_retention_analysis(all_lengths)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)

if __name__ == "__main__":
    main()