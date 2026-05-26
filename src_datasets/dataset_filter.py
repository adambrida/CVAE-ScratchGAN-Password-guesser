"""
===============================================================================
Script Name: dataset_filter.py
Description: This module filters password datasets.
Author:      Adam Brida
Email:       248201@vutbr.cz
Date:        2026
License:     MIT
===============================================================================
"""

import argparse
from pathlib import Path


def filter_passwords(
    input_path: Path,
    output_path: Path,
    min_len: int = 4,
    max_len: int = 15,
    split_colon: bool = False,
) -> tuple:
    """
    Reads an input file, applies filters and writes passwords to the output file.

    Args:
        input_path: The path to the input dataset file.
        output_path: The path where the filtered dataset will be saved.
        min_len: The minimum length of a password.
        max_len: The maximum length of a password.
        split_colon: If True, extracts only the part after the first colon.

    Returns:
        Tuple of (total lines read, total lines written).
    """
    prefiltered_count = 0
    filtered_count = 0

    # Create output directory, if it does not exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Open both files
    with (
        open(input_path, "r", encoding="utf-8", errors="ignore") as f_in,
        open(output_path, "w", encoding="utf-8") as f_out,
    ):

        for line in f_in:
            prefiltered_count += 1

            # Remove the newline character
            p = line.rstrip("\n")

            # Extracts only the part after the first colon if split_colon is True
            if split_colon and ":" in p:
                p = p.split(":", 1)[1]

            # Apply filters: not empty, valid length, only standard printable ASCII
            if p and min_len <= len(p) <= max_len and all(32 < ord(c) < 128 for c in p):
                f_out.write(p + "\n")
                filtered_count += 1

    return prefiltered_count, filtered_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter a dataset of passwords.")

    parser.add_argument(
        "--input-dataset-path",
        type=Path,
        required=True,
        help="Path to the input password dataset.",
    )
    parser.add_argument(
        "--output-dataset-path",
        type=Path,
        default="./passwords_cleaned.txt",
        help="Path to save the filtered passwords.",
    )
    parser.add_argument(
        "--min-len", type=int, default=4, help="Minimum length of passwords."
    )
    parser.add_argument(
        "--max-len", type=int, default=15, help="Maximum length of passwords."
    )
    parser.add_argument(
        "--split-colon",
        action="store_true",
        help="Extract only the text after the first colon (for hash:password formats).",
    )

    args = parser.parse_args()

    print(f"Started filtering passwords from {args.input_dataset_path}.")

    prefiltered_cnt, filtered_cnt = filter_passwords(
        input_path=args.input_dataset_path,
        output_path=args.output_dataset_path,
        min_len=args.min_len,
        max_len=args.max_len,
        split_colon=args.split_colon,
    )

    # Print statistics
    preserve_rate = (filtered_cnt / prefiltered_cnt) * 100 if prefiltered_cnt else 0
    print(f"Finished filtering! Saved to {args.output_dataset_path}.")
    print(
        f"Stats: {prefiltered_cnt} -> {filtered_cnt} = {preserve_rate:.2f} % preserved."
    )
