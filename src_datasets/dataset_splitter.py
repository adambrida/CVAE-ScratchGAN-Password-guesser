"""
===============================================================================
Script Name: dataset_splitter.py
Description: Splits dataset into a training and testing set using a modulo.
Author:      Adam Brida
Email:       248201@vutbr.cz
Date:        2026
License:     MIT
===============================================================================
"""

import argparse
from pathlib import Path


def split_dataset(
    input_path: Path, train_path: Path, test_path: Path, train_ratio: float = 0.8
) -> tuple:
    """
    Splits the input dataset into training and testing sets.

    Args:
        input_path: Path to the input dataset.
        train_path: Path to save the train dataset.
        test_path: Path to save the test dataset.
        train_ratio: A proportion between 0.0 and 1.0.

    Returns:
        Tuple of (total lines, lines in train set, lines in test set).
    """
    train_count = 0
    test_count = 0

    threshold = int(train_ratio * 100)

    # Create output directories, if they do not exist
    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    # Open files
    with (
        open(input_path, "r", encoding="utf-8", errors="ignore") as f_in,
        open(train_path, "w", encoding="utf-8") as f_train,
        open(test_path, "w", encoding="utf-8") as f_test,
    ):

        for i, line in enumerate(f_in):
            # Decide whether the password should be in train or test set
            if (i % 100) < threshold:
                f_train.write(line)
                train_count += 1
            else:
                f_test.write(line)
                test_count += 1

    total_count = train_count + test_count
    return total_count, train_count, test_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split a dataset into train and test sets."
    )

    parser.add_argument(
        "--input-dataset-path",
        type=Path,
        required=True,
        help="Path to the input password dataset.",
    )
    parser.add_argument(
        "--train-output-path",
        type=Path,
        default="./train_passwords.txt",
        help="Path to save the training dataset.",
    )
    parser.add_argument(
        "--test-output-path",
        type=Path,
        default="./test_passwords.txt",
        help="Path to save the testing dataset.",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Ratio of training data."
    )

    args = parser.parse_args()

    if not (0.0 < args.train_ratio < 1.0):
        print("Error: --train-ratio must be between 0.0 and 1.0")
        exit(1)

    print(f"Started splitting passwords from {args.input_dataset_path}.")

    total, train_n, test_n = split_dataset(
        input_path=args.input_dataset_path,
        train_path=args.train_output_path,
        test_path=args.test_output_path,
        train_ratio=args.train_ratio,
    )

    print("\nFinished splitting!")
    print(f"Total lines processed: {total}")
    print(f"Train set: {train_n} lines saved to {args.train_output_path}")
    print(f"Test set:  {test_n} lines saved to {args.test_output_path}")
