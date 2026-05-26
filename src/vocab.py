"""
===============================================================================
Script Name: vocab.py
Description: This module creates a vocabulary from a training dataset.
Author:      Adam Brida
Email:       248201@vutbr.cz
Date:        2026
License:     MIT
===============================================================================
"""

import json
from pathlib import Path

# Define special tokens
SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]


class Vocab:
    """
    Build a character vocabulary from a dataset and save it to JSON.
    """

    def __init__(self, dataset_path: Path, output_dir: Path) -> None:
        """
        Constructor for the Vocab class.

        Args:
            dataset_path: Path to the input dataset.
            output_dir: Path to the directory where vocab.json will be saved.

        Returns:
            None
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir

    def build(self) -> None:
        """
        Build the vocabulary from the dataset and write it to vocab.json.

        Returns:
            None
        """
        unique_chars = set()

        with open(self.dataset_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                unique_chars.update(line.strip())

        chars = sorted(unique_chars)
        vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS + chars)}

        # Create output directory if it does not exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        vocab_path = self.output_dir / "vocab.json"

        # Write the vocabulary to a JSON file
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
