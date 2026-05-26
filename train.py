"""
===============================================================================
Script Name: train.py
Description: This script provides a CLI for training the model.
Author:      Adam Brida
Email:       248201@vutbr.cz
Date:        2026
License:     MIT
===============================================================================
"""

import sys
from pathlib import Path

# Add src directory path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse
import random

import numpy as np
import torch

from trainer import Trainer
from vocab import Vocab
from config import Config


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(description="Train CVAE-ScratchGAN model")

    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to training dataset file.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/",
        help="Directory for saving model checkpoints (default: checkpoints/).",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default=None,
        help="Path for vocabulary JSON (default: <checkpoint-dir>/vocab.json).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs (default: 30).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Training batch size (default: 1024).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of DataLoader worker processes (default: 1).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=None,
        help="Enable bfloat16 mixed precision during training.",
    )

    return parser.parse_args()


def _set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.

    Args:
        seed: seed value.

    Returns:
        None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_cli_overrides(config: Config, args: argparse.Namespace) -> Config:
    """
    Override config parameters with command-line arguments.

    Args:
        config: Configuration object.
        args: Parsed command-line arguments.

    Returns:
        Updated configuration object.
    """
    config.train_data_path = args.dataset_path
    config.checkpoint_dir = args.checkpoint_dir
    config.EPOCHS = args.epochs
    config.TRAIN_BATCH_SIZE = args.batch_size
    config.NUM_WORKERS = args.num_workers
    if args.seed is not None:
        config.SEED = args.seed
    if args.bf16 is not None:
        config.USE_BF16 = args.bf16

    return config


def main() -> None:
    """
    Run training.

    Returns:
        None
    """
    # Parse command-line arguments
    args = parse_args()

    # Load configuration and apply CLI overrides
    config = Config.load_yaml("config.yaml")
    config = _apply_cli_overrides(config, args)

    # Set random seeds
    if args.seed is not None:
        _set_seed(args.seed)

    # Ensure checkpoint directory exists
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Set vocab path
    vocab_dir = Path(config.checkpoint_dir)
    config.vocab_path = (
        args.vocab_path
        if args.vocab_path is not None
        else str(vocab_dir / "vocab.json")
    )

    # Build vocab
    Vocab(Path(config.train_data_path), Path(config.vocab_path).parent).build()

    # Initialize and run trainer
    trainer = Trainer(config)
    trainer.prepare()
    trainer.train()


if __name__ == "__main__":
    main()
