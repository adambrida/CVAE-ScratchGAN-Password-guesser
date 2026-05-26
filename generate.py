"""
===============================================================================
Script Name: generate.py
Description: This script provides a CLI for generating passwords.
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

from generator import Generator
from pcfg import PCFG, PCFGSampler
from config import Config


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate passwords with CVAE-ScratchGAN model"
    )

    parser.add_argument(
        "--mode",
        choices=["build-pcfg", "generate", "infinite"],
        required=True,
        help="Operation mode.",
    )

    # build-pcfg args
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to dataset file (required for build-pcfg mode).",
    )
    parser.add_argument(
        "--pcfg-save-path",
        type=str,
        default="pcfg/",
        help="Directory to save the built PCFG (default: pcfg/).",
    )

    # generate/infinite args
    parser.add_argument(
        "--pcfg-path",
        type=str,
        default=None,
        help="Path to PCFG JSON file.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="generated_passwords.txt",
        help="Output file for finite generation (default: generated_passwords.txt).",
    )
    parser.add_argument(
        "--num-gen-samples",
        type=int,
        default=1_000_000,
        help="Number of passwords to generate (default: 1 000 000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10_000,
        help="Generation batch size (default: 10 000).",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.85,
        help="Sampling temperature (default: 0.85).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.85,
        help="Top-p sampling parameter (default: 0.85).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=35,
        help="Top-k sampling parameter (default: 35).",
    )
    parser.add_argument(
        "--max-gen-pass-len",
        type=int,
        default=15,
        help="Maximum generated password length (default: 15).",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=None,
        help="Enable bfloat16 mixed precision during generation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        default=None,
        help="Path to vocab.json. Defaults to vocab.json in the model checkpoint directory.",
    )

    # Structure filtering args
    parser.add_argument(
        "--policy-min-len",
        type=int,
        default=None,
        help="Minimum length of sampled structures.",
    )
    parser.add_argument(
        "--policy-max-len",
        type=int,
        default=None,
        help="Maximum length of sampled structures.",
    )
    parser.add_argument(
        "--policy-min-lower",
        type=int,
        default=None,
        help="Minimum number of lowercase letters.",
    )
    parser.add_argument(
        "--policy-min-upper",
        type=int,
        default=None,
        help="Minimum number of uppercase letters.",
    )
    parser.add_argument(
        "--policy-min-digit", type=int, default=None, help="Minimum number of digits."
    )
    parser.add_argument(
        "--policy-min-special",
        type=int,
        default=None,
        help="Minimum number of special characters.",
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
    if args.pcfg_path is not None:
        config.pcfg_path = args.pcfg_path
    if args.model_path is not None:
        config.saved_model_path = args.model_path
    config.output_path = args.output_path
    config.GEN_BATCH_SIZE = args.batch_size
    config.MAX_GEN_PASS_LEN = args.max_gen_pass_len
    config.NUM_GEN_SAMPLES = args.num_gen_samples
    config.TOP_P = args.top_p
    config.TOP_K = args.top_k
    config.TEMP = args.temp
    if args.dataset_path is not None:
        config.train_data_path = args.dataset_path
    if args.bf16 is not None:
        config.USE_BF16 = args.bf16
    if args.seed is not None:
        config.SEED = args.seed
    # Override vocab path if provided, otherwise
    # try to get it from model checkpoint directory
    if args.vocab_path is not None:
        config.vocab_path = args.vocab_path
    elif args.model_path is not None and not config.vocab_path:
        vocab = Path(args.model_path).parent / "vocab.json"
        if vocab.exists():
            config.vocab_path = str(vocab)
    return config


def main() -> None:
    """
    Run generation.

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

    # Build PCFG structures
    if args.mode == "build-pcfg":
        if not config.train_data_path:
            raise ValueError("--dataset-path is required for build-pcfg mode.")
        pcfg = PCFG(config.train_data_path, args.pcfg_save_path)
        pcfg.process()
        return

    # Ensure required parameters for generation
    if not config.pcfg_path:
        raise ValueError("--pcfg-path is required for generate/infinite mode.")
    if not config.saved_model_path:
        raise ValueError("--model-path is required for generate/infinite mode.")

    # Load PCFG and apply structure filters
    pcfg_sampler = PCFGSampler(config.pcfg_path)
    pcfg_sampler.apply_policy(
        min_len=args.policy_min_len,
        max_len=args.policy_max_len,
        min_lower=args.policy_min_lower,
        min_upper=args.policy_min_upper,
        min_digit=args.policy_min_digit,
        min_special=args.policy_min_special,
    )

    # Initialize generator
    generator = Generator(config)

    # Generate passwords
    if args.mode == "generate":
        plan = pcfg_sampler.get_counts(config.NUM_GEN_SAMPLES)
        generator.prepare(plan)
        generator.generate(
            max_samples=config.NUM_GEN_SAMPLES,
            output_path=config.output_path,
        )
        return

    if args.mode == "infinite":
        generator.prepare({})
        try:
            generator.generate(pcfg_sampler)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
