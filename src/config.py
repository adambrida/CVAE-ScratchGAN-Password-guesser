"""
===============================================================================
Script Name: config.py
Description: This module loads configuration parameters from a YAML.
Author:      Adam Brida
Email:       248201@vutbr.cz
Date:        2026
License:     MIT
===============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import yaml


@dataclass
class Config:
    """Configuration parameters for training and generation."""

    # Paths
    vocab_path: str = None
    train_data_path: str = None
    checkpoint_dir: str = None
    pcfg_path: str = None
    saved_model_path: str = None
    output_path: str = None

    # Training parameters
    LOG_EVERY_BATCHES: int = 100
    EPOCHS: int = 30
    TRAIN_BATCH_SIZE: int = 1024
    NUM_WORKERS: int = 1
    USE_BF16: bool = False
    SEED: Optional[int] = None

    # Learning rates
    GEN_LR: float = 1e-4
    DIS_LR: float = 1e-3

    # KL annealing
    KL_RAMPUP_EPOCHS: int = 15
    KL_LOSS_WEIGHT: float = 0.1

    # Adversarial training
    DIS_START_EPOCH: int = 10
    ADV_START_EPOCH: int = 15
    ADV_RAMPUP_EPOCHS: int = 5
    ADV_LOSS_WEIGHT: float = 0.25

    # REINFORCE
    REINFORCE_GAMMA: float = 0.23
    REINFORCE_LAMBDA: float = 0.95

    # Generation sampling
    GEN_BATCH_SIZE: int = 10000
    MAX_GEN_PASS_LEN: int = 15
    NUM_GEN_SAMPLES: int = 1000000
    TOP_P: float = 0.85
    TOP_K: int = 35
    TEMP: float = 0.85

    # Password length constraints
    MIN_PASS_LEN: int = 4
    MAX_PASS_LEN: int = 15

    # Special token indices
    PAD_IDX: int = 0
    BOS_IDX: int = 1
    EOS_IDX: int = 2
    UNK_IDX: int = 3

    # Device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Architecture: BiGRU encoder
    BIGRU_EMB_DIM: int = 32
    BIGRU_HIDDEN_DIM: int = 256
    BIGRU_LAYERS: int = 2
    BIGRU_DROPOUT: float = 0.1

    # Architecture: Transformer encoder
    TRANSFORMER_EMB_DIM: int = 256
    TRANSFORMER_FF_DIM: int = 512
    TRANSFORMER_HEADS: int = 8
    TRANSFORMER_LAYERS: int = 3
    TRANSFORMER_DROPOUT: float = 0.1

    # Architecture: Latent space
    LATENT_DIM: int = 128

    # Architecture: GRU decoder
    DECODER_EMB_DIM: int = 256
    DECODER_HIDDEN_DIM: int = 512
    DECODER_LAYERS: int = 2
    DECODER_DROPOUT: float = 0.2
    WORD_DROPOUT_RATE: float = 0.2

    # Architecture: Discriminator
    DISCRIMINATOR_EMB_DIM: int = 256
    DISCRIMINATOR_HIDDEN_DIM: int = 512
    DISCRIMINATOR_LAYERS: int = 1
    DISCRIMINATOR_DROPOUT: float = 0.2

    # Filled at runtime
    character_vocab_size: Optional[int] = None
    struct_vocab_size: Optional[int] = None

    @classmethod
    def _filter_dict(cls, data: dict) -> dict:
        """
        Filter only keys that are defined in the dataclass.

        Args:
            data: Dictionary of configuration values.

        Returns:
            A filtered dictionary containing only keys that match the dataclass fields.
        """
        field_names = set(cls.__dataclass_fields__.keys())
        return {k: v for k, v in data.items() if k in field_names}

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        """
        Create Config from a dictionary and ignoring unknown keys.

        Args:
            data: Dictionary of configuration values.

        Returns:
            Config instance with values from the dictionary.
        """
        # Filter out unknown keys
        filtered = cls._filter_dict(data)
        inst = cls()
        for k, v in filtered.items():
            setattr(inst, k, v)

        # Handle device separately to ensure valid value
        device_value = str(getattr(inst, "DEVICE", "")).strip().lower()
        if device_value == "cpu":
            inst.DEVICE = "cpu"
        elif torch.cuda.is_available():
            inst.DEVICE = "cuda"
        else:
            inst.DEVICE = "cpu"
        return inst

    @classmethod
    def load_yaml(cls, path: str) -> Config:
        """
        Load configuration from a YAML file.

        Args:
            path: Path to the YAML file.

        Returns:
            Config instance with values from the YAML file.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
