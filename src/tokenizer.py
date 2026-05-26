"""
===============================================================================
Script Name: tokenizer.py
Description: This module defines the PasswordTokenizer and StructTokenizer.
Author:      Adam Brida
Email:       248201@vutbr.cz
Date:        2026
License:     MIT
===============================================================================
"""

import json

from config import Config


class StructTokenizer:
    """
    Tokenizer for password structures.
    """

    def __init__(self, config: Config) -> None:
        """
        Constructor for the StructTokenizer class.

        Args:
            config: Configuration object.

        Returns:
            None
        """
        self.max_pass_len = config.MAX_PASS_LEN

        # Define a vocabulary for password structures
        self.vocab = {
            "<PAD>": config.PAD_IDX,
            "<BOS>": config.BOS_IDX,
            "<EOS>": config.EOS_IDX,
            "N": 3,
            "L": 4,
            "U": 5,
            "S": 6,
        }
        self.vocab_size = len(self.vocab)

        # Invert the vocabulary for decoding
        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    @staticmethod
    def _password_to_structure(pwd: str) -> list:
        """
        Convert a password string to its structural representation.

        Args:
            pwd: The password.

        Returns:
            The structural representation of the password.
        """
        out = []
        for c in pwd:
            if c.isupper():
                t = "U"
            elif c.islower():
                t = "L"
            elif c.isdigit():
                t = "N"
            else:
                t = "S"
            out.append(t)
        return out

    def encode(self, text: str) -> list:
        """
        Encodes a password into a sequence of structure token IDs.

        Args:
            text: The input password.

        Returns:
            A list of structure token IDs.
        """
        # Truncate password
        if len(text) > self.max_pass_len:
            text = text[: self.max_pass_len]

        structure_tokens = self._password_to_structure(text)
        encoded_structure = [self.vocab[c] for c in structure_tokens]

        tokens = [self.vocab["<BOS>"]] + encoded_structure + [self.vocab["<EOS>"]]
        return tokens

    def decode(self, ids: list, skip_special_tokens: bool = True) -> str:
        """
        Decodes a sequence of structure token IDs.

        Args:
            ids: A list of structure token IDs.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            A decoded password structure.
        """
        pass_tokens = []

        for i in ids:
            char = self.inv_vocab.get(int(i), "")
            if skip_special_tokens and char in {"<BOS>", "<EOS>", "<PAD>"}:
                continue
            pass_tokens.append(char)

        return "".join(pass_tokens)

    def encode_mask(self, mask: str) -> list:
        """
        Encodes a password structure mask into a sequence of token IDs.

        Args:
            mask: The password structure.

        Returns:
            A list of structure token IDs.
        """
        ids = [self.vocab[char] for char in mask if char in {"L", "U", "N", "S"}]
        return [self.vocab["<BOS>"]] + ids + [self.vocab["<EOS>"]]


class PasswordTokenizer:
    """
    Tokenizer for passwords.
    """

    def __init__(self, config: Config) -> None:
        """
        Constructor for the PasswordTokenizer class.

        Args:
            config: Configuration object.

        Returns:
            None
        """
        self.max_pass_len = config.MAX_PASS_LEN
        self.vocab = {}
        self.inv_vocab = {}

        if config.vocab_path is not None:
            self._load_vocab(config.vocab_path)
        else:
            raise ValueError("Vocabulary path must be provided.")

        self.vocab_size = len(self.vocab)

    def _load_vocab(self, vocab_path: str) -> None:
        """
        Loads the vocabulary from a JSON file.

        Args:
            vocab_path: The path to the vocabulary JSON file.

        Returns:
            None
        """
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        self.inv_vocab = {int(v): k for k, v in self.vocab.items()}

    def encode(self, text: str) -> list:
        """
        Encodes a password into a sequence of token IDs.

        Args:
            text: The input password.

        Returns:
            A list of token IDs representing the input password.
        """
        # Truncate password
        if len(text) > self.max_pass_len:
            text = text[: self.max_pass_len]

        encoded_text = [self.vocab.get(c, self.vocab.get("<UNK>")) for c in text]
        tokens = [self.vocab["<BOS>"]] + encoded_text + [self.vocab["<EOS>"]]
        return tokens

    def decode(self, ids: list, skip_special_tokens: bool = True) -> str:
        """
        Decodes a sequence of token IDs back into a password.

        Args:
            ids: A list of token IDs to decode.
            skip_special_tokens: Whether to skip special tokens.

        Returns:
            A decoded password.
        """
        pass_tokens = []

        for i in ids:
            char = self.inv_vocab.get(int(i), "")
            if skip_special_tokens and char in {"<BOS>", "<EOS>", "<PAD>", "<UNK>"}:
                continue
            pass_tokens.append(char)

        return "".join(pass_tokens)
