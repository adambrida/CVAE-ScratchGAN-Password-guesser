"""
===============================================================================
Script Name: dataset.py
Description: Password dataset class for training.
Author:      Adam Brida
Email:       248201@vutbr.cz
Date:        2026
License:     MIT
===============================================================================
"""

import torch
from torch.utils.data import Dataset

from config import Config
from tokenizer import StructTokenizer, PasswordTokenizer


class PasswordDataset(Dataset):
    """
    Dataset class for passwords.
    """

    def __init__(
        self,
        passwords: list,
        tok_struct: StructTokenizer,
        tok_pass: PasswordTokenizer,
        config: Config,
    ) -> None:
        """
        Constructor for the PasswordDataset class.

        Args:
            passwords: List of passwords.
            tok_struct: Tokenizer for password structures.
            tok_pass: Tokenizer for password characters.
            config: Configuration object.

        Returns:
            None
        """

        self.passwords = passwords
        self.tok_s = tok_struct
        self.tok_p = tok_pass
        self.max_len = config.MAX_PASS_LEN + 2  # MAX_PASS_LEN + BOS + EOS
        self.pad_idx = config.PAD_IDX
        self.character_vocab_size = config.character_vocab_size

    def __len__(self) -> int:
        """
        Returns the number of passwords in the dataset.

        Returns:
            The number of passwords in the dataset.
        """
        return len(self.passwords)

    def _pad(self, ids: list) -> tuple:
        """
        Pads the input list of token IDs to the maximum length.

        Args:
            ids: List of token IDs to be padded.

        Returns:
            Tuple of (padded list of token IDs, original length before padding).
        """
        length = len(ids)

        # Padding
        if length < self.max_len:
            ids = ids + [self.pad_idx] * (self.max_len - length)

        return ids, length

    def _extract_unique(self, pw: str) -> torch.Tensor:
        """
        Extracts unique character count from a password.

        Args:
            pw: The password.

        Returns:
            A tensor with the count of unique characters in the password. Shape: [1]
        """
        unique_count = torch.tensor([float(len(set(pw)))])
        return unique_count

    def _extract_bow(self, pw: str) -> torch.Tensor:
        """
        Extracts a Bag-of-Words vector from a password.

        Args:
            pw: The password.

        Returns:
            A tensor with 1.0 at each character index present in the password. Shape: [character_vocab_size]
        """
        bow = torch.zeros(self.character_vocab_size)
        for c in pw:
            idx = self.tok_p.vocab.get(c, self.tok_p.vocab["<UNK>"])
            if 0 <= idx < self.character_vocab_size:
                bow[idx] = 1.0
        return bow

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns the item at the specified index from the dataset.

        Args:
            idx: The index of the item.

        Returns:
            Tuple of (structure tokens, password tokens, structure length, unique character count, Bag-of-Words vector) where:
            structure tokens: Tensor of token IDs for the password structure. Shape: [max_len]
            password tokens: Tensor of token IDs for the password characters. Shape: [max_len]
            structure length: Tensor with the original length of the structure tokens before padding. Shape: [1]
            unique character count: Tensor with the count of unique characters in the password. Shape: [1]
            Bag-of-Words vector: Tensor with 1.0 at each character index present in the password. Shape: [character_vocab_size]
        """
        pw = self.passwords[idx]
        pw = pw[: self.max_len - 2]  # Truncate to max_len (without BOS and EOS)

        # Encode
        struct_ids = self.tok_s.encode(pw)
        pass_ids = self.tok_p.encode(pw)

        # Padding
        struct_ids_padded, struct_len = self._pad(struct_ids)
        pass_ids_padded, _ = self._pad(pass_ids)

        # Convert to tensors
        struct_tokens = torch.tensor(struct_ids_padded, dtype=torch.long)
        pass_tokens = torch.tensor(pass_ids_padded, dtype=torch.long)
        struct_len = torch.tensor(struct_len, dtype=torch.long)

        # Extract auxiliary features
        aux_unique = self._extract_unique(pw)
        bow = self._extract_bow(pw)

        return (
            struct_tokens,
            pass_tokens,
            struct_len,
            aux_unique,
            bow,
        )
