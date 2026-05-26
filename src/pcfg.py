"""
===============================================================================
Script Name: pcfg.py
Description: This module creates a PCFG of password structures from a training
             dataset and provides sampling utilities.
Author:      Adam Brida
Email:       248201@vutbr.cz
Date:        2026
License:     MIT
===============================================================================
"""

import json
import random
from collections import Counter
from pathlib import Path


class PCFG:
    """
    Build a PCFG of password structures from an input file.
    """

    def __init__(
        self,
        input_file: str,
        output_dir: str,
    ) -> None:
        """
        Constructor for the PCFG class.

        Args:
            input_file: Path to the input file with passwords.
            output_dir: Path to the output directory to save the PCFG JSON.

        Returns:
            None
        """
        self.input_file = input_file
        self.output_dir = output_dir

    @staticmethod
    def _password_to_structure(pwd: str) -> str:
        """
        Convert a password string to its structural representation.

        Args:
            pwd: The password.

        Returns:
            The structural representation of the password.
        """
        out = []
        for c in pwd:
            if c.islower():
                out.append("L")
            elif c.isupper():
                out.append("U")
            elif c.isdigit():
                out.append("N")
            else:
                out.append("S")
        return "".join(out)

    def _build_pcfg_structures(self, passwords: list) -> dict:
        """
        Build a PCFG of password structures.

        Args:
            passwords: A list of passwords.

        Returns:
            Dictionary mapping structure to probability.
        """
        structs = [self._password_to_structure(p) for p in passwords]
        counter = Counter(structs)
        total = sum(counter.values())
        pcfg = {s: c / total for s, c in counter.items()} if total > 0 else {}
        return pcfg

    def process(self) -> None:
        """
        Read passwords from input file, build PCFG and write to JSON.

        Returns:
            None
        """
        with open(self.input_file, "r", encoding="utf-8", errors="ignore") as f:
            passwords = [line.strip() for line in f if line.strip()]

        # Build PCFG
        pcfg = self._build_pcfg_structures(passwords)

        # Create output directory if it does not exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        pcfg_path = Path(self.output_dir) / "pcfg_structures.json"

        # Write PCFG to JSON
        with open(pcfg_path, "w", encoding="utf-8") as f:
            json.dump(pcfg, f, indent=2)


class PCFGSampler:
    """
    Load a PCFG JSON file and provide sampling utilities.
    """

    def __init__(self, json_path: str) -> None:
        """
        Constructor for the PCFGSampler class.

        Args:
            json_path: Path to the JSON file containing PCFG.

        Returns:
            None
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Load structures and probabilities
        self.structs = list(data.keys())
        self.probs = [float(data[s]) for s in self.structs]

    def sample(self, total_target: int) -> list:
        """
        Sample n structures from the PCFG distribution.

        Args:
            total_target: The total number of samples.

        Returns:
            A list of sampled structures.
        """
        return random.choices(self.structs, weights=self.probs, k=total_target)

    def get_counts(self, total_target: int) -> dict:
        """
        Compute counts per structure.

        Args:
            total_target: The total number of samples.

        Returns:
            Dictionary mapping structure to count.
        """
        sampled_list = random.choices(self.structs, weights=self.probs, k=total_target)
        return dict(Counter(sampled_list))

    def apply_policy(
        self,
        min_len: int = None,
        max_len: int = None,
        min_lower: int = None,
        min_upper: int = None,
        min_digit: int = None,
        min_special: int = None,
    ) -> None:
        """
        Filter structures. Raises ValueError if no structures remain.

        Args:
            min_len: Minimum structure length.
            max_len: Maximum structure length.
            min_lower: Minimum number of lowercase characters ('L').
            min_upper: Minimum number of uppercase characters ('U').
            min_digit: Minimum number of digit characters ('N').
            min_special: Minimum number of special characters ('S').

        Returns:
            None
        """
        filtered = [
            (s, p)
            for s, p in zip(self.structs, self.probs)
            if (min_len is None or len(s) >= min_len)
            and (max_len is None or len(s) <= max_len)
            and (min_lower is None or s.count("L") >= min_lower)
            and (min_upper is None or s.count("U") >= min_upper)
            and (min_digit is None or s.count("N") >= min_digit)
            and (min_special is None or s.count("S") >= min_special)
        ]
        if not filtered:
            raise ValueError("Policy constraints excluded all PCFG structures.")

        # Normalize probabilities after filtering
        self.structs, self.probs = map(list, zip(*filtered))
        total = sum(self.probs)
        self.probs = [p / total for p in self.probs]
