"""
===============================================================================
Script Name: generator.py
Description: This module implements the password generation.
Author:      Adam Brida
Email:       248201@vutbr.cz
Date:        2026
License:     MIT
===============================================================================
"""

from pathlib import Path

import torch
from tqdm import tqdm

from model import StructEncoder, PriorHead, Decoder
from pcfg import PCFGSampler
from tokenizer import StructTokenizer, PasswordTokenizer
from config import Config


class Generator:
    """
    Password generation.
    """

    def __init__(self, config: Config) -> None:
        """
        Constructor for the Generator class.

        Args:
            config: Configuration object.

        Returns:
            None
        """
        self.config = config
        self.plan = {}
        self.head_plan = {}
        self.tail_plan = {}

        # Initialised in prepare()
        self.struct_tokenizer = None
        self.password_tokenizer = None
        self.prior_encoder = None
        self.prior_head = None
        self.decoder = None

    @torch.no_grad()
    def _encode_prior(self, struct_masks: list) -> tuple:
        """
        Encode structure masks into prior distribution parameters.

        Args:
            struct_masks: List of structure masks.

        Returns:
            Tuple of (mu, log_var, h_c) where:
            mu: Tensor representing the mean of the prior distribution. Shape [B, LATENT_DIM].
            log_var: Tensor representing the log variance of the prior distribution. Shape [B, LATENT_DIM].
            h_c: Encoded structure representation from the StructEncoder. Shape [B, BIGRU_EMB_DIM].
        """
        # Encode structure masks
        struct_ids = [self.struct_tokenizer.encode_mask(m) for m in struct_masks]

        # Pad to max length and convert to tensor
        lengths = torch.tensor([len(x) for x in struct_ids], dtype=torch.long)
        max_len = int(lengths.max())
        struct_tensor = torch.tensor(
            [s + [self.config.PAD_IDX] * (max_len - len(s)) for s in struct_ids],
            dtype=torch.long,
            device=self.config.DEVICE,
        )

        # Encode with prior encoder and head
        with torch.amp.autocast(
            device_type=self.config.DEVICE,
            dtype=torch.bfloat16,
            enabled=self.config.USE_BF16,
        ):
            h_c = self.prior_encoder(struct_tensor, lengths)
            mu, log_var = self.prior_head(h_c)
        return mu, log_var, h_c

    @torch.no_grad()
    def _generate_from_z(
        self,
        z: torch.Tensor,
        h_c: torch.Tensor,
    ) -> list:
        """
        Generate passwords from latent vectors.

        Args:
            z: Sampled latent vector from the prior distribution. Shape [B, LATENT_DIM].
            h_c: Encoded structure representation from the StructEncoder. Shape [B, BIGRU_EMB_DIM].

        Returns:
            List of decoded passwords.
        """
        # Decode
        tokens = self.decoder.decode_from_z(
            z,
            h_c,
            max_len=self.config.MAX_GEN_PASS_LEN,
            temp=self.config.TEMP,
            top_k=self.config.TOP_K,
            top_p=self.config.TOP_P,
        )

        # Decode token IDs to password strings
        passwords = []
        for row in tokens.tolist():
            s = self.password_tokenizer.decode(row)
            if s:
                passwords.append(s)

        return passwords

    def prepare(self, plan: dict) -> None:
        """
        Initialise tokenizers, load trained model and build plans.

        Args:
            plan: Dictionary mapping structures to target counts.

        Returns:
            None
        """
        self.plan = plan

        # Create tokenizers
        self.struct_tokenizer = StructTokenizer(self.config)
        self.password_tokenizer = PasswordTokenizer(self.config)

        # Update vocab sizes
        self.config.character_vocab_size = self.password_tokenizer.vocab_size
        self.config.struct_vocab_size = self.struct_tokenizer.vocab_size

        # Load model weights
        state = torch.load(
            self.config.saved_model_path,
            map_location=self.config.DEVICE,
            weights_only=True,
        )

        def _load(module: torch.nn.Module, prefix: str) -> torch.nn.Module:
            """
            Load weights from the state dict.

            Args:
                module: The submodule to load weights into.
                prefix: Prefix used to extract the weights from the state dict.

            Returns:
                The module with loaded weights, moved to device, in eval mode.
            """
            sub = {
                k[len(prefix) + 1 :]: v
                for k, v in state.items()
                if k.startswith(prefix + ".")
            }
            module.load_state_dict(sub)
            return module.to(self.config.DEVICE).eval()

        # Load models
        self.prior_encoder = _load(StructEncoder(self.config), "prior_encoder")
        self.prior_head = _load(PriorHead(self.config), "prior_head")
        self.decoder = _load(Decoder(self.config), "decoder")

        # Split plan into head and tail plans
        threshold = self.config.GEN_BATCH_SIZE
        for k, v in self.plan.items():
            remaining = v
            while remaining >= threshold:
                remaining -= threshold
                self.head_plan[k] = self.head_plan.get(k, 0) + threshold
            self.tail_plan[k] = remaining

    def generate(
        self,
        pcfg_sampler: PCFGSampler = None,
        max_samples: int = None,
        output_path: str = None,
    ) -> None:
        """
        Generate passwords.

        Args:
            pcfg_sampler: PCFGSampler used for infinite-mode sampling.
            max_samples: Total number of passwords to generate.
            output_path: Path to the output file.

        Returns:
            None
        """
        # Create output directory
        out_file = Path(output_path) if output_path else None
        if out_file is not None:
            out_file.parent.mkdir(parents=True, exist_ok=True)

        # Open output file
        out_handle = None
        if out_file is not None:
            out_handle = open(out_file, "w", encoding="utf-8")

        # Progress bar
        pbar = (
            tqdm(total=max_samples, desc="Decoding")
            if max_samples is not None
            else None
        )

        try:
            # ***** Infinite mode *****
            if max_samples is None and (not self.plan):
                # Check if PCFG sampler is provided
                if pcfg_sampler is None:
                    return

                # Infinity loop
                while True:
                    # Sample structure masks for the batch
                    batch_masks = pcfg_sampler.sample(self.config.GEN_BATCH_SIZE)

                    # Encode unique masks only, then expand by index.
                    unique_masks, inverse_idx = [], {}
                    indices = []
                    for m in batch_masks:
                        if m not in inverse_idx:
                            inverse_idx[m] = len(unique_masks)
                            unique_masks.append(m)
                        indices.append(inverse_idx[m])
                    # Convert indices to tensor for indexing
                    idx_t = torch.tensor(indices, device=self.config.DEVICE)

                    # Get prior parameters for unique masks only
                    mu_u, log_var_u, h_c_u = self._encode_prior(unique_masks)

                    # Expand back to full batch
                    mu = mu_u[idx_t]
                    log_var = log_var_u[idx_t]
                    h_c = h_c_u[idx_t]

                    # Reparameterization trick
                    std = torch.exp(0.5 * log_var)
                    z = mu + std * torch.randn_like(std)

                    # Generate passwords
                    pws = self._generate_from_z(z, h_c)

                    for pw in pws:
                        print(pw)

            # ***** Finite mode *****
            # Process head plan first
            for mask, target_count in self.head_plan.items():
                if target_count <= 0:
                    continue

                # Get prior parameters for the current mask
                mu, log_var, h_c = self._encode_prior([mask])

                for i in range(0, target_count, self.config.GEN_BATCH_SIZE):
                    # Copy parameters for the current batch
                    mu_b = mu.repeat(self.config.GEN_BATCH_SIZE, 1)
                    std_b = torch.exp(
                        0.5 * log_var.repeat(self.config.GEN_BATCH_SIZE, 1)
                    )

                    # Reparameterization trick
                    z = mu_b + std_b * torch.randn_like(std_b)

                    # Generate passwords for the current batch
                    pws = self._generate_from_z(
                        z,
                        h_c.repeat(self.config.GEN_BATCH_SIZE, 1),
                    )

                    if pws:
                        out_handle.write("\n".join(pws) + "\n")

                    # Update progress bar
                    pbar.update(self.config.GEN_BATCH_SIZE)

            # Process tail plan
            # Create a list of masks according to the tail plan counts
            flat_tail_masks = []
            for mask, count in self.tail_plan.items():
                if count > 0:
                    flat_tail_masks.extend([mask] * count)

            for i in range(0, len(flat_tail_masks), self.config.GEN_BATCH_SIZE):
                # Get current batch
                batch_masks = flat_tail_masks[i : i + self.config.GEN_BATCH_SIZE]
                current_B = len(batch_masks)

                # Encode unique masks only, then expand by index.
                unique_masks, inverse_idx = [], {}
                indices = []
                for m in batch_masks:
                    if m not in inverse_idx:
                        inverse_idx[m] = len(unique_masks)
                        unique_masks.append(m)
                    indices.append(inverse_idx[m])
                # Convert indices to tensor for indexing
                idx_t = torch.tensor(indices, device=self.config.DEVICE)

                # Get prior parameters for unique masks only
                mu_u, log_var_u, h_c_u = self._encode_prior(unique_masks)

                # Expand back to full batch
                mu = mu_u[idx_t]
                log_var = log_var_u[idx_t]
                h_c = h_c_u[idx_t]

                # Reparameterization trick
                std = torch.exp(0.5 * log_var)
                z = mu + std * torch.randn_like(std)

                # Generate passwords for the current batch
                pws = self._generate_from_z(z, h_c)

                if pws:
                    out_handle.write("\n".join(pws) + "\n")

                # Update progress bar
                pbar.update(current_B)

        finally:
            # Close progress bar and output file
            if pbar is not None:
                pbar.close()
            if out_handle is not None:
                out_handle.close()
