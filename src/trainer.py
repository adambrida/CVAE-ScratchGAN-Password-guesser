"""
===============================================================================
Script Name: trainer.py
Description: This module defines the Trainer class, which encapsulates the
             training loop for the CVAE model and discriminator.
Author:      Adam Brida
Email:       248201@vutbr.cz
Date:        2026
License:     MIT
===============================================================================
"""

import csv
import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import CVAE_Model, Discriminator
from dataset import PasswordDataset
from tokenizer import StructTokenizer, PasswordTokenizer
from config import Config

# Metrics for logging
METRIC_KEYS = ("total", "ce", "kl", "aux", "d_loss", "g_adv", "d_real", "d_fake")


class Trainer:
    """
    Train CVAE-ScratchGAN.
    """

    def __init__(self, config: Config) -> None:
        """
        Constructor for the Trainer class.

        Args:
            config: Configuration object.

        Returns:
            None
        """
        self.config = config

        # Initialised in prepare()
        self.struct_tokenizer = None
        self.password_tokenizer = None
        self.dataset = None
        self.dataloader = None
        self.model = None
        self.discriminator = None
        self.optimizer_model = None
        self.optimizer_dis = None
        self.pos_weight = None

        # REINFORCE baseline
        self.b_i = 0.0

        # Set in _init_logging()
        self.logger = None
        self.metrics_epoch_path = None
        self.metrics_batch_path = None

    def _init_logging(self) -> None:
        """
        Initialize logging.

        Returns:
            None
        """
        # Create checkpoint directory if it doesn't exist
        log_dir = Path(self.config.checkpoint_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger("trainer")
        self.logger.setLevel(logging.INFO)
        # Clear existing handlers
        self.logger.handlers.clear()

        # Set logging format
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-5s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console logging
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

        # Create CSV files for metrics
        self.metrics_epoch_path = log_dir / "training_metrics_epoch.csv"
        self.metrics_batch_path = log_dir / "training_metrics_batch.csv"

        if not self.metrics_epoch_path.exists():
            with open(self.metrics_epoch_path, "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(["epoch", *METRIC_KEYS])

        if not self.metrics_batch_path.exists():
            with open(self.metrics_batch_path, "w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(["epoch", "batch", *METRIC_KEYS])

    def _log_batch(self, epoch: int, batch_idx: int, window: dict, n: int) -> None:
        """
        Write batch metrics to CSV.

        Args:
            epoch: Current epoch number.
            batch_idx: Current batch index.
            window: Accumulated metric sums over the last n batches.
            n: Number of batches in the window.

        Returns:
            None
        """
        avg = {k: window[k] / n for k in METRIC_KEYS}
        with open(self.metrics_batch_path, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([epoch, batch_idx, *avg.values()])

    def _log_epoch(self, epoch: int, avg: dict) -> None:
        """
        Write epoch metrics to CSV and log them.

        Args:
            epoch: Current epoch number.
            avg: Dictionary of averaged metric values for the epoch.

        Returns:
            None
        """
        with open(self.metrics_epoch_path, "a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow([epoch, *avg.values()])
        self.logger.info(
            "Epoch %d | Total %.4f  CE %.4f  KL %.4f  aux %.4f  "
            "D %.4f  G_adv %.4f  D(real) %.4f  D(fake) %.4f",
            epoch,
            avg["total"],
            avg["ce"],
            avg["kl"],
            avg["aux"],
            avg["d_loss"],
            avg["g_adv"],
            avg["d_real"],
            avg["d_fake"],
        )

    def prepare(self) -> None:
        """
        Instantiate tokenizers, dataset, dataloader, models and optimisers.

        Returns:
            None
        """
        # Tokenizers
        self.struct_tokenizer = StructTokenizer(self.config)
        self.password_tokenizer = PasswordTokenizer(self.config)
        self.config.character_vocab_size = self.password_tokenizer.vocab_size
        self.config.struct_vocab_size = self.struct_tokenizer.vocab_size

        # BoW weights
        self.pos_weight = torch.full(
            (self.config.character_vocab_size,), 50.0, device=self.config.DEVICE
        )

        # Dataset + dataloader
        with open(
            self.config.train_data_path, "r", encoding="utf-8", errors="ignore"
        ) as f:
            passwords = [x.strip() for x in f if x.strip()]

        self.dataset = PasswordDataset(
            passwords, self.struct_tokenizer, self.password_tokenizer, self.config
        )

        # Set seed for dataloader workers
        # https://docs.pytorch.org/docs/stable/notes/randomness.html
        if self.config.SEED is not None:

            def seed_worker(worker_id: int) -> None:
                worker_seed = torch.initial_seed() % 2**32
                np.random.seed(worker_seed)
                random.seed(worker_seed)

            g = torch.Generator()
            g.manual_seed(self.config.SEED)
        else:
            seed_worker = None
            g = None

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.TRAIN_BATCH_SIZE,
            num_workers=self.config.NUM_WORKERS,
            persistent_workers=self.config.NUM_WORKERS > 0,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

        # Models
        self.model = CVAE_Model(self.config).to(self.config.DEVICE)
        self.discriminator = Discriminator(self.config).to(self.config.DEVICE)

        # Optimizers
        self.optimizer_model = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.GEN_LR,
            fused=True,
        )
        self.optimizer_dis = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=self.config.DIS_LR,
            betas=(0.5, 0.999),
            fused=True,
        )

    def _get_kl_weight(self, epoch: int) -> float:
        """
        Compute KL weight for the current epoch.

        Args:
            epoch: Current epoch.

        Returns:
            KL weight in the range [0, KL_LOSS_WEIGHT].
        """
        return (
            min(epoch / self.config.KL_RAMPUP_EPOCHS, 1.0) * self.config.KL_LOSS_WEIGHT
        )

    def _get_adversarial_weight(self, epoch: int) -> float:
        """
        Compute the adversarial loss weight for the current epoch.

        Args:
            epoch: Current epoch.

        Returns:
            Adversarial weight in the range [0, ADV_LOSS_WEIGHT].
        """
        start = max(1, int(self.config.ADV_START_EPOCH))
        if epoch + 1 < start:
            return 0.0
        return (
            min((epoch + 1 - start) / self.config.ADV_RAMPUP_EPOCHS, 1.0)
            * self.config.ADV_LOSS_WEIGHT
        )

    def _compute_losses(
        self,
        logits: torch.Tensor,
        target_seq: torch.Tensor,
        aux_out: dict,
        aux_unique_target: torch.Tensor,
        mu_post: torch.Tensor,
        log_var_post: torch.Tensor,
        mu_prior: torch.Tensor,
        log_var_prior: torch.Tensor,
        kl_weight: float,
        bow_target: torch.Tensor,
    ) -> dict:
        """
        Compute the CVAE loss.

        Args:
            logits: Decoder output logits. Shape [B, T, vocab_size].
            target_seq: Real password token IDs. Shape [B, T].
            aux_out: Dictionary with auxiliary head outputs.
            aux_unique_target: Target unique-character count. Shape [B].
            mu_post: Posterior mean. Shape [B, LATENT_DIM].
            log_var_post: Posterior log variance. Shape [B, LATENT_DIM].
            mu_prior: Prior mean. Shape [B, LATENT_DIM].
            log_var_prior: Prior log variance. Shape [B, LATENT_DIM].
            kl_weight: Current weight for the KL divergence.
            bow_target: Bag-of-words target. Shape [B, vocab_size].

        Returns:
            Dictionary with 'total', 'ce', 'kl', 'aux'.
        """

        # Cross-entropy with label smoothing
        recon = F.cross_entropy(
            logits.permute(0, 2, 1),
            target_seq,
            ignore_index=self.config.PAD_IDX,
            label_smoothing=0.1,
        )

        # KL divergence
        posterior = Normal(mu_post, torch.exp(0.5 * log_var_post))
        prior = Normal(mu_prior, torch.exp(0.5 * log_var_prior))
        kl = kl_divergence(posterior, prior).sum(dim=1).mean()  # mean over batch

        # Auxiliary losses
        aux_unique = F.mse_loss(aux_out["unique"], aux_unique_target)
        aux_bow = F.binary_cross_entropy_with_logits(
            aux_out["bow"], bow_target, pos_weight=self.pos_weight
        )
        aux = aux_unique + aux_bow

        # Total CVAE loss
        total = recon + kl_weight * kl + aux
        return {"total": total, "ce": recon, "kl": kl, "aux": aux}

    def _step_discriminator(
        self,
        target_seq: torch.Tensor,
        gen_tokens: torch.Tensor,
        pad_mask: torch.Tensor,
        gen_pad_mask: torch.Tensor,
    ) -> tuple:
        """
        Discriminator update step.

        Args:
            target_seq: Real password token IDs. Shape [B, T].
            gen_tokens: Generated token IDs. Shape [B, T].
            pad_mask: Padding mask for real sequences (1 = real token). Shape [B, T].
            gen_pad_mask: Padding mask for generated sequences (1 = real token). Shape [B, T].

        Returns:
            Tuple of (d_loss, d_real, d_fake) where:
            d_loss: Sum of real and fake discriminator losses.
            d_real: Mean discriminator score on real passwords.
            d_fake: Mean discriminator score on fake passwords.
        """
        # Clear discriminator gradients from previous step
        self.optimizer_dis.zero_grad()

        # Get discriminator scores for real passwords
        d_real_logits = self.discriminator(target_seq)
        # Compute cross-entropy loss (only non-padding tokens)
        loss_real = F.binary_cross_entropy_with_logits(
            d_real_logits,
            torch.ones_like(d_real_logits),
            weight=pad_mask,
            reduction="sum",
        ) / pad_mask.sum().clamp(
            min=1
        )  # Compute mean over non-padding tokens and prevent division by zero

        # Get discriminator scores for generated passwords (detach to avoid backprop to CVAE)
        d_fake_logits = self.discriminator(gen_tokens.detach())
        # Compute cross-entropy loss (only non-padding tokens)
        loss_fake = F.binary_cross_entropy_with_logits(
            d_fake_logits,
            torch.zeros_like(d_fake_logits),
            weight=gen_pad_mask,
            reduction="sum",
        ) / gen_pad_mask.sum().clamp(
            min=1
        )  # Compute mean over non-padding tokens and prevent division by zero

        # Total discriminator loss
        d_loss = loss_real + loss_fake
        # Compute gradients
        d_loss.backward()
        # Update discriminator parameters
        self.optimizer_dis.step()

        # Compute mean discriminator scores on real and fake passwords
        d_real = (
            torch.sigmoid(d_real_logits) * pad_mask
        ).sum().item() / pad_mask.sum().clamp(min=1).item()
        d_fake = (
            torch.sigmoid(d_fake_logits) * gen_pad_mask
        ).sum().item() / gen_pad_mask.sum().clamp(min=1).item()
        return d_loss.item(), d_real, d_fake

    def _compute_reinforce_loss(
        self,
        gen_tokens: torch.Tensor,
        log_probs: torch.Tensor,
        gen_pad_mask: torch.Tensor,
        B: int,
        T: int,
    ) -> torch.Tensor:
        """
        Compute the REINFORCE adversarial loss and update baseline (from ScratchGAN paper).

        Args:
            gen_tokens: Generated token IDs. Shape [B, T].
            log_probs: Log-probabilities of generated tokens. Shape [B, T].
            gen_pad_mask: Padding mask for generated sequences (1 = real token). Shape [B, T].
            B: Batch size.
            T: Sequence length.

        Returns:
            REINFORCE loss tensor.
        """
        # Compute dense rewards r_t
        with torch.no_grad():
            r_t = 2.0 * torch.sigmoid(self.discriminator(gen_tokens)) - 1.0  # [B, T]

        # Discounted sum R_t
        R_t = torch.zeros_like(r_t)
        running = torch.zeros(B, device=self.config.DEVICE)
        for t in reversed(range(T)):
            running = (
                r_t[:, t] * gen_pad_mask[:, t] + self.config.REINFORCE_GAMMA * running
            )
            R_t[:, t] = running

        # Baseline update
        R_i = (R_t * gen_pad_mask).sum() / gen_pad_mask.sum().clamp(
            min=1
        )  # Prevent division by zero
        self.b_i = (
            self.config.REINFORCE_LAMBDA * self.b_i
            + (1.0 - self.config.REINFORCE_LAMBDA) * R_i.item()
        )

        # Return REINFORCE loss
        return -(
            (R_t - self.b_i) * log_probs * gen_pad_mask
        ).sum() / gen_pad_mask.sum().clamp(
            min=1
        )  # Prevent division by zero

    def train(self, start_epoch: int = 0) -> None:
        """
        Run the training loop.

        Args:
            start_epoch: Epoch index to start from.

        Returns:
            None
        """
        # Ensure checkpoint directory exists
        Path(self.config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

        # Initialize logging
        self._init_logging()
        self.logger.info(
            "Training started — device: %s  bf16: %s",
            self.config.DEVICE,
            self.config.USE_BF16,
        )

        # Loop over epochs
        for epoch in range(start_epoch, self.config.EPOCHS):
            # Set models to training mode
            self.model.train()
            self.discriminator.train()

            # Get flags for whether to train discriminator and use adversarial loss
            dis_start = max(1, int(self.config.DIS_START_EPOCH))
            adv_start = max(int(self.config.ADV_START_EPOCH), dis_start)
            train_dis = (epoch + 1) >= dis_start
            use_adv = (epoch + 1) >= adv_start

            # Get KL and adversarial weights for this epoch
            kl_w = self._get_kl_weight(epoch)
            adv_w = self._get_adversarial_weight(epoch)

            # Log epoch start
            self.logger.info(
                "── Epoch %d/%d  kl_w=%.4f  adv_w=%.4f  dis=%s  adv=%s",
                epoch + 1,
                self.config.EPOCHS,
                kl_w,
                adv_w,
                train_dis,
                use_adv,
            )

            # Initialize epoch stats for logging
            epoch_stats = {k: 0.0 for k in METRIC_KEYS}

            # Initialize window stats for batch logging
            window_stats = {k: 0.0 for k in METRIC_KEYS}

            # Create progress bar
            pbar = tqdm(
                self.dataloader, desc=f"Epoch {epoch + 1}", unit="batch", leave=False
            )

            # Loop over batches
            for batch_idx, batch in enumerate(pbar):
                # Extract batch data
                struct_tokens = batch[0].to(self.config.DEVICE)
                pass_tokens = batch[1].to(self.config.DEVICE)
                struct_lengths = batch[2].to(self.config.DEVICE)
                aux_unique_target = batch[3].to(self.config.DEVICE)
                bow_target = batch[4].to(self.config.DEVICE)

                # Prepare target sequence and padding mask
                target_seq = pass_tokens[:, 1:]
                B, T = target_seq.shape
                pad_mask = (target_seq != self.config.PAD_IDX).float()

                # CVAE forward
                with autocast(
                    device_type=self.config.DEVICE,
                    dtype=torch.bfloat16,
                    enabled=self.config.USE_BF16,
                ):
                    # Forward pass through the CVAE
                    (
                        logits,
                        aux_out,
                        mu_post,
                        log_var_post,
                        mu_prior,
                        log_var_prior,
                        z_prior,
                        h_c,
                    ) = self.model(pass_tokens, struct_tokens, struct_lengths)

                    # Compute losses
                    losses = self._compute_losses(
                        logits,
                        target_seq,
                        aux_out,
                        aux_unique_target,
                        mu_post,
                        log_var_post,
                        mu_prior,
                        log_var_prior,
                        kl_w,
                        bow_target,
                    )

                # Discriminator step
                d_loss_val = d_real_val = d_fake_val = 0.0
                g_adv_loss = None

                if train_dis:
                    # Generate samples for discriminator
                    gen_tokens, log_probs = self.model.decoder.decode_from_z(
                        z_prior,
                        h_c,
                        max_len=T,
                        return_log_probs=True,
                    )

                    # Create padding mask
                    gen_pad_mask = (gen_tokens != self.config.PAD_IDX).float()

                    # Update discriminator
                    d_loss_val, d_real_val, d_fake_val = self._step_discriminator(
                        target_seq, gen_tokens, pad_mask, gen_pad_mask
                    )

                    # Compute REINFORCE loss for generator if adversarial training is active
                    if use_adv:
                        g_adv_loss = self._compute_reinforce_loss(
                            gen_tokens, log_probs, gen_pad_mask, B, T
                        )

                # Total CVAE loss
                total_gen_loss = losses["total"]

                # Add adversarial loss
                if use_adv and g_adv_loss is not None:
                    total_gen_loss = losses["total"] + adv_w * g_adv_loss

                # Clear gradients from previous step
                self.optimizer_model.zero_grad()
                # Compute gradients
                total_gen_loss.backward()
                # Gradient clipping to prevent exploding gradients
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # Update CVAE parameters
                self.optimizer_model.step()

                g_adv_val = g_adv_loss.item() if g_adv_loss is not None else 0.0

                # Update stats
                batch_vals = {
                    "total": total_gen_loss.item(),
                    "ce": losses["ce"].item(),
                    "kl": losses["kl"].item(),
                    "aux": losses["aux"].item(),
                    "d_loss": d_loss_val,
                    "g_adv": g_adv_val,
                    "d_real": d_real_val,
                    "d_fake": d_fake_val,
                }

                # Accumulate stats
                for k, v in batch_vals.items():
                    epoch_stats[k] += v
                    window_stats[k] += v

                # Update progress bar
                pbar.set_postfix(
                    CE=f"{batch_vals['ce']:.2f}",
                    KL=f"{batch_vals['kl']:.2f}",
                    D=f"{d_loss_val:.2f}",
                    G=f"{g_adv_val:.2f}",
                )

                # Periodic batch logging
                if (batch_idx + 1) % self.config.LOG_EVERY_BATCHES == 0:
                    self._log_batch(
                        epoch + 1,
                        batch_idx + 1,
                        window_stats,
                        self.config.LOG_EVERY_BATCHES,
                    )

                    # Reset window stats
                    window_stats = {k: 0.0 for k in METRIC_KEYS}

                # Clear CUDA cache every 200 batches
                if batch_idx % 200 == 0 and batch_idx > 0:
                    torch.cuda.empty_cache()

            # Compute epoch averages
            n = len(self.dataloader)  # Total number of batches in the epoch
            avg = {k: epoch_stats[k] / n for k in METRIC_KEYS}
            # Log epoch metrics
            self._log_epoch(epoch + 1, avg)

            # Save checkpoint
            ckpt = (
                Path(self.config.checkpoint_dir) / f"trained_model_epoch_{epoch + 1}.pt"
            )
            torch.save(self.model.state_dict(), ckpt)
            self.logger.info("Checkpoint saved: %s", ckpt)

            # Clear CUDA cache after each epoch
            torch.cuda.empty_cache()

        self.logger.info("Training complete.")
