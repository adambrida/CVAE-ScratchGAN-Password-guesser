"""
===============================================================================
Script Name: model.py
Description: This module defines the CVAE-ScratchGAN architecture.
Author:      Adam Brida
Email:       248201@vutbr.cz
Date:        2026
License:     MIT
===============================================================================
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from config import Config


class StructEncoder(nn.Module):
    """
    Encodes the password structure.
    """

    def __init__(self, config: Config) -> None:
        """
        Constructor for the StructEncoder class.

        Args:
            config: Configuration object.

        Returns:
            None
        """
        super().__init__()

        # Embedding for structure tokens
        self.emb = nn.Embedding(
            config.struct_vocab_size, config.BIGRU_EMB_DIM, padding_idx=config.PAD_IDX
        )

        # BiGRU
        self.gru = nn.GRU(
            input_size=config.BIGRU_EMB_DIM,
            hidden_size=config.BIGRU_HIDDEN_DIM,
            num_layers=config.BIGRU_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=config.BIGRU_DROPOUT,
        )

        # Projection layer: 2 * BIGRU_HIDDEN_DIM to BIGRU_EMB_DIM
        self.out_proj = nn.Linear(2 * config.BIGRU_HIDDEN_DIM, config.BIGRU_EMB_DIM)

        # Layer normalization
        self.norm = nn.LayerNorm(config.BIGRU_EMB_DIM)

    def forward(
        self, struct_tokens: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the StructEncoder.

        Args:
            struct_tokens: Tensor with structure token IDs. Shape [B, T].
            lengths: Tensor with actual lengths of structures. Shape [B].

        Returns:
            h_c: Tensor representing the encoded structure. Shape [B, BIGRU_EMB_DIM].
        """
        # Embed structure tokens
        embedding = self.emb(struct_tokens)  # [B, T, BIGRU_EMB_DIM]

        # Pack the sequence for BiGRU
        packed = pack_padded_sequence(
            embedding, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Pass through BiGRU
        _, h_n = self.gru(packed)  # [2*BIGRU_LAYERS, B, BIGRU_HIDDEN_DIM]

        # Concatenate left and right directional outputs
        # h_n[-2] - second layer (left to right)
        # h_n[-1] - second layer (right to left)
        h_cat = torch.cat((h_n[-2], h_n[-1]), dim=1)  # [B, 2*BIGRU_HIDDEN_DIM]

        # Back to BIGRU_EMB_DIM through projection layer
        h_c = self.out_proj(h_cat)  # [B, BIGRU_EMB_DIM]

        # Normalize
        h_c = self.norm(h_c)  # [B, BIGRU_EMB_DIM]
        return h_c


class PriorHead(nn.Module):
    """
    Estimates the prior distribution parameters (mu and log_var).
    """

    def __init__(self, config: Config) -> None:
        """
        Constructor for the PriorHead class.

        Args:
            config: Configuration object.

        Returns:
            None
        """
        super().__init__()

        # Simple MLP to get mu and log_var from h_c
        self.prior_head = nn.Sequential(
            nn.Linear(config.BIGRU_EMB_DIM, config.BIGRU_HIDDEN_DIM),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(config.BIGRU_HIDDEN_DIM, config.BIGRU_HIDDEN_DIM),
            nn.GELU(),
            nn.Linear(config.BIGRU_HIDDEN_DIM, 2 * config.LATENT_DIM),
        )

    def forward(self, h_c: torch.Tensor) -> tuple:
        """
        Forward pass for the PriorHead.

        Args:
            h_c: Tensor with encoded structure from the StructEncoder. Shape [B, BIGRU_EMB_DIM].

        Returns:
            Tuple of (mu, log_var) where:
            mu: Tensor representing the mean of the prior distribution. Shape [B, LATENT_DIM].
            log_var: Tensor representing the log variance of the prior distribution. Shape [B, LATENT_DIM].
        """
        # Pass h_c through MLP
        out = self.prior_head(h_c)  # [B, 2*LATENT_DIM]

        # Splits tensor into 2 chunks (mu and log_var)
        mu, log_var = out.chunk(
            2, dim=-1
        )  # mu: [B, LATENT_DIM], log_var: [B, LATENT_DIM]

        # Clamp log_var for more stable training
        log_var = log_var.clamp(min=-10, max=10)
        return mu, log_var


class PasswordEncoder(nn.Module):
    """
    Encodes the password and structure.
    """

    def __init__(self, config: Config) -> None:
        """
        Constructor for the PasswordEncoder class.

        Args:
            config: Configuration object.

        Returns:
            None
        """
        super().__init__()
        self.pad_idx = int(config.PAD_IDX)

        # Embeddings for password tokens and structure tokens
        self.emb_pass = nn.Embedding(
            config.character_vocab_size,
            config.TRANSFORMER_EMB_DIM,
            padding_idx=self.pad_idx,
        )
        self.emb_struct = nn.Embedding(
            config.struct_vocab_size,
            config.TRANSFORMER_EMB_DIM,
            padding_idx=self.pad_idx,
        )

        # Positional embedding (+2 because BOS and EOS tokens)
        self.pos = nn.Embedding(config.MAX_PASS_LEN + 2, config.TRANSFORMER_EMB_DIM)

        # Transformer encoder layer
        enc_layer = nn.TransformerEncoderLayer(
            d_model=config.TRANSFORMER_EMB_DIM,
            nhead=config.TRANSFORMER_HEADS,
            dim_feedforward=config.TRANSFORMER_FF_DIM,
            dropout=config.TRANSFORMER_DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # Stack of Transformer encoder layers
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=config.TRANSFORMER_LAYERS, enable_nested_tensor=False
        )

        # Normalize before attention pooling
        self.norm = nn.LayerNorm(config.TRANSFORMER_EMB_DIM)

        # Attention pooling MLP
        self.attn_pool = nn.Sequential(
            nn.Linear(config.TRANSFORMER_EMB_DIM, 150, bias=False),
            nn.Tanh(),
            nn.Linear(150, 1, bias=False),
        )

    def forward(
        self, pass_tokens: torch.Tensor, struct_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the PasswordEncoder.

        Args:
            pass_tokens: Tensor with password token IDs. Shape [B, T].
            struct_tokens: Tensor with structure token IDs. Shape [B, T].

        Returns:
            h_x: Tensor representing the encoded password. Shape [B, TRANSFORMER_EMB_DIM].
        """
        # Get sequence length
        T = pass_tokens.size(1)

        # Create positional IDs for the sequence
        pos_ids = torch.arange(T, device=pass_tokens.device).unsqueeze(0)  # [1, T]

        # Embed password tokens, structure tokens and positional tokens and sum them
        embedding = (
            self.emb_pass(pass_tokens)
            + self.emb_struct(struct_tokens)
            + self.pos(pos_ids)
        )  # [B, T, TRANSFORMER_EMB_DIM]

        # Padding mask
        pad_mask = pass_tokens == self.pad_idx  # [B, T]

        # Pass through Transformer encoder
        enc_out = self.encoder(
            embedding, src_key_padding_mask=pad_mask
        )  # [B, T, TRANSFORMER_EMB_DIM]

        # Normalize
        enc_out = self.norm(enc_out)  # [B, T, TRANSFORMER_EMB_DIM]

        # Attention pooling
        attn_scores = self.attn_pool(enc_out)  # [B, T, 1]

        # Mask pads
        attn_scores = attn_scores.masked_fill(pad_mask.unsqueeze(-1), -1e4)  # [B, T, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, T, 1]

        # Weighted sum of encoder outputs
        h_x = torch.sum(enc_out * attn_weights, dim=1)  # [B, TRANSFORMER_EMB_DIM]

        return h_x


class PosteriorHead(nn.Module):
    """
    Estimates the posterior distribution parameters (mu and log_var).
    """

    def __init__(self, config: Config) -> None:
        """
        Constructor for the PosteriorHead class.

        Args:
            config: Configuration object.

        Returns:
            None
        """
        super().__init__()

        # Simple MLP to get mu and log_var from h_x
        self.posterior_head = nn.Sequential(
            nn.Linear(config.TRANSFORMER_EMB_DIM, config.TRANSFORMER_FF_DIM),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(config.TRANSFORMER_FF_DIM, config.TRANSFORMER_FF_DIM),
            nn.GELU(),
            nn.Linear(config.TRANSFORMER_FF_DIM, 2 * config.LATENT_DIM),
        )

    def forward(self, h_x: torch.Tensor) -> tuple:
        """
        Forward pass for the PosteriorHead.

        Args:
            h_x: Tensor with encoded password from the PasswordEncoder. Shape [B, TRANSFORMER_EMB_DIM].

        Returns:
            Tuple of (mu, log_var) where:
            mu: Tensor representing the mean of the posterior distribution. Shape [B, LATENT_DIM].
            log_var: Tensor representing the log variance of the posterior distribution. Shape [B, LATENT_DIM].
        """

        # Pass h_x through MLP to get mu and log_var
        out = self.posterior_head(h_x)  # [B, 2*LATENT_DIM]

        # Splits tensor into 2 chunks (mu and log_var)
        mu, log_var = out.chunk(
            2, dim=-1
        )  # mu: [B, LATENT_DIM], log_var: [B, LATENT_DIM]

        # Clamp log_var for more stable training
        log_var = log_var.clamp(-10, 10)
        return mu, log_var


class Decoder(nn.Module):
    """
    Decodes the latent representation back into a password.
    """

    def __init__(self, config: Config) -> None:
        """
        Constructor for the Decoder class.

        Args:
            config: Configuration object.

        Returns:
            None
        """
        super().__init__()
        self.config = config
        self.pad_idx = int(config.PAD_IDX)
        self.bos_idx = int(config.BOS_IDX)
        self.eos_idx = int(config.EOS_IDX)
        self.unk_idx = int(config.UNK_IDX)

        self.word_dropout_rate = config.WORD_DROPOUT_RATE

        self.decoder_layers = config.DECODER_LAYERS
        self.decoder_hidden = config.DECODER_HIDDEN_DIM

        # Embedding for password tokens
        self.emb = nn.Embedding(
            config.character_vocab_size,
            config.DECODER_EMB_DIM,
            padding_idx=self.pad_idx,
        )

        # GRU decoder
        self.gru = nn.GRU(
            input_size=config.DECODER_EMB_DIM
            + config.LATENT_DIM
            + config.BIGRU_EMB_DIM,
            hidden_size=config.DECODER_HIDDEN_DIM,
            num_layers=config.DECODER_LAYERS,
            dropout=config.DECODER_DROPOUT,
            batch_first=True,
        )

        # MLP to compute initial hidden state of GRU from z and h_c
        self.init_h = nn.Sequential(
            nn.Linear(
                config.LATENT_DIM + config.BIGRU_EMB_DIM, config.DECODER_HIDDEN_DIM
            ),
            nn.GELU(),
            nn.Linear(
                config.DECODER_HIDDEN_DIM,
                config.DECODER_HIDDEN_DIM * config.DECODER_LAYERS,
            ),
        )

        # Projection from GRU hidden dimension to embedding dimension
        self.hidden_proj = nn.Linear(config.DECODER_HIDDEN_DIM, config.DECODER_EMB_DIM)

        # Output projection to vocabulary size, shared with embedding weights
        self.fc_out = nn.Linear(
            config.DECODER_EMB_DIM, config.character_vocab_size, bias=False
        )
        self.fc_out.weight = self.emb.weight

    def forward(
        self,
        x_tokens: torch.Tensor,
        z: torch.Tensor,
        h_c: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the Decoder.

        Args:
            x_tokens: Tensor with input token IDs for the decoder (Teacher Forcing). Shape [B, T].
            z: Tensor with sampled latent vector. Shape [B, LATENT_DIM].
            h_c: Tensor with encoded structure from the StructEncoder. Shape [B, BIGRU_EMB_DIM].

        Returns:
            logits: Tensor with the output logits. Shape [B, T, character_vocab_size].
        """
        # Get batch size and sequence length
        B, T = x_tokens.shape

        # Word dropout
        if self.training and self.word_dropout_rate > 0:
            prob = torch.rand(x_tokens.shape, device=x_tokens.device)
            mask = prob < self.word_dropout_rate
            dec_input = x_tokens.clone()
            dec_input[mask] = self.unk_idx
            embedded = self.emb(dec_input)  # [B, T, DECODER_EMB_DIM]
        else:
            embedded = self.emb(x_tokens)  # [B, T, DECODER_EMB_DIM]

        # Initialize GRU hidden state from z and h_c
        h_0 = self.init_h(
            torch.cat([z, h_c], dim=1)
        )  # [B, DECODER_HIDDEN_DIM * DECODER_LAYERS]
        h_0 = h_0.view(
            self.decoder_layers, B, self.decoder_hidden
        ).contiguous()  # [DECODER_LAYERS, B, DECODER_HIDDEN_DIM]

        # Expand z and h_c
        z_expanded = z[:, None, :].expand(-1, T, -1)  # [B, T, LATENT_DIM]
        h_c_expanded = h_c[:, None, :].expand(-1, T, -1)  # [B, T, BIGRU_EMB_DIM]

        # Concatenate embedded input, z and h_c for GRU input
        gru_input = torch.cat(
            [embedded, z_expanded, h_c_expanded], dim=2
        )  # [B, T, DECODER_EMB_DIM + LATENT_DIM + BIGRU_EMB_DIM]

        # Pass through GRU
        out, _ = self.gru(gru_input, h_0)  # [B, T, DECODER_HIDDEN_DIM]

        # Project GRU output to embedding dimension
        out = self.hidden_proj(out)  # [B, T, DECODER_EMB_DIM]

        # Project to vocabulary size
        logits = self.fc_out(out)  # [B, T, character_vocab_size]
        return logits

    def decode_from_z(
        self,
        z: torch.Tensor,
        h_c: torch.Tensor,
        max_len: int,
        temp: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        return_log_probs: bool = False,
    ) -> tuple:
        """
        Autoregressive decoding from a latent vector z and encoded structure h_c.

        Args:
            z: Prior latent vector. Shape [B, LATENT_DIM].
            h_c: Encoded structure from StructEncoder. Shape [B, BIGRU_EMB_DIM].
            max_len: Maximum length of the generated password.
            temp: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Top-p sampling parameter.
            return_log_probs: If True, also return log-probabilities.

        Returns:
            Tuple of (tokens, log_probs) where:
            tokens: Generated token IDs. Shape [B, steps].
            log_probs: Log-probabilities of sampled tokens. Shape [B, steps].
        """
        # Get batch size
        B = z.size(0)

        # Prepare initial hidden state for GRU
        with torch.amp.autocast(
            device_type=self.config.DEVICE,
            dtype=torch.bfloat16,
            enabled=self.config.USE_BF16,
        ):
            h_0 = self.init_h(
                torch.cat([z, h_c], dim=1)
            )  # [B, DECODER_HIDDEN_DIM * DECODER_LAYERS]
            h_0 = h_0.view(
                self.decoder_layers, B, self.decoder_hidden
            ).contiguous()  # [DECODER_LAYERS, B, DECODER_HIDDEN_DIM]
            z_step = z.unsqueeze(1)  # [B, 1, LATENT_DIM]
            h_c_step = h_c.unsqueeze(1)  # [B, 1, BIGRU_EMB_DIM]

        # Start with BOS token
        curr = torch.full(
            (B, 1), self.bos_idx, dtype=torch.long, device=z.device
        )  # [B, 1]

        # Finished flags
        finished = torch.zeros(B, dtype=torch.bool, device=z.device)  # [B]

        # Outputs
        tokens_out = []
        log_probs_out = []

        # Loop for autoregressive decoding
        for _ in range(max_len):

            # Compute logits for the current step
            with torch.amp.autocast(
                device_type=self.config.DEVICE,
                dtype=torch.bfloat16,
                enabled=self.config.USE_BF16,
            ):
                emb = self.emb(curr)  # [B, 1, DECODER_EMB_DIM]
                gru_input = torch.cat(
                    [emb, z_step, h_c_step], dim=2
                )  # [B, 1, DECODER_EMB_DIM + LATENT_DIM + BIGRU_EMB_DIM]
                out, h_0 = self.gru(gru_input, h_0)  # [B, 1, DECODER_HIDDEN_DIM]
                # Take the last output of the GRU
                out = self.hidden_proj(out[:, -1])  # [B, DECODER_EMB_DIM]
                logits = self.fc_out(out)  # [B, character_vocab_size]

            # Apply temperature
            logits = logits / temp

            # Top-k
            if top_k > 0:
                # Check if top_k is less than vocab size
                k = min(top_k, logits.size(-1))
                # Get the threshold logit value for top-k
                threshold = logits.topk(k, dim=-1).values[:, -1:]  # [B, 1]
                # Mask out logits below the threshold
                logits = logits.masked_fill(logits < threshold, float("-inf"))

            # Top-p
            if top_p < 1.0:
                # Sort logits and calculate probabilities
                probs_sorted, sorted_idx = torch.sort(
                    F.softmax(logits, dim=-1), descending=True
                )
                # Create a mask for tokens to remove
                remove = (torch.cumsum(probs_sorted, dim=-1) - probs_sorted) > top_p
                # Mask out tokens that are above the threshold
                logits = logits.masked_fill(
                    remove.scatter(1, sorted_idx, remove), float("-inf")
                )

            # Get probabilities and sample the next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            # If a sequence is finished, keep generating PAD tokens
            done = finished.unsqueeze(-1)  # [B, 1]
            # If done, set next_token to PAD_IDX, otherwise keep the sampled token
            next_token = torch.where(done, self.pad_idx, next_token)

            if return_log_probs:
                log_prob = torch.log(
                    probs.gather(1, next_token) + 1e-8
                )  # Prevent log(0)
                # If done, log_prob of PAD tokens should be 0
                log_prob = torch.where(done, torch.zeros_like(log_prob), log_prob)
                log_probs_out.append(log_prob)

            # Append the sampled token to the output
            tokens_out.append(next_token)
            # Update current token for the next step
            curr = next_token

            # Set finished flags for sequences that generated EOS
            finished = finished | (next_token.squeeze(-1) == self.eos_idx)  # [B]

            # Early exit when all sequences have finished
            if not return_log_probs and finished.all():
                break

        # Concatenate outputs
        tokens = torch.cat(tokens_out, dim=1)

        if return_log_probs:
            return tokens, torch.cat(log_probs_out, dim=1)

        return tokens


class AuxiliaryHeads(nn.Module):
    """
    Auxiliary heads to predict password properties from the latent vector z.
    """

    def __init__(self, config: Config) -> None:
        """
        Constructor for the AuxiliaryHeads class.

        Args:
            config: Configuration object.

        Returns:
            None
        """
        super().__init__()

        # MLP for unique chars
        self.head_unique = nn.Sequential(
            nn.Linear(config.LATENT_DIM, config.LATENT_DIM),
            nn.GELU(),
            nn.Linear(config.LATENT_DIM, 1),
        )

        # MLP for bag-of-words
        self.head_bow = nn.Sequential(
            nn.Linear(config.LATENT_DIM, config.LATENT_DIM),
            nn.GELU(),
            nn.Linear(config.LATENT_DIM, config.character_vocab_size),
        )

    def forward(self, z: torch.Tensor) -> dict:
        """
        Forward pass for the AuxiliaryHeads.

        Args:
            z: Tensor representing the sampled latent vector. Shape [B, LATENT_DIM].

        Returns:
            Dictionary containing outputs of both auxiliary heads.
        """
        return {
            "unique": self.head_unique(z),  # [B, 1]
            "bow": self.head_bow(z),  # [B, character_vocab_size]
        }


class Discriminator(nn.Module):
    """
    Discriminator that scores passwords.
    """

    def __init__(self, config: Config) -> None:
        """
        Constructor for the Discriminator class.

        Args:
            config: Configuration object.

        Returns:
            None
        """
        super().__init__()
        self.device = config.DEVICE
        self.max_len = config.MAX_PASS_LEN

        # Token embedding with dropout
        self.emb = nn.Embedding(
            config.character_vocab_size,
            config.DISCRIMINATOR_EMB_DIM,
            padding_idx=config.PAD_IDX,
        )
        self.emb_drop = nn.Dropout(config.DISCRIMINATOR_DROPOUT)

        # GRU discriminator
        self.gru = nn.GRU(
            input_size=config.DISCRIMINATOR_EMB_DIM + 8,  # 8 for positional signal
            hidden_size=config.DISCRIMINATOR_HIDDEN_DIM,
            num_layers=config.DISCRIMINATOR_LAYERS,
            batch_first=True,
        )

        # Normalize gru output
        self.norm = nn.LayerNorm(config.DISCRIMINATOR_HIDDEN_DIM)

        # Get score from GRU output
        self.fc_out = nn.Linear(config.DISCRIMINATOR_HIDDEN_DIM, 1)

    def _get_positional_signal(self, seq_len: int) -> torch.Tensor:
        """
        Compute the positional signal (ScratchGAN paper).

        Args:
            seq_len: Length of the sequence to generate the signal for.

        Returns:
            pos_signal: Positional signal. Shape [seq_len, 8].
        """
        # Positions (+1 because positions start at 1 in the paper)
        pos = torch.arange(
            1, seq_len + 1, device=self.device, dtype=torch.float32
        ).unsqueeze(
            1
        )  # [seq_len, 1]

        # Log-linearly spaced periods: T_1 = 2, T_8 = 4 * max_len
        T_1 = 2.0
        T_8 = 4.0 * float(self.max_len)
        periods = torch.logspace(
            math.log10(T_1), math.log10(T_8), 8, device=self.device
        )  # [8]

        # Compute the positional signal using sin
        pos_signal = torch.sin(2 * math.pi * pos / periods)  # [seq_len, 8]
        return pos_signal

    def forward(self, x_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Discriminator.

        Args:
            x_tokens: Tensor containing discrete token IDs. Shape [B, T].

        Returns:
            logits: Tensor containing scores for each character. Shape [B, T].
        """
        # Get batch size and sequence length
        B, T = x_tokens.shape

        # Embed input tokens with dropout
        embedded = self.emb_drop(self.emb(x_tokens))  # [B, T, EMB_DIM]

        # Add positional signal
        pos_signal = self._get_positional_signal(T).to(embedded.dtype)  # [T, 8]
        pos_signal = pos_signal.unsqueeze(0).expand(B, -1, -1)  # [B, T, 8]
        gru_input = torch.cat([embedded, pos_signal], dim=-1)  # [B, T, EMB_DIM + 8]

        # Pass through GRU
        out, _ = self.gru(gru_input)  # [B, T, HIDDEN_DIM]

        # Normalize and compute logits
        out = self.norm(out)  # [B, T, HIDDEN_DIM]
        logits = self.fc_out(out).squeeze(-1)  # [B, T]
        return logits


class CVAE_Model(nn.Module):
    """
    Conditional Variational Autoencoder model.
    """

    def __init__(self, config: Config) -> None:
        """
        Constructor for the CVAE_Model class.

        Args:
            config: Configuration object.

        Returns:
            None
        """
        super().__init__()
        self.config = config

        self.pad_idx = config.PAD_IDX

        # Prior encoder
        self.prior_encoder = StructEncoder(config)
        self.prior_head = PriorHead(config)

        # Posterior encoder
        self.posterior_encoder = PasswordEncoder(config)
        self.posterior_head = PosteriorHead(config)

        # Decoder
        self.decoder = Decoder(config)

        # Auxiliary heads
        self.aux_heads = AuxiliaryHeads(config)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick.

        Args:
            mu: Tensor with the mean of the distribution. Shape [B, LATENT_DIM].
            log_var: Tensor with the log variance of the distribution. Shape [B, LATENT_DIM].

        Returns:
            z: Tensor with the sampled latent vector. Shape [B, LATENT_DIM].
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        pass_tokens: torch.Tensor,
        struct_tokens: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple:
        """
        Forward pass for the CVAE model.

        Args:
            pass_tokens: Tensor with password token IDs (Teacher Forcing). Shape [B, T].
            struct_tokens: Tensor with structure token IDs. Shape [B, T].
            lengths: Tensor with the actual lengths of each sequence. Shape [B].

        Returns:
            Tuple of (logits, aux_outputs, mu_post, log_var_post, mu_prior, log_var_prior, z_prior, h_c) where:
            logits: Tensor containing the output logits. Shape [B, T, VOCAB_SIZE].
            aux_outputs: Dictionary containing the outputs of each auxiliary head.
            mu_post: Tensor representing the mean of the posterior distribution. Shape [B, LATENT_DIM].
            log_var_post: Tensor representing the log variance of the posterior distribution. Shape [B, LATENT_DIM].
            mu_prior: Tensor representing the mean of the prior distribution. Shape [B, LATENT_DIM].
            log_var_prior: Tensor representing the log variance of the prior distribution. Shape [B, LATENT_DIM].
            z_prior: Sampled latent vector from the prior distribution. Shape [B, LATENT_DIM].
            h_c: Encoded structure representation from the StructEncoder. Shape [B, BIGRU_EMB_DIM].
        """

        # Prior
        h_c = self.prior_encoder(struct_tokens, lengths)
        mu_prior, log_var_prior = self.prior_head(h_c)

        # Posterior
        h_x = self.posterior_encoder(pass_tokens, struct_tokens)
        mu_post, log_var_post = self.posterior_head(h_x)

        # Sample Z
        z_post = self.reparameterize(mu_post, log_var_post)
        z_prior = self.reparameterize(mu_prior, log_var_prior)

        # Decode
        dec_input = pass_tokens[:, :-1]
        logits = self.decoder(dec_input, z_post, h_c)

        # Aux Heads
        aux_outputs = self.aux_heads(z_post)

        return (
            logits,
            aux_outputs,
            mu_post,
            log_var_post,
            mu_prior,
            log_var_prior,
            z_prior,
            h_c,
        )
