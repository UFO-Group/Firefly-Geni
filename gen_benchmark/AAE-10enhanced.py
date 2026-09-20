"""Original molecular AAE adapted to the Firefly-Geni token dataset.

The AAE framework is retained from the supplied implementation:

* shared character embedding;
* one-layer (optionally bidirectional) GRU encoder;
* linear encoder projection into an unconstrained latent vector;
* three-layer GRU decoder receiving [token embedding, repeated z];
* decoder initial state obtained by a linear projection of z;
* latent discriminator: Linear-ReLU-Linear-ReLU-Linear-Sigmoid;
* generator update: weighted reconstruction + adversarial BCE;
* discriminator update: N(0, I) samples are real, encoded samples are fake;
* ancestral generation from N(0, I) with full-softmax multinomial sampling.

The AAE architecture remains unchanged.  Shared training and sampling
hyperparameters are aligned with CLLaMA where a direct counterpart exists.
The supplied AAE is unconditional, so Ptrain.npy and Ptest.npy are intentionally
not passed into the model.
"""

import argparse
import csv
import math
import os
import random
import time
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader

from module.char import charset_list
from module.dataload import UserDataset
from module.eval_def import (
    calculate_diversity,
    novelty,
    uniqueness,
    valid_molecules,
)
from module.other_function import vec_to_char


class Encoder(nn.Module):
    """Original AAE GRU encoder."""

    def __init__(
        self,
        embedding_layer,
        encoder_hidden_dim,
        latent_dim,
        bidirectional,
    ):
        super().__init__()
        self.embedding_layer = embedding_layer
        self.gru_encoder = nn.GRU(
            embedding_layer.embedding_dim,
            encoder_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        last_hidden_dim = encoder_hidden_dim * (2 if bidirectional else 1)
        self.hidden_to_latent = nn.Linear(last_hidden_dim, latent_dim)

    def forward(self, sequences):
        lengths = torch.tensor(
            [sequence.numel() for sequence in sequences],
            dtype=torch.long,
        )
        padded = pad_sequence(sequences, batch_first=True, padding_value=0)
        embedded = self.embedding_layer(padded)

        # Packing adapts the original variable-length list interface to the
        # Firefly fixed-width arrays without changing the GRU architecture.
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru_encoder(packed)

        hidden = hidden[-(1 + int(self.gru_encoder.bidirectional)) :]
        hidden = torch.cat(hidden.split(1), dim=-1).squeeze(0)
        return self.hidden_to_latent(hidden)


class Decoder(nn.Module):
    """Original three-layer latent-conditioned GRU decoder."""

    def __init__(
        self,
        vocab_size,
        embedding_layer,
        decoder_hidden_dim,
        latent_dim,
        end_idx,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_layer = embedding_layer
        self.end_idx = end_idx

        self.latent_to_hidden = nn.Linear(latent_dim, decoder_hidden_dim)
        self.gru_decoder = nn.GRU(
            embedding_layer.embedding_dim + latent_dim,
            decoder_hidden_dim,
            num_layers=3,
            batch_first=True,
            bidirectional=False,
        )
        self.decipher = nn.Linear(decoder_hidden_dim, vocab_size)

    def forward(self, sequences, z):
        lengths = torch.tensor(
            [sequence.numel() for sequence in sequences],
            dtype=torch.long,
            device=z.device,
        )
        padded = pad_sequence(
            sequences,
            batch_first=True,
            padding_value=self.end_idx,
        )
        embedded = self.embedding_layer(padded)

        repeated_z = z.unsqueeze(1).repeat(1, embedded.size(1), 1)
        decoder_input = torch.cat([embedded, repeated_z], dim=-1)

        initial_hidden = self.latent_to_hidden(z)
        initial_hidden = initial_hidden.unsqueeze(0).repeat(
            self.gru_decoder.num_layers, 1, 1
        )
        output, _ = self.gru_decoder(decoder_input, initial_hidden)
        logits = self.decipher(output)

        # Teacher forcing: token t predicts token t+1.  A length mask is used
        # because Firefly uses '>' for both EOS and post-EOS padding, whereas
        # the supplied AAE had a separate <pad> symbol.
        predictions = logits[:, :-1, :]
        targets = padded[:, 1:]
        token_loss = F.cross_entropy(
            predictions.reshape(-1, predictions.size(-1)),
            targets.reshape(-1),
            reduction="none",
        ).view(targets.size(0), targets.size(1))
        valid_steps = (lengths - 1).clamp(min=1, max=targets.size(1))
        positions = torch.arange(
            targets.size(1), device=targets.device
        ).unsqueeze(0)
        mask = positions < valid_steps.unsqueeze(1)
        reconstruction_loss = (
            token_loss * mask
        ).sum() / mask.sum().clamp(min=1)
        return reconstruction_loss


class Discriminator(nn.Module):
    """Original latent-space discriminator."""

    def __init__(self, latent_dim, discriminator_hidden_dim):
        super().__init__()
        self.discriminator_layer = nn.Sequential(
            nn.Linear(latent_dim, discriminator_hidden_dim),
            nn.ReLU(),
            nn.Linear(
                discriminator_hidden_dim,
                int(discriminator_hidden_dim / 2),
            ),
            nn.ReLU(),
            nn.Linear(int(discriminator_hidden_dim / 2), 1),
            nn.Sigmoid(),
        )

    def forward(self, latent):
        return self.discriminator_layer(latent)


class AAE(nn.Module):
    """Original AAE composition with the Firefly vocabulary indices."""

    def __init__(
        self,
        vocab_size,
        start_idx,
        end_idx,
        embedding_dim=64,
        encoder_hidden_dim=256,
        decoder_hidden_dim=512,
        latent_dim=128,
        bidirectional=True,
        discriminator_hidden_dim=1024,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.latent_size = latent_dim

        self.embedding_layer = nn.Embedding(vocab_size, embedding_dim)
        self.encoder = Encoder(
            self.embedding_layer,
            encoder_hidden_dim,
            latent_dim,
            bidirectional,
        )
        self.decoder = Decoder(
            vocab_size,
            self.embedding_layer,
            decoder_hidden_dim,
            latent_dim,
            end_idx,
        )
        self.discriminator = Discriminator(
            latent_dim,
            discriminator_hidden_dim,
        )

    def encoder_forward(self, sequences):
        return self.encoder(sequences)

    def decoder_forward(self, sequences, z):
        return self.decoder(sequences, z)

    def discriminator_forward(self, z):
        return self.discriminator(z)

    def random_sample_latent(self, number, device=None):
        if device is None:
            device = next(self.parameters()).device
        return torch.randn(number, self.latent_size, device=device)

    @torch.no_grad()
    def samples_generation(
        self,
        number,
        max_len=100,
        temperature=1.0,
        top_k=10,
    ):
        """AAE ancestral sampling with CLLaMA-aligned temperature/top-k."""

        self.eval()
        device = next(self.parameters()).device
        latent = self.random_sample_latent(number, device=device)
        latent_step = latent.unsqueeze(1)

        hidden = self.decoder.latent_to_hidden(latent)
        hidden = hidden.unsqueeze(0).repeat(
            self.decoder.gru_decoder.num_layers, 1, 1
        )
        current_token = torch.full(
            (number,),
            self.start_idx,
            dtype=torch.long,
            device=device,
        )
        generated = torch.full(
            (number, max_len),
            self.end_idx,
            dtype=torch.long,
            device=device,
        )
        generated[:, 0] = self.start_idx
        finished = torch.zeros(number, dtype=torch.bool, device=device)

        for position in range(1, max_len):
            embedded = self.embedding_layer(current_token).unsqueeze(1)
            decoder_input = torch.cat([embedded, latent_step], dim=-1)
            output, hidden = self.decoder.gru_decoder(decoder_input, hidden)
            logits = self.decoder.decipher(output.squeeze(1))
            temperature = max(float(temperature), 1e-8)
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                k = min(int(top_k), logits.size(-1))
                top_values, top_indices = torch.topk(logits, k=k, dim=-1)
                top_probabilities = F.softmax(top_values, dim=-1)
                sampled_position = torch.multinomial(
                    top_probabilities, num_samples=1
                )
                current_token = top_indices.gather(
                    dim=-1, index=sampled_position
                ).squeeze(1)
            else:
                probabilities = F.softmax(logits, dim=-1)
                current_token = torch.multinomial(
                    probabilities, num_samples=1
                ).squeeze(1)
            generated[~finished, position] = current_token[~finished]
            finished = finished | (current_token == self.end_idx)
            current_token = torch.where(
                finished,
                torch.full_like(current_token, self.end_idx),
                current_token,
            )
            if bool(finished.all()):
                break

        return generated


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rows_to_sequence_list(tokens, start_idx, end_idx):
    """Convert fixed-width Firefly arrays into original AAE sequence lists."""

    sequences = []
    for row in tokens:
        row = row.long()
        end_locations = torch.nonzero(row == end_idx, as_tuple=False)
        if end_locations.numel() > 0:
            end_position = int(end_locations[0].item())
            sequence = row[: end_position + 1]
        else:
            sequence = row

        if sequence.numel() == 0 or int(sequence[0].item()) != start_idx:
            sequence = torch.cat(
                [
                    torch.tensor(
                        [start_idx], dtype=torch.long, device=row.device
                    ),
                    sequence,
                ]
            )
        if int(sequence[-1].item()) != end_idx:
            sequence = torch.cat(
                [
                    sequence,
                    torch.tensor(
                        [end_idx], dtype=torch.long, device=row.device
                    ),
                ]
            )
        sequences.append(sequence)
    return sequences


def decode_sequences(sequences):
    output = []
    for sequence in sequences:
        smiles = vec_to_char(
            sequence.detach().cpu().numpy(), charset_list
        )
        output.append(smiles.replace("^", "").split(">")[0])
    return output


def load_reference_smiles(data_directory, train_dataset):
    train_csv = os.path.join(data_directory, "train.csv")
    try:
        frame = pd.read_csv(train_csv)
        if "SMILES" in frame.columns:
            column = "SMILES"
        elif "TADF_SMILES" in frame.columns:
            column = "TADF_SMILES"
        else:
            column = frame.columns[0]
        smiles = frame[column].dropna().astype(str).tolist()
        print(f"Loaded {len(smiles)} reference SMILES from {train_csv}.")
        return smiles
    except Exception as exc:
        print(f"Reference CSV unavailable ({exc}); decoding Strain.npy.")
        return decode_sequences(train_dataset.Xdata)


def make_optimizers(model, learning_rate):
    """Exact optimizer groups and betas from the supplied AAE trainer."""

    # The embedding module is shared by encoder and decoder.  The supplied
    # code consequently lists the same Parameter twice; modern PyTorch warns
    # that this will become an error.  Keep the same parameter set but include
    # each shared object exactly once.
    generator_parameters = []
    seen_parameter_ids = set()
    for parameter in chain(
        model.encoder.parameters(), model.decoder.parameters()
    ):
        if id(parameter) not in seen_parameter_ids:
            generator_parameters.append(parameter)
            seen_parameter_ids.add(id(parameter))

    generator_optimizer = optim.Adam(
        generator_parameters,
        lr=learning_rate,
        betas=(0.9, 0.99),
    )
    discriminator_optimizer = optim.Adam(
        model.discriminator.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.99),
    )
    return generator_optimizer, discriminator_optimizer


def train_epoch(
    model,
    loader,
    generator_optimizer,
    discriminator_optimizer,
    device,
    reconstruction_ratio,
    start_idx,
    end_idx,
    gradient_clip,
):
    model.train()
    totals = {
        "generator_loss": 0.0,
        "discriminator_loss": 0.0,
        "reconstruction_loss": 0.0,
        "adversarial_loss": 0.0,
        "real_loss": 0.0,
        "fake_loss": 0.0,
    }
    batches = 0

    for batch in loader:
        tokens = batch[0].to(device)
        sequences = rows_to_sequence_list(tokens, start_idx, end_idx)
        batch_size = len(sequences)
        real_targets = torch.ones(batch_size, 1, device=device)
        fake_targets = torch.zeros(batch_size, 1, device=device)

        # 1. Generator/autoencoder update, exactly as in the supplied trainer.
        latent = model.encoder_forward(sequences)
        reconstruction_loss = model.decoder_forward(sequences, latent)
        encoded_probability = model.discriminator_forward(latent)
        adversarial_loss = F.binary_cross_entropy(
            encoded_probability, real_targets
        )
        generator_loss = (
            reconstruction_ratio * reconstruction_loss
            + (1.0 - reconstruction_ratio) * adversarial_loss
        )

        generator_optimizer.zero_grad(set_to_none=True)
        generator_loss.backward()
        if gradient_clip is not None and gradient_clip > 0:
            nn.utils.clip_grad_norm_(
                generator_optimizer.param_groups[0]["params"],
                gradient_clip,
            )
        generator_optimizer.step()

        # 2. Latent discriminator update, exactly as in the supplied trainer.
        prior_latent = model.random_sample_latent(batch_size, device=device)
        real_probability = model.discriminator_forward(prior_latent)
        real_loss = F.binary_cross_entropy(real_probability, real_targets)
        fake_probability = model.discriminator_forward(latent.detach())
        fake_loss = F.binary_cross_entropy(fake_probability, fake_targets)
        discriminator_loss = 0.5 * (real_loss + fake_loss)

        discriminator_optimizer.zero_grad(set_to_none=True)
        discriminator_loss.backward()
        if gradient_clip is not None and gradient_clip > 0:
            nn.utils.clip_grad_norm_(
                discriminator_optimizer.param_groups[0]["params"],
                gradient_clip,
            )
        discriminator_optimizer.step()

        totals["generator_loss"] += generator_loss.item()
        totals["discriminator_loss"] += discriminator_loss.item()
        totals["reconstruction_loss"] += reconstruction_loss.item()
        totals["adversarial_loss"] += adversarial_loss.item()
        totals["real_loss"] += real_loss.item()
        totals["fake_loss"] += fake_loss.item()
        batches += 1

    if batches == 0:
        raise RuntimeError("The training DataLoader produced no batches.")
    return {name: value / batches for name, value in totals.items()}


@torch.no_grad()
def evaluate_epoch(
    model,
    loader,
    device,
    reconstruction_ratio,
    start_idx,
    end_idx,
):
    model.eval()
    totals = {
        "generator_loss": 0.0,
        "discriminator_loss": 0.0,
        "reconstruction_loss": 0.0,
        "adversarial_loss": 0.0,
        "real_loss": 0.0,
        "fake_loss": 0.0,
    }
    batches = 0

    for batch in loader:
        tokens = batch[0].to(device)
        sequences = rows_to_sequence_list(tokens, start_idx, end_idx)
        batch_size = len(sequences)
        real_targets = torch.ones(batch_size, 1, device=device)
        fake_targets = torch.zeros(batch_size, 1, device=device)

        latent = model.encoder_forward(sequences)
        reconstruction_loss = model.decoder_forward(sequences, latent)
        encoded_probability = model.discriminator_forward(latent)
        adversarial_loss = F.binary_cross_entropy(
            encoded_probability, real_targets
        )
        generator_loss = (
            reconstruction_ratio * reconstruction_loss
            + (1.0 - reconstruction_ratio) * adversarial_loss
        )

        prior_latent = model.random_sample_latent(batch_size, device=device)
        real_probability = model.discriminator_forward(prior_latent)
        real_loss = F.binary_cross_entropy(real_probability, real_targets)
        fake_probability = model.discriminator_forward(latent)
        fake_loss = F.binary_cross_entropy(fake_probability, fake_targets)
        discriminator_loss = 0.5 * (real_loss + fake_loss)

        totals["generator_loss"] += generator_loss.item()
        totals["discriminator_loss"] += discriminator_loss.item()
        totals["reconstruction_loss"] += reconstruction_loss.item()
        totals["adversarial_loss"] += adversarial_loss.item()
        totals["real_loss"] += real_loss.item()
        totals["fake_loss"] += fake_loss.item()
        batches += 1

    if batches == 0:
        raise RuntimeError("The test DataLoader produced no batches.")
    return {name: value / batches for name, value in totals.items()}


@torch.no_grad()
def evaluate_vund(
    model,
    reference_smiles,
    max_length,
    total_to_generate=10000,
    generation_batch_size=100,
    temperature=1.0,
    top_k=10,
):
    model.eval()
    generated_smiles = []
    completed = 0

    while completed < total_to_generate:
        current_batch = min(
            generation_batch_size, total_to_generate - completed
        )
        sequences = model.samples_generation(
            current_batch,
            max_len=max_length,
            temperature=temperature,
            top_k=top_k,
        )
        generated_smiles.extend(decode_sequences(sequences))
        completed += current_batch
        print(f"    Generated {completed}/{total_to_generate}")

    valid_count, valid_ratio = valid_molecules(generated_smiles)
    unique_ratio = uniqueness(generated_smiles)
    novelty_ratio = novelty(generated_smiles, reference_smiles)
    diversity_score = calculate_diversity(generated_smiles[:1000])
    return {
        "valid_count": valid_count,
        "valid_ratio": valid_ratio,
        "unique_ratio": unique_ratio,
        "novelty_ratio": novelty_ratio,
        "diversity_score": diversity_score,
        "generated_smiles": generated_smiles,
    }


def save_rows(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the original unconditional molecular AAE using "
            "CLLaMA-aligned shared hyperparameters on the 10x-augmented "
            "Firefly-Geni TADF token dataset."
        )
    )

    # CLLaMA-aligned values where a direct AAE counterpart exists.
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--encoder-hidden-dim", type=int, default=512)
    parser.add_argument("--decoder-hidden-dim", type=int, default=512)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--discriminator-hidden-dim", type=int, default=1024)
    bidirectional_group = parser.add_mutually_exclusive_group()
    bidirectional_group.add_argument(
        "--bidirectional",
        dest="bidirectional",
        action="store_true",
    )
    bidirectional_group.add_argument(
        "--no-bidirectional",
        dest="bidirectional",
        action="store_false",
    )
    parser.set_defaults(bidirectional=True)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--reconstruction-ratio", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient-clip", type=float, default=5.0)

    # Evaluation/output controls; these do not change the AAE framework.
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--gen-total", type=int, default=10000)
    parser.add_argument("--gen-batch-size", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-generation-length", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.reconstruction_ratio <= 1.0:
        raise ValueError("--reconstruction-ratio must be in [0, 1].")
    set_seed(args.seed)

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_directory = args.data_dir or os.path.abspath(
        os.path.join(
            script_directory,
            "..",
            "dataset_tadf",
            "dataset_gen",
            "gendata_est_sa",
            "enhanced10",
        )
    )
    save_directory = args.save_dir or os.path.join(
        script_directory,
        "AAE_10enhanced_cllamahy-shuffle",
        (
            f"emb{args.embedding_dim}_enc{args.encoder_hidden_dim}_"
            f"dec{args.decoder_hidden_dim}_latent{args.latent_dim}_"
            f"bs{args.batch_size}_lr{args.lr}"
        ),
    )
    os.makedirs(save_directory, exist_ok=True)
    print(f"Data directory: {data_directory}")
    print(f"Output directory: {save_directory}")

    train_dataset = UserDataset(data_directory, "train")
    test_dataset = UserDataset(data_directory, "test")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(
        "The supplied AAE is unconditional; property arrays are "
        "intentionally ignored."
    )

    char_to_index = {
        character: index for index, character in enumerate(charset_list)
    }
    if "^" not in char_to_index or ">" not in char_to_index:
        raise RuntimeError("charset_list must contain '^' and '>'.")
    start_idx = char_to_index["^"]
    end_idx = char_to_index[">"]
    vocab_size = len(charset_list)
    sequence_width = train_dataset.Xdata.shape[1]
    generation_length = (
        args.max_generation_length
        if args.max_generation_length is not None
        else sequence_width + 10
    )
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training sequence width: {sequence_width}")
    print(f"Maximum generation length: {generation_length}")

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=loader_generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = AAE(
        vocab_size=vocab_size,
        start_idx=start_idx,
        end_idx=end_idx,
        embedding_dim=args.embedding_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        decoder_hidden_dim=args.decoder_hidden_dim,
        latent_dim=args.latent_dim,
        bidirectional=args.bidirectional,
        discriminator_hidden_dim=args.discriminator_hidden_dim,
    ).to(device)
    print(
        "Number of trainable parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters()):,}"
    )

    generator_optimizer, discriminator_optimizer = make_optimizers(
        model, args.lr
    )
    reference_smiles = load_reference_smiles(
        data_directory, train_dataset
    )
    loss_rows = []
    vund_rows = []
    best_test_generator_loss = math.inf
    started = time.time()

    for epoch in range(args.epochs):
        train_metrics = train_epoch(
            model=model,
            loader=train_loader,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            device=device,
            reconstruction_ratio=args.reconstruction_ratio,
            start_idx=start_idx,
            end_idx=end_idx,
            gradient_clip=args.gradient_clip,
        )
        test_metrics = evaluate_epoch(
            model=model,
            loader=test_loader,
            device=device,
            reconstruction_ratio=args.reconstruction_ratio,
            start_idx=start_idx,
            end_idx=end_idx,
        )

        row = {"epoch": epoch}
        row.update(
            {f"train_{name}": value for name, value in train_metrics.items()}
        )
        row.update(
            {f"test_{name}": value for name, value in test_metrics.items()}
        )
        loss_rows.append(row)
        save_rows(
            os.path.join(save_directory, "loss_records.csv"),
            list(row.keys()),
            loss_rows,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "generator_optimizer_state_dict": (
                generator_optimizer.state_dict()
            ),
            "discriminator_optimizer_state_dict": (
                discriminator_optimizer.state_dict()
            ),
            "args": vars(args),
            "vocab": charset_list,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "architecture": "supplied unconditional molecular AAE",
        }
        torch.save(
            checkpoint,
            os.path.join(save_directory, f"model_epoch_{epoch:03d}.pth"),
        )
        if test_metrics["generator_loss"] < best_test_generator_loss:
            best_test_generator_loss = test_metrics["generator_loss"]
            torch.save(
                checkpoint,
                os.path.join(save_directory, "best_model.pth"),
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train_G={train_metrics['generator_loss']:.4f} "
            f"train_D={train_metrics['discriminator_loss']:.4f} "
            f"train_recon={train_metrics['reconstruction_loss']:.4f} | "
            f"test_G={test_metrics['generator_loss']:.4f} "
            f"test_D={test_metrics['discriminator_loss']:.4f} "
            f"test_recon={test_metrics['reconstruction_loss']:.4f}"
        )

        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            print(f"Starting VUND evaluation at epoch {epoch:03d}...")
            result = evaluate_vund(
                model=model,
                reference_smiles=reference_smiles,
                max_length=generation_length,
                total_to_generate=args.gen_total,
                generation_batch_size=args.gen_batch_size,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            vund_row = {
                "epoch": epoch,
                "validity": result["valid_ratio"],
                "uniqueness": result["unique_ratio"],
                "novelty": result["novelty_ratio"],
                "diversity": result["diversity_score"],
            }
            vund_rows.append(vund_row)
            save_rows(
                os.path.join(save_directory, "vund_records.csv"),
                list(vund_row.keys()),
                vund_rows,
            )

            generated_path = os.path.join(
                save_directory,
                f"generated_smiles_epoch_{epoch:03d}.csv",
            )
            with open(
                generated_path,
                "w",
                newline="",
                encoding="utf-8",
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["SMILES"])
                writer.writerows(
                    [[smiles] for smiles in result["generated_smiles"]]
                )

            print(
                f"VUND | validity={result['valid_ratio']:.4f} "
                f"({result['valid_count']}/{args.gen_total}) | "
                f"uniqueness={result['unique_ratio']:.4f} | "
                f"novelty={result['novelty_ratio']:.4f} | "
                f"diversity={result['diversity_score']:.4f}"
            )

    elapsed = time.time() - started
    print(f"Training finished in {elapsed:.2f} seconds.")
    print(
        "Best checkpoint: "
        f"{os.path.join(save_directory, 'best_model.pth')}"
    )


if __name__ == "__main__":
    main()

