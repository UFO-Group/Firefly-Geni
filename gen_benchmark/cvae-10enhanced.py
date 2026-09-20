"""PyTorch port of the molecular CVAE of Lim et al. (2018).

Architecture retained from the original implementation:
- LSTM encoder input at every step: [token embedding, raw properties]
- latent mean/log-variance from the final encoder hidden state
- LSTM decoder input at every step: [z, token embedding, raw properties]
- zero decoder initial state
- reconstruction cross-entropy + standard KL loss
- N(0, I) prior sampling and temperature/top-k autoregressive generation

Only the data/training interface is adapted to the existing Firefly-Geni
module directory and the same token_dataset used by CLLaMA.
"""

import argparse
import csv
import math
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

from module.char import charset_list
from module.dataload import UserDataset
from module.other_function import vec_to_char
from module.eval_def import (
    calculate_diversity,
    novelty,
    uniqueness,
    valid_molecules,
)


class LimCVAE(nn.Module):
    """Lim et al. molecular CVAE implemented in PyTorch."""

    def __init__(
        self,
        vocab_size,
        num_properties,
        start_idx,
        end_idx,
        latent_size=200,
        hidden_size=512,
        n_rnn_layers=3,
        latent_mean=0.0,
        latent_stddev=1.0,
    ):
        super().__init__()
        if latent_size < vocab_size:
            raise ValueError(
                "latent_size must be at least vocab_size to preserve the "
                "embedding layout used in the original Lim implementation."
            )

        self.vocab_size = vocab_size
        self.num_properties = num_properties
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.n_rnn_layers = n_rnn_layers
        self.latent_mean = latent_mean
        self.latent_stddev = latent_stddev

        # Original TensorFlow shape: [latent_size, vocab_size]. Token IDs index
        # the first dimension, so the effective embedding width is vocab_size.
        self.token_embedding = nn.Embedding(latent_size, vocab_size)

        self.encoder = nn.LSTM(
            input_size=vocab_size + num_properties,
            hidden_size=hidden_size,
            num_layers=n_rnn_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.to_mean = nn.Linear(hidden_size, latent_size)
        self.to_log_variance = nn.Linear(hidden_size, latent_size)

        self.decoder = nn.LSTM(
            input_size=latent_size + vocab_size + num_properties,
            hidden_size=hidden_size,
            num_layers=n_rnn_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.output_layer = nn.Linear(hidden_size, vocab_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.token_embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.output_layer.weight, -0.1, 0.1)
        nn.init.zeros_(self.output_layer.bias)
        nn.init.xavier_uniform_(self.to_mean.weight)
        nn.init.zeros_(self.to_mean.bias)
        nn.init.xavier_uniform_(self.to_log_variance.weight)
        nn.init.zeros_(self.to_log_variance.bias)

    def encode(self, tokens, lengths, properties):
        token_features = self.token_embedding(tokens)
        repeated_properties = properties.unsqueeze(1).expand(
            -1, tokens.size(1), -1
        )
        encoder_input = torch.cat(
            [token_features, repeated_properties], dim=-1
        )

        packed = pack_padded_sequence(
            encoder_input,
            lengths.clamp(min=1, max=tokens.size(1)).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.encoder(packed)
        final_hidden = hidden[-1]
        return self.to_mean(final_hidden), self.to_log_variance(final_hidden)

    def reparameterize(self, mean, log_variance):
        epsilon = torch.normal(
            mean=self.latent_mean,
            std=self.latent_stddev,
            size=mean.shape,
            device=mean.device,
        )
        return mean + torch.exp(log_variance / 2.0) * epsilon

    def zero_decoder_state(self, batch_size, device):
        hidden = torch.zeros(
            self.n_rnn_layers,
            batch_size,
            self.hidden_size,
            device=device,
        )
        return hidden, torch.zeros_like(hidden)

    def decode_teacher_forcing(self, tokens, z, properties):
        token_features = self.token_embedding(tokens)
        repeated_z = z.unsqueeze(1).expand(-1, tokens.size(1), -1)
        repeated_properties = properties.unsqueeze(1).expand(
            -1, tokens.size(1), -1
        )
        decoder_input = torch.cat(
            [repeated_z, token_features, repeated_properties], dim=-1
        )
        decoder_output, _ = self.decoder(
            decoder_input,
            self.zero_decoder_state(tokens.size(0), tokens.device),
        )
        return self.output_layer(decoder_output)

    def forward(self, tokens, lengths, properties):
        mean, log_variance = self.encode(tokens, lengths, properties)
        z = self.reparameterize(mean, log_variance)
        logits = self.decode_teacher_forcing(tokens, z, properties)
        return logits, mean, log_variance

    @torch.no_grad()
    def generate(
        self,
        properties,
        max_length,
        z=None,
        temperature=1.0,
        top_k=10,
    ):
        """Prior sampling with temperature-scaled top-k multinomial decoding."""
        self.eval()
        batch_size = properties.size(0)
        device = properties.device

        if z is None:
            z = torch.normal(
                mean=self.latent_mean,
                std=self.latent_stddev,
                size=(batch_size, self.latent_size),
                device=device,
            )

        state = self.zero_decoder_state(batch_size, device)
        current_token = torch.full(
            (batch_size,), self.start_idx, dtype=torch.long, device=device
        )
        generated = [current_token]
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length - 1):
            token_features = self.token_embedding(current_token).unsqueeze(1)
            decoder_input = torch.cat(
                [z.unsqueeze(1), token_features, properties.unsqueeze(1)],
                dim=-1,
            )
            decoder_output, state = self.decoder(decoder_input, state)
            logits = self.output_layer(decoder_output[:, -1, :])
            temperature = max(float(temperature), 1e-8)
            logits = logits / temperature

            if top_k is not None and top_k > 0:
                k = min(int(top_k), logits.size(-1))
                top_values, top_indices = torch.topk(logits, k=k, dim=-1)
                top_probabilities = F.softmax(top_values, dim=-1)
                sampled_position = torch.multinomial(
                    top_probabilities, num_samples=1
                )
                next_token = top_indices.gather(
                    dim=-1, index=sampled_position
                ).squeeze(-1)
            else:
                probabilities = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(
                    probabilities, num_samples=1
                ).squeeze(-1)
            next_token = torch.where(
                finished,
                torch.full_like(next_token, self.end_idx),
                next_token,
            )
            generated.append(next_token)
            finished = finished | (next_token == self.end_idx)
            current_token = next_token
            if bool(finished.all()):
                break

        return torch.stack(generated, dim=1)


def lim_cvae_loss(logits, targets, lengths, mean, log_variance):
    """Original objective: reconstruction loss + unweighted KL loss."""
    batch_size, seq_len, vocab_size = logits.shape
    token_losses = F.cross_entropy(
        logits.reshape(batch_size * seq_len, vocab_size),
        targets.reshape(batch_size * seq_len),
        reduction="none",
    ).view(batch_size, seq_len)

    positions = torch.arange(seq_len, device=targets.device).unsqueeze(0)
    mask = positions < lengths.clamp(min=1, max=seq_len).unsqueeze(1)
    reconstruction_loss = (
        token_losses * mask
    ).sum() / mask.sum().clamp(min=1)

    # Matches tf.reduce_mean in the original implementation.
    latent_loss = torch.mean(
        -0.5
        * (
            1.0
            + log_variance
            - mean.pow(2)
            - torch.exp(log_variance)
        )
    )
    return (
        reconstruction_loss + latent_loss,
        reconstruction_loss,
        latent_loss,
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base_train_smiles(data_directory):
    train_csv = os.path.join(data_directory, "train.csv")
    print(f"Loading the non-augmented reference set: {train_csv}")
    try:
        frame = pd.read_csv(train_csv)
        if "SMILES" in frame.columns:
            column = "SMILES"
        elif "TADF_SMILES" in frame.columns:
            column = "TADF_SMILES"
        else:
            column = frame.columns[0]
        smiles = frame[column].dropna().astype(str).tolist()
        print(f"Loaded {len(smiles)} reference SMILES for novelty.")
        return smiles
    except Exception as exc:
        print(f"Warning: failed to load novelty reference set: {exc}")
        return []


def decode_generated(sequences):
    output = []
    for sequence in sequences:
        smiles = vec_to_char(
            sequence.detach().cpu().numpy(), charset_list
        )
        output.append(smiles.replace("^", "").split(">")[0])
    return output


@torch.no_grad()
def evaluate_vund(
    model,
    base_train_smiles,
    train_properties,
    num_properties,
    max_length,
    device,
    total_to_generate=10000,
    generation_batch_size=100,
    property_source="uniform",
    temperature=1.0,
    top_k=5,
):
    model.eval()
    generated_smiles = []
    completed = 0

    while completed < total_to_generate:
        current_batch = min(
            generation_batch_size, total_to_generate - completed
        )
        if property_source == "train":
            indices = torch.randint(
                0, train_properties.size(0), (current_batch,)
            )
            properties = train_properties[indices].to(device)
        else:
            properties = torch.rand(
                current_batch, num_properties, device=device
            )

        sequences = model.generate(
            properties,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
        )
        generated_smiles.extend(decode_generated(sequences))
        completed += current_batch
        print(f"    Generated {completed}/{total_to_generate}")

    valid_count, valid_ratio = valid_molecules(generated_smiles)
    unique_ratio = uniqueness(generated_smiles)
    novelty_ratio = novelty(generated_smiles, base_train_smiles)
    diversity_score = calculate_diversity(generated_smiles[:1000])
    return {
        "valid_count": valid_count,
        "valid_ratio": valid_ratio,
        "unique_ratio": unique_ratio,
        "novelty_ratio": novelty_ratio,
        "diversity_score": diversity_score,
        "generated_smiles": generated_smiles,
    }


def run_epoch(model, data_loader, optimizer, device, gradient_clip):
    training = optimizer is not None
    model.train(training)
    sums = {"loss": 0.0, "reconstruction": 0.0, "latent": 0.0}
    batches = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in data_loader:
            if len(batch) != 3:
                raise RuntimeError(
                    "Expected (tokens, lengths, properties) from UserDataset."
                )
            tokens, lengths, properties = batch
            tokens = tokens.to(device)
            lengths = lengths.to(device)
            properties = properties.to(device).float()

            model_inputs = tokens[:, :-1]
            targets = tokens[:, 1:]
            effective_lengths = lengths.clamp(
                min=1, max=targets.size(1)
            )

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits, mean, log_variance = model(
                model_inputs, effective_lengths, properties
            )
            loss, reconstruction_loss, latent_loss = lim_cvae_loss(
                logits,
                targets,
                effective_lengths,
                mean,
                log_variance,
            )

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip
                )
                optimizer.step()

            sums["loss"] += loss.item()
            sums["reconstruction"] += reconstruction_loss.item()
            sums["latent"] += latent_loss.item()
            batches += 1

    if batches == 0:
        raise RuntimeError("The DataLoader produced no batches.")
    return {key: value / batches for key, value in sums.items()}


def save_rows(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the Lim et al. CVAE with the existing 10x-augmented "
            "TADF token dataset."
        )
    )
    # Original Lim architecture defaults.
    parser.add_argument("--latent-size", type=int, default=200)
    parser.add_argument("--unit-size", type=int, default=512)
    parser.add_argument("--n-rnn-layers", type=int, default=3)
    parser.add_argument("--latent-mean", type=float, default=0.0)
    parser.add_argument("--latent-stddev", type=float, default=1.0)

    # Shared training settings aligned with CLLaMA where possible.
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--gen-total", type=int, default=10000)
    parser.add_argument("--gen-batch-size", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--eval-property-source",
        choices=["uniform", "train"],
        default="uniform",
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    torch.set_num_threads(10 if use_cuda else 20)
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
        "CVAE_10enhanced_topk5",
        (
            f"limcvae_latent{args.latent_size}_hidden{args.unit_size}_"
            f"layers{args.n_rnn_layers}_bs{args.batch_size}_lr{args.lr}"
        ),
    )
    os.makedirs(save_directory, exist_ok=True)

    print(f"Data directory: {data_directory}")
    print(f"Output directory: {save_directory}")
    train_dataset = UserDataset(data_directory, "train")
    test_dataset = UserDataset(data_directory, "test")
    if train_dataset.Pdata is None or test_dataset.Pdata is None:
        raise RuntimeError("Ptrain.npy and Ptest.npy are required.")

    smiles_length = train_dataset.Xdata.shape[1]
    num_properties = train_dataset.Pdata.shape[1]
    vocab_size = len(charset_list)
    char_to_index = {
        character: index for index, character in enumerate(charset_list)
    }
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"SMILES sequence length: {smiles_length}")
    print(f"Conditional property count: {num_properties}")
    print(f"Vocabulary size: {vocab_size}")

    loader_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        generator=loader_generator,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )

    model = LimCVAE(
        vocab_size=vocab_size,
        num_properties=num_properties,
        start_idx=char_to_index["^"],
        end_idx=char_to_index[">"],
        latent_size=args.latent_size,
        hidden_size=args.unit_size,
        n_rnn_layers=args.n_rnn_layers,
        latent_mean=args.latent_mean,
        latent_stddev=args.latent_stddev,
    ).to(device)
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Number of trainable parameters: {parameter_count:,}")

    # Original CVAE optimizer family; learning rate aligned with CLLaMA.
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    base_train_smiles = load_base_train_smiles(data_directory)
    train_properties_cpu = train_dataset.Pdata.detach().cpu()
    loss_records = []
    vund_records = []
    best_test_loss = math.inf
    started = time.time()

    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model, train_loader, optimizer, device, args.gradient_clip
        )
        test_metrics = run_epoch(
            model, test_loader, None, device, args.gradient_clip
        )

        loss_row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_reconstruction": train_metrics["reconstruction"],
            "train_latent": train_metrics["latent"],
            "test_loss": test_metrics["loss"],
            "test_reconstruction": test_metrics["reconstruction"],
            "test_latent": test_metrics["latent"],
        }
        loss_records.append(loss_row)
        save_rows(
            os.path.join(save_directory, "loss_records.csv"),
            list(loss_row.keys()),
            loss_records,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "vocab": charset_list,
            "smiles_length": smiles_length,
            "num_properties": num_properties,
            "architecture": "Lim et al. molecular CVAE PyTorch port",
        }
        torch.save(
            checkpoint,
            os.path.join(save_directory, f"model_epoch_{epoch:03d}.pth"),
        )
        if test_metrics["loss"] < best_test_loss:
            best_test_loss = test_metrics["loss"]
            torch.save(
                checkpoint,
                os.path.join(save_directory, "best_model.pth"),
            )

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_metrics['loss']:.4f} "
            f"(recon={train_metrics['reconstruction']:.4f}, "
            f"latent={train_metrics['latent']:.4f}) | "
            f"test={test_metrics['loss']:.4f} "
            f"(recon={test_metrics['reconstruction']:.4f}, "
            f"latent={test_metrics['latent']:.4f})"
        )

        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            print(f"Starting VUND evaluation at epoch {epoch:03d}...")
            result = evaluate_vund(
                model=model,
                base_train_smiles=base_train_smiles,
                train_properties=train_properties_cpu,
                num_properties=num_properties,
                max_length=smiles_length + 10,
                device=device,
                total_to_generate=args.gen_total,
                generation_batch_size=args.gen_batch_size,
                property_source=args.eval_property_source,
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
            vund_records.append(vund_row)
            save_rows(
                os.path.join(save_directory, "vund_records.csv"),
                list(vund_row.keys()),
                vund_records,
            )

            generated_path = os.path.join(
                save_directory, f"generated_smiles_epoch_{epoch:03d}.csv"
            )
            with open(
                generated_path, "w", newline="", encoding="utf-8"
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

    print(f"Training finished in {time.time() - started:.2f} seconds.")
    print(f"Best checkpoint: {os.path.join(save_directory, 'best_model.pth')}")


if __name__ == "__main__":
    main()


