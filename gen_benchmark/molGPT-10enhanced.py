"""Train the original property-conditioned MolGPT on the Firefly-Geni data.

The MolGPT architecture and conditioning route are retained from the original
repository.  The only project-specific changes are the Firefly-Geni data/module
interfaces, training hyperparameter defaults, checkpointing, and VUND output.

Original MolGPT conditioning route:
    property vector -> Linear(num_props, n_embd) -> one property prefix token
    SMILES token -> token embedding + position embedding + token type embedding
    Transformer input -> [property prefix token, SMILES token sequence]

No activation, extra property encoder, repeated property concatenation, or
weight tying is introduced here.
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


class GPTConfig:
    """Configuration container used by the original MolGPT model."""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for key, value in kwargs.items():
            setattr(self, key, value)


class CausalSelfAttention(nn.Module):
    """Original MolGPT multi-head masked self-attention."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

        # In property-only MolGPT, exactly one prefix token is added.
        prefix_length = int(bool(config.num_props)) + int(config.scaffold_maxlen)
        total_length = config.block_size + prefix_length
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(total_length, total_length)).view(
                1, 1, total_length, total_length
            ),
        )
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        batch_size, seq_len, embedding_dim = x.size()

        key = self.key(x).view(
            batch_size, seq_len, self.n_head, embedding_dim // self.n_head
        ).transpose(1, 2)
        query = self.query(x).view(
            batch_size, seq_len, self.n_head, embedding_dim // self.n_head
        ).transpose(1, 2)
        value = self.value(x).view(
            batch_size, seq_len, self.n_head, embedding_dim // self.n_head
        ).transpose(1, 2)

        attention = (query @ key.transpose(-2, -1)) * (
            1.0 / math.sqrt(key.size(-1))
        )
        attention = attention.masked_fill(
            self.mask[:, :, :seq_len, :seq_len] == 0,
            float("-inf"),
        )
        attention = F.softmax(attention, dim=-1)
        attention_map = attention
        attention = self.attn_drop(attention)

        output = attention @ value
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embedding_dim
        )
        output = self.resid_drop(self.proj(output))
        return output, attention_map


class Block(nn.Module):
    """Original pre-layer-normalized MolGPT Transformer block."""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        attention_output, attention_map = self.attn(self.ln1(x))
        x = x + attention_output
        x = x + self.mlp(self.ln2(x))
        return x, attention_map


class GPT(nn.Module):
    """Original property-conditioned MolGPT model without scaffold conditioning."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.type_emb = nn.Embedding(2, config.n_embd)
        if config.num_props:
            # Exact MolGPT property encoder: one Linear layer, no activation.
            self.prop_nn = nn.Linear(config.num_props, config.n_embd)

        # The original implementation uses a learnable positional parameter.
        self.pos_emb = nn.Parameter(
            torch.zeros(1, config.block_size, config.n_embd)
        )
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(
            *[Block(config) for _ in range(config.n_layer)]
        )
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, learning_rate, weight_decay, betas):
        """Original MolGPT separation of decay and no-decay parameters."""

        decay = set()
        no_decay = set()
        whitelist = (nn.Linear, nn.LSTM)
        blacklist = (nn.LayerNorm, nn.Embedding)

        for module_name, module in self.named_modules():
            for parameter_name, _ in module.named_parameters():
                full_name = (
                    f"{module_name}.{parameter_name}"
                    if module_name
                    else parameter_name
                )
                if parameter_name.endswith("bias") or "bias" in parameter_name:
                    no_decay.add(full_name)
                elif (
                    parameter_name.endswith("weight") or "weight" in parameter_name
                ) and isinstance(module, whitelist):
                    decay.add(full_name)
                elif parameter_name.endswith("weight") and isinstance(
                    module, blacklist
                ):
                    no_decay.add(full_name)

        no_decay.add("pos_emb")
        parameters = {name: parameter for name, parameter in self.named_parameters()}
        overlap = decay & no_decay
        unassigned = parameters.keys() - (decay | no_decay)
        if overlap:
            raise RuntimeError(f"Parameters assigned twice: {sorted(overlap)}")
        if unassigned:
            raise RuntimeError(f"Parameters not assigned: {sorted(unassigned)}")

        groups = [
            {
                "params": [parameters[name] for name in sorted(decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [parameters[name] for name in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return torch.optim.AdamW(
            groups,
            lr=learning_rate,
            betas=betas,
        )

    def forward(self, idx, targets=None, prop=None, scaffold=None):
        batch_size, token_length = idx.size()
        if token_length > self.block_size:
            raise ValueError("Cannot forward: model block size is exhausted.")
        if self.config.num_props:
            if prop is None:
                raise ValueError("Property values are required.")
            if prop.size(-1) != self.config.num_props:
                raise ValueError(
                    f"Expected {self.config.num_props} properties, "
                    f"received {prop.size(-1)}."
                )

        token_embeddings = self.tok_emb(idx)
        position_embeddings = self.pos_emb[:, :token_length, :]
        token_type_embeddings = self.type_emb(
            torch.ones(
                (batch_size, token_length),
                dtype=torch.long,
                device=idx.device,
            )
        )
        x = self.drop(
            token_embeddings + position_embeddings + token_type_embeddings
        )

        if self.config.num_props:
            property_type_embedding = self.type_emb(
                torch.zeros(
                    (batch_size, 1),
                    dtype=torch.long,
                    device=idx.device,
                )
            )
            if prop.ndim == 2:
                property_embedding = self.prop_nn(prop.unsqueeze(1))
            else:
                property_embedding = self.prop_nn(prop)
            property_embedding = property_embedding + property_type_embedding

            # Exact original MolGPT conditioning: prepend the property token.
            x = torch.cat([property_embedding, x], dim=1)

        attention_maps = []
        for layer in self.blocks:
            x, attention_map = layer(x)
            attention_maps.append(attention_map)

        x = self.ln_f(x)
        logits = self.head(x)

        # Remove the property-prefix output before comparing with SMILES targets.
        prefix_length = int(bool(self.config.num_props))
        logits = logits[:, prefix_length:, :]

        loss = None
        if targets is not None:
            # Exact original objective: CE over the entire padded token sequence.
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss, attention_maps


def top_k_logits(logits, top_k):
    values, _ = torch.topk(logits, top_k)
    output = logits.clone()
    output[output < values[:, [-1]]] = -float("inf")
    return output


@torch.no_grad()
def sample(
    model,
    x,
    steps,
    temperature=1.0,
    do_sample=True,
    top_k=5,
    prop=None,
):
    """Original MolGPT autoregressive sampling procedure."""

    block_size = model.get_block_size()
    model.eval()
    for _ in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]
        logits, _, _ = model(x_cond, prop=prop)
        logits = logits[:, -1, :] / temperature
        if top_k is not None and top_k > 0:
            logits = top_k_logits(logits, min(top_k, logits.size(-1)))
        probabilities = F.softmax(logits, dim=-1)
        if do_sample:
            next_token = torch.multinomial(probabilities, num_samples=1)
        else:
            _, next_token = torch.topk(probabilities, k=1, dim=-1)
        x = torch.cat((x, next_token), dim=1)
    return x


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_base_train_smiles(data_dir):
    train_csv = os.path.join(data_dir, "train.csv")
    try:
        frame = pd.read_csv(train_csv)
        if "SMILES" in frame.columns:
            column = "SMILES"
        elif "TADF_SMILES" in frame.columns:
            column = "TADF_SMILES"
        else:
            column = frame.columns[0]
        return frame[column].dropna().astype(str).tolist()
    except Exception as exc:
        print(f"Warning: novelty reference set could not be loaded: {exc}")
        return []


def decode_generated(sequences):
    generated = []
    for sequence in sequences:
        smiles = vec_to_char(sequence.detach().cpu().numpy(), charset_list)
        smiles = smiles.replace("^", "").split(">")[0]
        generated.append(smiles)
    return generated


@torch.no_grad()
def evaluate_vund(
    model,
    base_train_smiles,
    train_props,
    prop_len,
    start_idx,
    device,
    gen_total,
    gen_batch_size,
    prop_source,
    temperature,
    top_k,
):
    model.eval()
    generated_smiles = []
    completed = 0

    while completed < gen_total:
        current_batch = min(gen_batch_size, gen_total - completed)
        if prop_source == "train":
            indices = torch.randint(
                0, train_props.size(0), (current_batch,)
            )
            props = train_props[indices].to(device)
        else:
            props = torch.rand(current_batch, prop_len, device=device)

        context = torch.full(
            (current_batch, 1),
            start_idx,
            dtype=torch.long,
            device=device,
        )
        sequences = sample(
            model=model,
            x=context,
            # Match CLLaMA/AAE/CVAE: allow ten positions beyond the
            # tokenized training-sequence width during generation.
            steps=model.get_block_size() + 10,
            temperature=temperature,
            do_sample=True,
            top_k=top_k,
            prop=props,
        )
        generated_smiles.extend(decode_generated(sequences))
        completed += current_batch
        print(f"    Generated {completed}/{gen_total}")

    valid_count, valid_ratio = valid_molecules(generated_smiles)
    unique_ratio = uniqueness(generated_smiles)
    novelty_ratio = novelty(generated_smiles, base_train_smiles)
    diversity_source = generated_smiles[:1000]
    diversity_score = (
        calculate_diversity(diversity_source) if diversity_source else 0.0
    )
    return {
        "valid_count": valid_count,
        "valid_ratio": valid_ratio,
        "unique_ratio": unique_ratio,
        "novelty_ratio": novelty_ratio,
        "diversity_score": diversity_score,
        "generated_smiles": generated_smiles,
    }


def set_learning_rate(
    optimizer,
    tokens_processed,
    base_learning_rate,
    warmup_tokens,
    final_tokens,
):
    """Original MolGPT linear-warmup/cosine-decay schedule."""

    if tokens_processed < warmup_tokens:
        multiplier = float(tokens_processed) / float(max(1, warmup_tokens))
    else:
        progress = float(tokens_processed - warmup_tokens) / float(
            max(1, final_tokens - warmup_tokens)
        )
        multiplier = max(
            0.1,
            0.5 * (1.0 + math.cos(math.pi * progress)),
        )
    learning_rate = base_learning_rate * multiplier
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def run_epoch(
    model,
    loader,
    optimizer,
    device,
    grad_clip,
    schedule_state=None,
):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    batches = 0
    current_lr = None

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            if len(batch) != 3:
                raise RuntimeError(
                    "MolGPT requires (tokens, lengths, properties) from UserDataset."
                )
            tokens, _lengths, props = batch
            tokens = tokens.to(device)
            props = props.to(device).float()
            model_inputs = tokens[:, :-1]
            targets = tokens[:, 1:]

            if training:
                optimizer.zero_grad(set_to_none=True)

            _, loss, _ = model(
                model_inputs,
                targets=targets,
                prop=props,
            )

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                schedule_state["tokens"] += targets.numel()
                current_lr = set_learning_rate(
                    optimizer=optimizer,
                    tokens_processed=schedule_state["tokens"],
                    base_learning_rate=schedule_state["base_lr"],
                    warmup_tokens=schedule_state["warmup_tokens"],
                    final_tokens=schedule_state["final_tokens"],
                )

            total_loss += loss.item()
            batches += 1

    if batches == 0:
        raise RuntimeError("The DataLoader produced no batches.")
    return total_loss / batches, current_lr


def save_rows(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the original property-conditioned MolGPT architecture on "
            "the 10x-augmented Firefly-Geni TADF token dataset."
        )
    )

    # CLLaMA-aligned defaults where the same hyperparameter exists.
    parser.add_argument("--n-layers", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    # Optimizer/sampling values remain individually configurable.
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--gen-total", type=int, default=10000)
    parser.add_argument("--gen-batch-size", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--eval-prop-source",
        choices=["uniform", "train"],
        default="uniform",
    )
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = args.data_dir or os.path.abspath(
        os.path.join(
            script_dir,
            "..",
            "dataset_tadf",
            "dataset_gen",
            "gendata_est_sa",
            "enhanced10",
        )
    )
    save_dir = args.save_dir or os.path.join(
        script_dir,
        "MolGPT_10enhanced_original-tok5",
        (
            f"layers{args.n_layers}_heads{args.n_heads}_"
            f"emb{args.embedding_dim}_bs{args.batch_size}_lr{args.lr}"
        ),
    )
    os.makedirs(save_dir, exist_ok=True)

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {save_dir}")

    train_dataset = UserDataset(data_dir, "train")
    test_dataset = UserDataset(data_dir, "test")
    if train_dataset.Pdata is None or test_dataset.Pdata is None:
        raise RuntimeError("Ptrain.npy and Ptest.npy are required.")

    full_sequence_length = train_dataset.Xdata.shape[1]
    if test_dataset.Xdata.shape[1] != full_sequence_length:
        raise RuntimeError("Train and test sequence lengths differ.")
    block_size = full_sequence_length - 1
    num_props = train_dataset.Pdata.shape[1]
    vocab_size = len(charset_list)
    char_to_index = {
        character: index for index, character in enumerate(charset_list)
    }
    if "^" not in char_to_index or ">" not in char_to_index:
        raise RuntimeError("charset_list must contain '^' and '>'.")

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Full SMILES sequence length: {full_sequence_length}")
    print(f"MolGPT block size: {block_size}")
    print(f"Conditional property count: {num_props}")
    print(f"Vocabulary size: {vocab_size}")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        num_props=num_props,
        n_layer=args.n_layers,
        n_head=args.n_heads,
        n_embd=args.embedding_dim,
        scaffold=False,
        scaffold_maxlen=0,
        lstm=False,
        lstm_layers=0,
        embd_pdrop=args.dropout,
        resid_pdrop=args.dropout,
        attn_pdrop=args.dropout,
    )
    model = GPT(config).to(device)
    print(
        "Number of trainable parameters: "
        f"{sum(parameter.numel() for parameter in model.parameters()):,}"
    )

    optimizer = model.configure_optimizers(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    tokens_per_epoch = len(train_dataset) * block_size
    schedule_state = {
        "tokens": 0,
        "base_lr": args.lr,
        "warmup_tokens": int(0.1 * tokens_per_epoch),
        "final_tokens": int(args.epochs * tokens_per_epoch),
    }

    base_train_smiles = load_base_train_smiles(data_dir)
    train_props_cpu = train_dataset.Pdata.detach().cpu()
    loss_rows = []
    vund_rows = []
    best_test_loss = math.inf
    started = time.time()

    for epoch in range(args.epochs):
        train_loss, current_lr = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            schedule_state=schedule_state,
        )
        test_loss, _ = run_epoch(
            model=model,
            loader=test_loader,
            optimizer=None,
            device=device,
            grad_clip=args.grad_clip,
        )

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_loss,
            "test_loss": test_loss,
        }
        loss_rows.append(row)
        save_rows(
            os.path.join(save_dir, "loss_records.csv"),
            list(row.keys()),
            loss_rows,
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": vars(args),
            "vocab": charset_list,
            "block_size": block_size,
            "num_props": num_props,
            "schedule_state": schedule_state,
        }
        torch.save(
            checkpoint,
            os.path.join(save_dir, f"model_epoch_{epoch:03d}.pth"),
        )
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(checkpoint, os.path.join(save_dir, "best_model.pth"))

        print(
            f"Epoch {epoch:03d} | lr={current_lr:.2e} | "
            f"train_loss={train_loss:.4f} | test_loss={test_loss:.4f}"
        )

        if args.eval_every > 0 and (epoch + 1) % args.eval_every == 0:
            print(f"Starting VUND evaluation at epoch {epoch:03d}...")
            result = evaluate_vund(
                model=model,
                base_train_smiles=base_train_smiles,
                train_props=train_props_cpu,
                prop_len=num_props,
                start_idx=char_to_index["^"],
                device=device,
                gen_total=args.gen_total,
                gen_batch_size=args.gen_batch_size,
                prop_source=args.eval_prop_source,
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
                os.path.join(save_dir, "vund_records.csv"),
                list(vund_row.keys()),
                vund_rows,
            )

            generated_path = os.path.join(
                save_dir,
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
    print(f"Best checkpoint: {os.path.join(save_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()
