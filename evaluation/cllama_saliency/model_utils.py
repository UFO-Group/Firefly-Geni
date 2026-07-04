# -*- coding: utf-8 -*-
"""
Model loading utilities.

No user-editable parameters are stored here. Pass all model/device settings
from the top-level run_cllama_saliency.py file.
"""

import torch

from module.char import charset_list
from module.LLaMa_3_scl import DarwinLLaMA
from module.dataload import UserDataset


def get_device(force_cpu=False, device_id=0):
    """
    Select CUDA device or CPU.
    """
    use_cuda = torch.cuda.is_available() and (not force_cpu)

    if use_cuda:
        device = torch.device("cuda:%d" % int(device_id))
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")
    return device


def load_cllama_model(
    datadir,
    model_path,
    device,
    dim,
    n_layers,
    n_heads,
    dropout,
):
    """
    Load CLLaMA model, character dictionary, and training SMILES length.
    """
    train_dataset = UserDataset(datadir, "train")
    dynamic_smiles_len = train_dataset.Xdata.shape[1]

    dict_len = len(charset_list)
    char_dict = {c: i for i, c in enumerate(charset_list)}
    idx_to_char = {i: c for c, i in char_dict.items()}

    print(f"\nCharacter set size: {dict_len}")
    print(f"Training SMILES length: {dynamic_smiles_len}")
    print(f"\nLoading model: {model_path}")

    model = DarwinLLaMA(
        vocab_size=dict_len,
        prop_len=2,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        max_seq_len=dynamic_smiles_len + 50,
        dropout=dropout,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("Model loaded.")

    return model, char_dict, idx_to_char, dynamic_smiles_len, dict_len