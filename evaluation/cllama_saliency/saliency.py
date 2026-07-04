# -*- coding: utf-8 -*-
"""
Calculation of Low-EST-specific generation saliency.
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from .data_utils import normalize_value
from .smiles_utils import clean_smiles, check_smiles_and_charset
from .mapping import smiles_char_to_atom_bond_topology_map, aggregate_token_scores_to_structure


def smiles_to_input_target(smi, char_dict, device):
    """
    teacher forcing input

    input：
        ^ + SMILES

    target：
        SMILES + >
    """
    input_str = "^" + smi
    target_str = smi + ">"

    input_ids = torch.tensor(
        [[char_dict[ch] for ch in input_str]],
        dtype=torch.long,
        device=device,
    )

    target_ids = torch.tensor(
        [[char_dict[ch] for ch in target_str]],
        dtype=torch.long,
        device=device,
    )

    return input_ids, target_ids, input_str, target_str


def build_props_tensor(est_value, sa_value, est_min, est_max, sa_min, sa_max, device):
    """
    Construct the normalized props tensor.

    props = [EST_norm, SA_norm]
    """
    est_norm = normalize_value(est_value, est_min, est_max)
    sa_norm = normalize_value(sa_value, sa_min, sa_max)

    props = torch.tensor(
        [[est_norm, sa_norm]],
        dtype=torch.float32,
        device=device,
    )

    prop_mask = torch.ones_like(props, device=device)
    return props, prop_mask


def get_token_log_probs(model, smi, char_dict, props_norm, prop_mask, device):
    """
    Do teacher forcing for fixed SMILES.

    return：
        token_log_probs: The log probability of each target token, with a length of len(SMILES) + 1
        target_str: SMILES + ">"
    """
    tokens, targets, input_str, target_str = smiles_to_input_target(
        smi=smi,
        char_dict=char_dict,
        device=device,
    )

    logits, _ = model(tokens=tokens, props=props_norm, prop_mask=prop_mask)

    prefix_len = model.prop_len if props_norm is not None else 0
    valid_logits = logits[:, prefix_len:, :].contiguous()

    log_probs = F.log_softmax(valid_logits, dim=-1)

    seq_len = targets.shape[1]
    token_log_probs = log_probs[
        0,
        torch.arange(seq_len, device=device),
        targets[0],
    ]

    return token_log_probs, target_str


def est_likelihood_shift_token_scores(
    model,
    smi,
    char_dict,
    est_low,
    est_high,
    sa_value,
    est_min,
    est_max,
    sa_min,
    sa_max,
    device,
):
    """
    Conditional perturbation method.

    signed_score_t =
        logP(token | low EST) - logP(token | high EST)

    Final retention:
        saliency_t = max(0, signed_score_t)
    """
    model.eval()

    props_low, prop_mask_low = build_props_tensor(
        est_value=est_low,
        sa_value=sa_value,
        est_min=est_min,
        est_max=est_max,
        sa_min=sa_min,
        sa_max=sa_max,
        device=device,
    )

    props_high, prop_mask_high = build_props_tensor(
        est_value=est_high,
        sa_value=sa_value,
        est_min=est_min,
        est_max=est_max,
        sa_min=sa_min,
        sa_max=sa_max,
        device=device,
    )

    with torch.no_grad():
        logp_low, target_str = get_token_log_probs(
            model=model,
            smi=smi,
            char_dict=char_dict,
            props_norm=props_low,
            prop_mask=prop_mask_low,
            device=device,
        )

        logp_high, _ = get_token_log_probs(
            model=model,
            smi=smi,
            char_dict=char_dict,
            props_norm=props_high,
            prop_mask=prop_mask_high,
            device=device,
        )

    signed_shift_scores = (
        logp_low[:len(smi)] - logp_high[:len(smi)]
    ).detach().cpu().numpy().astype(float)

    saliency_scores = np.maximum(0.0, signed_shift_scores)

    return signed_shift_scores, saliency_scores, target_str[:len(smi)]


def compute_lowEST_scores_for_one_smiles(
    model,
    smi,
    char_dict,
    est_low,
    est_high,
    sa_value,
    est_min,
    est_max,
    sa_min,
    sa_max,
    device,
    atom_agg_mode="sum",
    include_topology_on_atoms=True,
):
    """
    Calculate for a SMILES expression:
        - token-level lowEST saliency
        - atom + bond + topology saliency
    """
    smi = clean_smiles(smi)
    mol = check_smiles_and_charset(smi, char_dict)

    shift_signed_token_scores, lowEST_specific_token_scores, token_str = (
        est_likelihood_shift_token_scores(
            model=model,
            smi=smi,
            char_dict=char_dict,
            est_low=est_low,
            est_high=est_high,
            sa_value=sa_value,
            est_min=est_min,
            est_max=est_max,
            sa_min=sa_min,
            sa_max=sa_max,
            device=device,
        )
    )

    lowEST_structure = aggregate_token_scores_to_structure(
        smi=smi,
        token_scores=lowEST_specific_token_scores,
        mode=atom_agg_mode,
        include_topology_on_atoms=include_topology_on_atoms,
    )

    atom_char_map, bond_char_map, topology_atom_map, _ = smiles_char_to_atom_bond_topology_map(smi)

    token_rows = []
    for pos, ch in enumerate(smi):
        structural_type = "unmapped"
        mapped_atom = None
        mapped_bond = None

        if pos in atom_char_map:
            structural_type = "atom"
            mapped_atom = atom_char_map[pos]
        elif pos in bond_char_map:
            structural_type = "bond_or_ring_closure"
            mapped_bond = "-".join(map(str, bond_char_map[pos]))
        elif pos in topology_atom_map:
            structural_type = "topology_attachment"
            mapped_atom = topology_atom_map[pos]

        token_rows.append({
            "smiles": smi,
            "token_position": pos,
            "token": ch,
            "structural_type": structural_type,
            "mapped_atom_index": mapped_atom,
            "mapped_bond_atom_pair": mapped_bond,
            "likelihood_shift_signed_score": shift_signed_token_scores[pos],
            "lowEST_specific_generation_saliency": lowEST_specific_token_scores[pos],
        })

    token_df = pd.DataFrame(token_rows)

    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    n_atoms = min(
        len(atom_symbols),
        len(lowEST_structure["atom_scores"]),
        len(lowEST_structure["atom_scores_for_plot"]),
    )

    atom_rows = []
    for i in range(n_atoms):
        atom_rows.append({
            "smiles": smi,
            "atom_index_in_this_smiles": i,
            "atom_symbol": atom_symbols[i],
            "lowEST_atom_only_saliency": lowEST_structure["atom_scores"][i],
            "lowEST_topology_only_saliency": lowEST_structure["topology_atom_scores"][i],
            "lowEST_atom_topology_saliency_for_plot": lowEST_structure["atom_scores_for_plot"][i],
        })

    atom_df = pd.DataFrame(atom_rows)

    bond_rows = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        key = tuple(sorted((int(a), int(b))))

        bond_rows.append({
            "smiles": smi,
            "bond_atom_pair": f"{key[0]}-{key[1]}",
            "begin_atom_index": key[0],
            "end_atom_index": key[1],
            "bond_type": str(bond.GetBondType()),
            "lowEST_bond_saliency": lowEST_structure["bond_scores_by_key"].get(key, 0.0),
        })

    bond_df = pd.DataFrame(bond_rows)

    return {
        "smi": smi,
        "mol": mol,
        "token_df": token_df,
        "atom_df": atom_df,
        "bond_df": bond_df,
        "shift_signed_token_scores": shift_signed_token_scores,
        "lowEST_specific_token_scores": lowEST_specific_token_scores,
        "lowEST_structure": lowEST_structure,
    }
