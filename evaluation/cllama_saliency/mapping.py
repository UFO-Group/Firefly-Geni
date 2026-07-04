# -*- coding: utf-8 -*-
"""
The mapping of SMILES Tokens to atom/bond/topology.
"""

import numpy as np


def smiles_char_to_atom_bond_topology_map(smi):
    """
    Map the positions of the SMILES characters to:
        1. atom_char_map:
           atom char -> atom index

        2. bond_char_map:
           Bond symbols, ring numbers -> bond atom pair, e.g. (3, 8)

        3. topology_atom_map:
           Topological symbols such as branch parentheses -> attachment atom index
    """
    atom_char_map = {}
    bond_char_map = {}
    topology_atom_map = {}

    one_char_atoms = set([
        "B", "C", "N", "O", "P", "S", "F", "I", "H",
        "b", "c", "n", "o", "p", "s",
    ])

    two_char_atoms = set([
        "Cl", "Br", "Si", "Se", "As", "Al", "Li",
        "Na", "Mg", "Ca", "Zn", "Sn", "Te",
    ])

    bond_symbols = set(["-", "=", "#", ":", "/", "\\"])

    atom_idx = 0
    last_atom = None
    branch_stack = []
    ring_open = {}
    pending_bond_positions = []

    i = 0

    while i < len(smi):
        ch = smi[i]

        if ch == "[":
            j = smi.find("]", i)
            if j == -1:
                i += 1
                continue

            current_atom = atom_idx

            for k in range(i, j + 1):
                atom_char_map[k] = current_atom

            if last_atom is not None and len(pending_bond_positions) > 0:
                bond_key = tuple(sorted((last_atom, current_atom)))
                for k in pending_bond_positions:
                    bond_char_map[k] = bond_key
                pending_bond_positions = []

            last_atom = current_atom
            atom_idx += 1
            i = j + 1
            continue

        if i + 1 < len(smi) and smi[i:i + 2] in two_char_atoms:
            current_atom = atom_idx

            atom_char_map[i] = current_atom
            atom_char_map[i + 1] = current_atom

            if last_atom is not None and len(pending_bond_positions) > 0:
                bond_key = tuple(sorted((last_atom, current_atom)))
                for k in pending_bond_positions:
                    bond_char_map[k] = bond_key
                pending_bond_positions = []

            last_atom = current_atom
            atom_idx += 1
            i += 2
            continue

        if ch in one_char_atoms:
            current_atom = atom_idx

            atom_char_map[i] = current_atom

            if last_atom is not None and len(pending_bond_positions) > 0:
                bond_key = tuple(sorted((last_atom, current_atom)))
                for k in pending_bond_positions:
                    bond_char_map[k] = bond_key
                pending_bond_positions = []

            last_atom = current_atom
            atom_idx += 1
            i += 1
            continue

        if ch in bond_symbols:
            pending_bond_positions.append(i)
            i += 1
            continue

        if ch == "(":
            if last_atom is not None:
                branch_stack.append(last_atom)
                topology_atom_map[i] = last_atom
            i += 1
            continue

        if ch == ")":
            if len(branch_stack) > 0:
                attach_atom = branch_stack.pop()
                topology_atom_map[i] = attach_atom
                last_atom = attach_atom
            elif last_atom is not None:
                topology_atom_map[i] = last_atom
            i += 1
            continue

        if ch.isdigit() or ch == "%":
            if ch == "%" and i + 2 < len(smi):
                ring_id = smi[i:i + 3]
                ring_positions = [i, i + 1, i + 2]
                i_next = i + 3
            else:
                ring_id = ch
                ring_positions = [i]
                i_next = i + 1

            if last_atom is not None:
                if ring_id in ring_open:
                    open_atom, open_positions = ring_open.pop(ring_id)
                    bond_key = tuple(sorted((open_atom, last_atom)))

                    for k in open_positions + ring_positions:
                        bond_char_map[k] = bond_key

                    for k in pending_bond_positions:
                        bond_char_map[k] = bond_key

                    pending_bond_positions = []
                else:
                    ring_open[ring_id] = (last_atom, ring_positions)

                    for k in ring_positions:
                        topology_atom_map[k] = last_atom

            i = i_next
            continue

        i += 1

    return atom_char_map, bond_char_map, topology_atom_map, atom_idx


def aggregate_token_scores_to_structure(
    smi,
    token_scores,
    mode="sum",
    include_topology_on_atoms=True,
):
    """
    Aggregate the token saliency to:
        atom_scores
        bond_scores_by_key
        topology_atom_scores
    """
    atom_char_map, bond_char_map, topology_atom_map, n_atoms = (
        smiles_char_to_atom_bond_topology_map(smi)
    )

    atom_scores = np.zeros(n_atoms, dtype=float)
    atom_counts = np.zeros(n_atoms, dtype=float)

    topology_atom_scores = np.zeros(n_atoms, dtype=float)
    topology_counts = np.zeros(n_atoms, dtype=float)

    bond_scores_by_key = {}
    bond_counts_by_key = {}

    total_token_saliency = 0.0
    mapped_atom_saliency = 0.0
    mapped_bond_saliency = 0.0
    mapped_topology_saliency = 0.0
    unmapped_saliency = 0.0

    for char_pos, score in enumerate(token_scores):
        score = float(score)
        total_token_saliency += abs(score)

        if char_pos in atom_char_map:
            atom_idx = atom_char_map[char_pos]
            atom_scores[atom_idx] += score
            atom_counts[atom_idx] += 1.0
            mapped_atom_saliency += abs(score)

        elif char_pos in bond_char_map:
            bond_key = bond_char_map[char_pos]
            bond_scores_by_key[bond_key] = bond_scores_by_key.get(bond_key, 0.0) + score
            bond_counts_by_key[bond_key] = bond_counts_by_key.get(bond_key, 0.0) + 1.0
            mapped_bond_saliency += abs(score)

        elif char_pos in topology_atom_map:
            atom_idx = topology_atom_map[char_pos]
            topology_atom_scores[atom_idx] += score
            topology_counts[atom_idx] += 1.0
            mapped_topology_saliency += abs(score)

        else:
            unmapped_saliency += abs(score)

    if mode == "mean":
        atom_scores = atom_scores / np.maximum(atom_counts, 1.0)
        topology_atom_scores = topology_atom_scores / np.maximum(topology_counts, 1.0)

        for bond_key in list(bond_scores_by_key.keys()):
            bond_scores_by_key[bond_key] = (
                bond_scores_by_key[bond_key] /
                max(1.0, bond_counts_by_key[bond_key])
            )

    if include_topology_on_atoms:
        atom_scores_for_plot = atom_scores + topology_atom_scores
    else:
        atom_scores_for_plot = atom_scores.copy()

    mapped_fraction = {
        "total_abs_token_saliency": total_token_saliency,
        "atom_abs_saliency": mapped_atom_saliency,
        "bond_abs_saliency": mapped_bond_saliency,
        "topology_abs_saliency": mapped_topology_saliency,
        "unmapped_abs_saliency": unmapped_saliency,
        "mapped_fraction": (
            (mapped_atom_saliency + mapped_bond_saliency + mapped_topology_saliency) /
            max(total_token_saliency, 1e-12)
        ),
    }

    return {
        "atom_scores": atom_scores,
        "topology_atom_scores": topology_atom_scores,
        "atom_scores_for_plot": atom_scores_for_plot,
        "bond_scores_by_key": bond_scores_by_key,
        "atom_char_map": atom_char_map,
        "bond_char_map": bond_char_map,
        "topology_atom_map": topology_atom_map,
        "mapped_fraction": mapped_fraction,
    }
