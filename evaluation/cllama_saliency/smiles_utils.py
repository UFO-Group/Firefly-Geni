# -*- coding: utf-8 -*-
"""
SMILES, RDKit molecule, and randomized SMILES mapping utilities.
"""

import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def clean_smiles(smi):
    """
    Clean possible start and end tokens in a SMILES string.
    """
    smi = str(smi).strip()
    smi = smi.replace("^", "")
    smi = smi.split(">")[0]
    return smi.strip()


def is_smiles_supported_by_charset(smi, char_dict):
    """
    Check whether all characters in the SMILES are covered by the character dictionary.
    """
    input_str = "^" + smi
    target_str = smi + ">"
    unknown_chars = sorted(set(input_str + target_str) - set(char_dict.keys()))
    return len(unknown_chars) == 0, unknown_chars


def prepare_mol_for_drawing(smi):
    """
    Prepare an RDKit molecule and compute 2D coordinates.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smi}")

    mol = Chem.Mol(mol)

    try:
        AllChem.Compute2DCoords(mol)
    except Exception:
        pass

    return mol


def check_smiles_and_charset(smi, char_dict):
    """
    Check RDKit validity and character-set coverage.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smi}")

    ok, unknown_chars = is_smiles_supported_by_charset(smi, char_dict)

    if not ok:
        raise ValueError(
            f"The SMILES contains characters not included in the character set: {unknown_chars}\n"
            f"Please check whether module.char.charset_list contains these characters.\n"
            f"SMILES: {smi}"
        )

    return mol


def generate_randomized_smiles_list(
    input_smi,
    n_enum=10,
    seed=42,
    char_dict=None,
    max_trials=500,
):
    """
    Generate randomized SMILES strings.
    """
    input_smi = clean_smiles(input_smi)
    mol = Chem.MolFromSmiles(input_smi)

    if mol is None:
        raise ValueError(f"RDKit failed to parse input SMILES: {input_smi}")

    random.seed(seed)
    np.random.seed(seed)

    smiles_list = []
    seen = set()

    ok, unknown = is_smiles_supported_by_charset(input_smi, char_dict)
    if ok:
        smiles_list.append(input_smi)
        seen.add(input_smi)
    else:
        print(f"Warning: the original SMILES contains unknown characters and will be skipped: {unknown}")

    trial = 0

    while len(smiles_list) < n_enum and trial < max_trials:
        trial += 1

        random_smi = Chem.MolToSmiles(
            mol,
            canonical=False,
            doRandom=True,
            isomericSmiles=True,
        )

        random_smi = clean_smiles(random_smi)

        if random_smi in seen:
            continue

        mol_check = Chem.MolFromSmiles(random_smi)
        if mol_check is None:
            continue

        ok, unknown = is_smiles_supported_by_charset(random_smi, char_dict)
        if not ok:
            print(f"Skipping randomized SMILES unsupported by the character set: {random_smi}, unknown={unknown}")
            continue

        smiles_list.append(random_smi)
        seen.add(random_smi)

    if len(smiles_list) < n_enum:
        print(
            f"Warning: only {len(smiles_list)} unique randomized SMILES supported by the character set were generated. "
            f"Existing SMILES will be repeated to reach {n_enum} entries."
        )
        while len(smiles_list) < n_enum:
            smiles_list.append(smiles_list[len(smiles_list) % max(1, len(smiles_list))])

    return smiles_list[:n_enum]


def map_random_atom_scores_to_reference(reference_mol, random_smi, random_atom_scores):
    """
    Map atom scores from a randomized SMILES representation back to the atom indices
    of the reference molecule.
    """
    random_mol = Chem.MolFromSmiles(random_smi)

    if random_mol is None:
        raise ValueError(f"RDKit failed to parse randomized SMILES: {random_smi}")

    n_ref = reference_mol.GetNumAtoms()
    n_rand = random_mol.GetNumAtoms()

    if len(random_atom_scores) != n_rand:
        print("Warning: the length of random_atom_scores does not match the number of atoms in random_mol.")
        print(f"len(random_atom_scores) = {len(random_atom_scores)}, random_mol atoms = {n_rand}")

    match = reference_mol.GetSubstructMatch(random_mol)

    if len(match) != n_rand:
        match2 = random_mol.GetSubstructMatch(reference_mol)
        if len(match2) == n_ref:
            ref_scores = np.zeros(n_ref, dtype=float)
            for ref_idx, rand_idx in enumerate(match2):
                if rand_idx < len(random_atom_scores):
                    ref_scores[ref_idx] = float(random_atom_scores[rand_idx])
            return ref_scores, "reverse_match"

        raise ValueError(
            "Failed to map randomized SMILES atoms back to the reference molecule.\n"
            f"random_smi = {random_smi}\n"
            f"reference atoms = {n_ref}, random atoms = {n_rand}, match len = {len(match)}"
        )

    ref_scores = np.zeros(n_ref, dtype=float)

    for rand_idx, ref_idx in enumerate(match):
        if rand_idx < len(random_atom_scores):
            ref_scores[int(ref_idx)] += float(random_atom_scores[rand_idx])

    return ref_scores, "forward_match"


def map_random_bond_scores_to_reference(reference_mol, random_smi, random_bond_scores_by_key):
    """
    Map bond saliency scores from a randomized SMILES representation back to the
    bonds of the reference molecule.
    """
    random_mol = Chem.MolFromSmiles(random_smi)

    if random_mol is None:
        raise ValueError(f"RDKit failed to parse randomized SMILES: {random_smi}")

    n_ref = reference_mol.GetNumAtoms()
    n_rand = random_mol.GetNumAtoms()

    match = reference_mol.GetSubstructMatch(random_mol)

    if len(match) != n_rand:
        match2 = random_mol.GetSubstructMatch(reference_mol)

        if len(match2) == n_ref:
            random_to_ref = {
                int(rand_idx): int(ref_idx)
                for ref_idx, rand_idx in enumerate(match2)
            }
            map_type = "reverse_match"
        else:
            raise ValueError(
                "Failed to map randomized SMILES bond saliency back to the reference molecule.\n"
                f"random_smi = {random_smi}"
            )
    else:
        random_to_ref = {
            int(rand_idx): int(ref_idx)
            for rand_idx, ref_idx in enumerate(match)
        }
        map_type = "forward_match"

    ref_bond_scores_by_key = {}

    for random_bond_key, score in random_bond_scores_by_key.items():
        ra, rb = int(random_bond_key[0]), int(random_bond_key[1])

        if ra not in random_to_ref or rb not in random_to_ref:
            continue

        ref_a = random_to_ref[ra]
        ref_b = random_to_ref[rb]
        ref_key = tuple(sorted((ref_a, ref_b)))

        if reference_mol.GetBondBetweenAtoms(int(ref_key[0]), int(ref_key[1])) is None:
            continue

        ref_bond_scores_by_key[ref_key] = ref_bond_scores_by_key.get(ref_key, 0.0) + float(score)

    return ref_bond_scores_by_key, map_type


def get_reference_bond_keys(mol):
    """
    Return all bond atom-pair keys in the reference molecule.
    """
    bond_keys = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        bond_keys.append(tuple(sorted((int(a), int(b)))))
    return bond_keys


def bond_scores_dict_to_vector(bond_scores_by_key, reference_bond_keys):
    """
    Convert a bond score dictionary into a vector with a fixed bond order.
    """
    return np.array(
        [float(bond_scores_by_key.get(key, 0.0)) for key in reference_bond_keys],
        dtype=float,
    )


def bond_vector_to_scores_dict(bond_vector, reference_bond_keys):
    """
    Convert a fixed-order bond score vector back into a dictionary.
    """
    out = {}
    for key, value in zip(reference_bond_keys, bond_vector):
        if np.isfinite(value) and abs(float(value)) > 1e-12:
            out[key] = float(value)
    return out
