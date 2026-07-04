# -*- coding: utf-8 -*-
"""
RDKit drawing utilities.

No user-editable drawing parameters are stored here. Pass drawing settings
from the top-level run_cllama_saliency.py file.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from .smiles_utils import prepare_mol_for_drawing


def draw_plain_structure_png(
    smi,
    output_png,
    title="Plain molecular structure with atom indices",
    add_atom_indices=True,
    width=1300,
    height=850,
    legend_font_size=28,
    annotation_font_scale=0.70,
):
    """
    Draw plain molecular structure PNG.
    """
    mol = prepare_mol_for_drawing(smi)

    drawer = rdMolDraw2D.MolDraw2DCairo(int(width), int(height))
    opts = drawer.drawOptions()
    opts.addAtomIndices = bool(add_atom_indices)
    opts.legendFontSize = int(legend_font_size)
    opts.annotationFontScale = float(annotation_font_scale)

    drawer.DrawMolecule(mol, legend=title)
    drawer.FinishDrawing()

    with open(output_png, "wb") as f:
        f.write(drawer.GetDrawingText())

    print(f"Saved plain structure PNG: {output_png}")


def score_to_red(score, max_score, gamma=0.65):
    """
    Single-channel red saliency color map.
    """
    if max_score < 1e-12:
        return (1.0, 0.94, 0.94)

    v = float(score) / max_score
    v = max(0.0, min(1.0, v))
    v = v ** gamma

    r = 1.0
    g = 0.94 - 0.86 * v
    b = 0.94 - 0.86 * v

    g = max(0.08, g)
    b = max(0.08, b)

    return (r, g, b)


def draw_atom_bond_score_red_png(
    smi,
    atom_scores,
    bond_scores_by_key,
    output_png,
    title,
    annotate_scores=True,
    add_atom_indices=True,
    scale_percentile=95,
    gamma=0.65,
    width=1500,
    height=1000,
    legend_font_size=26,
    annotation_font_scale=0.65,
    bond_line_width=2.2,
    atom_radius_base=0.25,
    atom_radius_scale=0.32,
):
    """
    Draw atom + bond + topology saliency map.
    """
    mol = prepare_mol_for_drawing(smi)

    if mol.GetNumAtoms() != len(atom_scores):
        print("Warning: RDKit atom count does not match the length of atom_scores.")
        print(f"RDKit atoms = {mol.GetNumAtoms()}, atom_scores = {len(atom_scores)}")
        print("Drawing will still proceed using the shorter length.")

    mol = Chem.Mol(mol)
    n_atoms = min(mol.GetNumAtoms(), len(atom_scores))

    all_positive_scores = []

    for i in range(n_atoms):
        if atom_scores[i] > 1e-12:
            all_positive_scores.append(float(atom_scores[i]))

    for bond_key, score in bond_scores_by_key.items():
        if score > 1e-12:
            all_positive_scores.append(float(score))

    if len(all_positive_scores) == 0:
        max_score = 1.0
    else:
        max_score = np.percentile(np.array(all_positive_scores), scale_percentile)

    if max_score < 1e-12:
        max_score = 1.0

    atom_colors = {}
    atom_radii = {}

    for i in range(n_atoms):
        score = float(atom_scores[i])
        atom_colors[i] = score_to_red(score=score, max_score=max_score, gamma=gamma)

        v = min(1.0, max(0.0, score / max_score))
        v = v ** gamma
        atom_radii[i] = float(atom_radius_base) + float(atom_radius_scale) * v

        if annotate_scores:
            mol.GetAtomWithIdx(i).SetProp("atomNote", f"{score:.2f}")

    highlight_bonds = []
    bond_colors = {}

    for bond_key, score in bond_scores_by_key.items():
        a, b = int(bond_key[0]), int(bond_key[1])
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is None:
            continue

        bond_idx = bond.GetIdx()
        highlight_bonds.append(bond_idx)
        bond_colors[bond_idx] = score_to_red(score=float(score), max_score=max_score, gamma=gamma)

    drawer = rdMolDraw2D.MolDraw2DCairo(int(width), int(height))
    opts = drawer.drawOptions()
    opts.addAtomIndices = bool(add_atom_indices)
    opts.legendFontSize = int(legend_font_size)
    opts.annotationFontScale = float(annotation_font_scale)
    opts.bondLineWidth = float(bond_line_width)

    legend_text = (
        f"{title}\n"
        f"Atom color = atom/topology-token saliency; bond color = bond/ring-token saliency"
    )

    drawer.DrawMolecule(
        mol,
        highlightAtoms=list(atom_colors.keys()),
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
        legend=legend_text,
    )

    drawer.FinishDrawing()

    with open(output_png, "wb") as f:
        f.write(drawer.GetDrawingText())

    print(f"Saved atom+bond+topology saliency map: {output_png}")
