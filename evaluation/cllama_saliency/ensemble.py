# -*- coding: utf-8 -*-
"""
Main workflow for randomized SMILES ensemble saliency.
"""

import os
import numpy as np
import pandas as pd
from rdkit import Chem

from .drawing import draw_plain_structure_png, draw_atom_bond_score_red_png
from .saliency import compute_lowEST_scores_for_one_smiles
from .smiles_utils import (
    clean_smiles,
    generate_randomized_smiles_list,
    map_random_atom_scores_to_reference,
    map_random_bond_scores_to_reference,
    get_reference_bond_keys,
    bond_scores_dict_to_vector,
    bond_vector_to_scores_dict,
)


def explain_smiles_ensemble_lowEST_saliency(
    model,
    input_smi,
    char_dict,
    est_low,
    est_high,
    sa_value,
    est_min,
    est_max,
    sa_min,
    sa_max,
    device,
    output_dir,
    n_enum_smiles=10,
    atom_agg_mode="sum",
    include_topology_on_atoms=True,
    seed=42,
    max_random_smiles_trials=1000,
    draw_plain_add_atom_indices=True,
    plain_width=1300,
    plain_height=850,
    plain_legend_font_size=28,
    plain_annotation_font_scale=0.70,
    draw_enum_annotate_scores=True,
    draw_enum_add_atom_indices=True,
    draw_enum_scale_percentile=95,
    draw_enum_gamma=0.65,
    draw_ensemble_annotate_scores=True,
    draw_ensemble_add_atom_indices=True,
    draw_ensemble_scale_percentile=95,
    draw_ensemble_gamma=0.65,
    saliency_map_width=1500,
    saliency_map_height=1000,
    saliency_legend_font_size=26,
    saliency_annotation_font_scale=0.65,
    saliency_bond_line_width=2.2,
    atom_radius_base=0.25,
    atom_radius_scale=0.32,
):
    """
    Main function:
    1. Generate multiple randomized SMILES.
    2. Compute lowEST saliency for each SMILES separately.
    3. Save images for each SMILES.
    4. Map each result back to the reference molecule.
    5. Calculate the mean and standard deviation.
    6. Save the averaged image.
    """
    os.makedirs(output_dir, exist_ok=True)

    input_smi = clean_smiles(input_smi)
    reference_mol = Chem.MolFromSmiles(input_smi)

    if reference_mol is None:
        raise ValueError(f"RDKit failed to parse reference SMILES: {input_smi}")

    reference_smi = Chem.MolToSmiles(reference_mol, canonical=True)
    reference_mol = Chem.MolFromSmiles(reference_smi)

    reference_bond_keys = get_reference_bond_keys(reference_mol)

    print("\n" + "=" * 100)
    print("Starting randomized SMILES ensemble Low-EST-specific generation saliency analysis")
    print("=" * 100)
    print(f"Input SMILES:\n{input_smi}")
    print(f"Reference canonical SMILES:\n{reference_smi}")
    print(f"Reference atoms: {reference_mol.GetNumAtoms()}")
    print(f"Reference bonds: {len(reference_bond_keys)}")
    print(f"N_ENUM_SMILES = {n_enum_smiles}")
    print(f"MAX_RANDOM_SMILES_TRIALS = {max_random_smiles_trials}")
    print(f"EST_LOW  = {est_low}")
    print(f"EST_HIGH = {est_high}")
    print(f"SA_FIXED = {sa_value}")
    print(f"ATOM_AGG_MODE = {atom_agg_mode}")
    print("=" * 100)

    draw_plain_structure_png(
        smi=reference_smi,
        output_png=os.path.join(output_dir, "reference_plain_structure.png"),
        title="Reference canonical structure with atom indices",
        add_atom_indices=draw_plain_add_atom_indices,
        width=plain_width,
        height=plain_height,
        legend_font_size=plain_legend_font_size,
        annotation_font_scale=plain_annotation_font_scale,
    )

    enum_smiles_list = generate_randomized_smiles_list(
        input_smi=input_smi,
        n_enum=n_enum_smiles,
        seed=seed,
        char_dict=char_dict,
        max_trials=max_random_smiles_trials,
    )

    enum_rows = []
    for idx, smi in enumerate(enum_smiles_list, start=1):
        enum_rows.append({"enum_id": idx, "smiles": smi, "length": len(smi)})

    enum_df = pd.DataFrame(enum_rows)
    enum_csv = os.path.join(output_dir, "ensemble_randomized_smiles_list.csv")
    enum_df.to_csv(enum_csv, index=False, encoding="utf-8-sig")
    print(f"Saved randomized SMILES list: {enum_csv}")

    mapped_structure_atom_scores = []
    mapped_bond_vectors = []

    all_token_dfs = []
    all_atom_dfs = []
    all_bond_dfs = []

    mapped_atom_rows = []
    mapped_bond_rows = []
    mapped_fraction_rows = []

    for enum_id, smi in enumerate(enum_smiles_list, start=1):
        print("\n" + "-" * 100)
        print(f"Starting calculation for randomized SMILES {enum_id}/{n_enum_smiles}")
        print(smi)
        print("-" * 100)

        enum_dir = os.path.join(output_dir, f"enum_{enum_id:02d}")
        os.makedirs(enum_dir, exist_ok=True)

        result = compute_lowEST_scores_for_one_smiles(
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
            atom_agg_mode=atom_agg_mode,
            include_topology_on_atoms=include_topology_on_atoms,
        )

        token_df = result["token_df"].copy()
        atom_df = result["atom_df"].copy()
        bond_df = result["bond_df"].copy()

        token_df.insert(0, "enum_id", enum_id)
        atom_df.insert(0, "enum_id", enum_id)
        bond_df.insert(0, "enum_id", enum_id)

        all_token_dfs.append(token_df)
        all_atom_dfs.append(atom_df)
        all_bond_dfs.append(bond_df)

        token_df.to_csv(os.path.join(enum_dir, f"enum_{enum_id:02d}_token_scores.csv"), index=False, encoding="utf-8-sig")
        atom_df.to_csv(os.path.join(enum_dir, f"enum_{enum_id:02d}_atom_scores_this_smiles.csv"), index=False, encoding="utf-8-sig")
        bond_df.to_csv(os.path.join(enum_dir, f"enum_{enum_id:02d}_bond_scores_this_smiles.csv"), index=False, encoding="utf-8-sig")

        frac_info = result["lowEST_structure"]["mapped_fraction"]
        mapped_fraction_rows.append({
            "enum_id": enum_id,
            "enum_smiles": smi,
            "method": "lowEST_specific_generation_saliency",
            **frac_info,
        })

        draw_plain_structure_png(
            smi=smi,
            output_png=os.path.join(enum_dir, f"enum_{enum_id:02d}_plain_structure.png"),
            title=f"Enum {enum_id:02d} randomized SMILES structure",
            add_atom_indices=draw_plain_add_atom_indices,
            width=plain_width,
            height=plain_height,
            legend_font_size=plain_legend_font_size,
            annotation_font_scale=plain_annotation_font_scale,
        )

        draw_atom_bond_score_red_png(
            smi=smi,
            atom_scores=result["lowEST_structure"]["atom_scores_for_plot"],
            bond_scores_by_key=result["lowEST_structure"]["bond_scores_by_key"],
            output_png=os.path.join(enum_dir, f"enum_{enum_id:02d}_lowEST_atom_bond_topology_saliency.png"),
            title=f"Enum {enum_id:02d}: Low-EST atom/bond/topology saliency",
            annotate_scores=draw_enum_annotate_scores,
            add_atom_indices=draw_enum_add_atom_indices,
            scale_percentile=draw_enum_scale_percentile,
            gamma=draw_enum_gamma,
            width=saliency_map_width,
            height=saliency_map_height,
            legend_font_size=saliency_legend_font_size,
            annotation_font_scale=saliency_annotation_font_scale,
            bond_line_width=saliency_bond_line_width,
            atom_radius_base=atom_radius_base,
            atom_radius_scale=atom_radius_scale,
        )

        mapped_struct_atom, map_type_atom = map_random_atom_scores_to_reference(
            reference_mol=reference_mol,
            random_smi=smi,
            random_atom_scores=result["lowEST_structure"]["atom_scores_for_plot"],
        )
        mapped_structure_atom_scores.append(mapped_struct_atom)

        mapped_bond_dict, map_type_bond = map_random_bond_scores_to_reference(
            reference_mol=reference_mol,
            random_smi=smi,
            random_bond_scores_by_key=result["lowEST_structure"]["bond_scores_by_key"],
        )
        mapped_bond_vectors.append(bond_scores_dict_to_vector(mapped_bond_dict, reference_bond_keys))

        ref_atom_symbols = [atom.GetSymbol() for atom in reference_mol.GetAtoms()]
        for atom_idx in range(reference_mol.GetNumAtoms()):
            mapped_atom_rows.append({
                "enum_id": enum_id,
                "enum_smiles": smi,
                "reference_smiles": reference_smi,
                "reference_atom_index": atom_idx,
                "reference_atom_symbol": ref_atom_symbols[atom_idx],
                "lowEST_atom_topology_saliency_mapped": mapped_struct_atom[atom_idx],
                "map_type_lowEST_structure_atom": map_type_atom,
            })

        for bond_key in reference_bond_keys:
            mapped_bond_rows.append({
                "enum_id": enum_id,
                "enum_smiles": smi,
                "reference_smiles": reference_smi,
                "reference_bond_atom_pair": f"{bond_key[0]}-{bond_key[1]}",
                "begin_atom_index": bond_key[0],
                "end_atom_index": bond_key[1],
                "lowEST_bond_saliency_mapped": mapped_bond_dict.get(bond_key, 0.0),
                "map_type_lowEST_bond": map_type_bond,
            })

        print(f"Completed randomized SMILES {enum_id}/{n_enum_smiles}")

    all_token_df = pd.concat(all_token_dfs, axis=0, ignore_index=True)
    all_atom_df = pd.concat(all_atom_dfs, axis=0, ignore_index=True)
    all_bond_df = pd.concat(all_bond_dfs, axis=0, ignore_index=True)

    all_token_csv = os.path.join(output_dir, "ensemble_all_token_scores.csv")
    all_atom_csv = os.path.join(output_dir, "ensemble_all_atom_scores_this_smiles.csv")
    all_bond_csv = os.path.join(output_dir, "ensemble_all_bond_scores_this_smiles.csv")

    all_token_df.to_csv(all_token_csv, index=False, encoding="utf-8-sig")
    all_atom_df.to_csv(all_atom_csv, index=False, encoding="utf-8-sig")
    all_bond_df.to_csv(all_bond_csv, index=False, encoding="utf-8-sig")

    print(f"\nSaved all token scores: {all_token_csv}")
    print(f"Saved all atom scores: {all_atom_csv}")
    print(f"Saved all bond scores: {all_bond_csv}")

    mapped_atom_df = pd.DataFrame(mapped_atom_rows)
    mapped_atom_csv = os.path.join(output_dir, "ensemble_mapped_atom_scores_each_enum.csv")
    mapped_atom_df.to_csv(mapped_atom_csv, index=False, encoding="utf-8-sig")
    print(f"Saved atom scores mapped back to the reference for each enumeration: {mapped_atom_csv}")

    mapped_bond_df = pd.DataFrame(mapped_bond_rows)
    mapped_bond_csv = os.path.join(output_dir, "ensemble_mapped_bond_scores_each_enum.csv")
    mapped_bond_df.to_csv(mapped_bond_csv, index=False, encoding="utf-8-sig")
    print(f"Saved bond scores mapped back to the reference for each enumeration: {mapped_bond_csv}")

    mapped_fraction_df = pd.DataFrame(mapped_fraction_rows)
    mapped_fraction_csv = os.path.join(output_dir, "ensemble_token_mapping_fraction.csv")
    mapped_fraction_df.to_csv(mapped_fraction_csv, index=False, encoding="utf-8-sig")
    print(f"Saved token saliency mapping coverage: {mapped_fraction_csv}")

    mapped_structure_atom_scores = np.vstack(mapped_structure_atom_scores)
    structure_atom_mean = np.nanmean(mapped_structure_atom_scores, axis=0)
    structure_atom_std = np.nanstd(mapped_structure_atom_scores, axis=0)

    mapped_bond_vectors = np.vstack(mapped_bond_vectors)
    bond_mean_vec = np.nanmean(mapped_bond_vectors, axis=0)
    bond_std_vec = np.nanstd(mapped_bond_vectors, axis=0)

    bond_mean_dict = bond_vector_to_scores_dict(bond_mean_vec, reference_bond_keys)

    ref_atom_symbols = [atom.GetSymbol() for atom in reference_mol.GetAtoms()]
    summary_rows = []
    for atom_idx in range(reference_mol.GetNumAtoms()):
        summary_rows.append({
            "reference_smiles": reference_smi,
            "reference_atom_index": atom_idx,
            "reference_atom_symbol": ref_atom_symbols[atom_idx],
            "lowEST_atom_topology_saliency_mean": structure_atom_mean[atom_idx],
            "lowEST_atom_topology_saliency_std": structure_atom_std[atom_idx],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = os.path.join(output_dir, "ensemble_atom_scores_summary_mean_std.csv")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    print(f"Saved ensemble atom mean and standard deviation: {summary_csv}")

    bond_summary_rows = []
    for bond_idx, bond_key in enumerate(reference_bond_keys):
        bond = reference_mol.GetBondBetweenAtoms(int(bond_key[0]), int(bond_key[1]))
        bond_summary_rows.append({
            "reference_smiles": reference_smi,
            "reference_bond_atom_pair": f"{bond_key[0]}-{bond_key[1]}",
            "begin_atom_index": bond_key[0],
            "end_atom_index": bond_key[1],
            "bond_type": str(bond.GetBondType()) if bond is not None else "None",
            "lowEST_bond_saliency_mean": bond_mean_vec[bond_idx],
            "lowEST_bond_saliency_std": bond_std_vec[bond_idx],
        })

    bond_summary_df = pd.DataFrame(bond_summary_rows)
    bond_summary_csv = os.path.join(output_dir, "ensemble_bond_scores_summary_mean_std.csv")
    bond_summary_df.to_csv(bond_summary_csv, index=False, encoding="utf-8-sig")
    print(f"Saved ensemble bond mean and standard deviation: {bond_summary_csv}")

    draw_atom_bond_score_red_png(
        smi=reference_smi,
        atom_scores=structure_atom_mean,
        bond_scores_by_key=bond_mean_dict,
        output_png=os.path.join(output_dir, "ensemble_mean_lowEST_atom_bond_topology_saliency.png"),
        title=f"Ensemble mean low-EST atom/bond/topology saliency, n={n_enum_smiles}",
        annotate_scores=draw_ensemble_annotate_scores,
        add_atom_indices=draw_ensemble_add_atom_indices,
        scale_percentile=draw_ensemble_scale_percentile,
        gamma=draw_ensemble_gamma,
        width=saliency_map_width,
        height=saliency_map_height,
        legend_font_size=saliency_legend_font_size,
        annotation_font_scale=saliency_annotation_font_scale,
        bond_line_width=saliency_bond_line_width,
        atom_radius_base=atom_radius_base,
        atom_radius_scale=atom_radius_scale,
    )

    print("\nTop atoms in ensemble mean low-EST atom+bond+topology saliency:")
    print(
        summary_df.sort_values("lowEST_atom_topology_saliency_mean", ascending=False)
        .head(10)[[
            "reference_atom_index",
            "reference_atom_symbol",
            "lowEST_atom_topology_saliency_mean",
            "lowEST_atom_topology_saliency_std",
        ]]
        .to_string(index=False)
    )

    print("\nTop bonds in ensemble mean low-EST bond saliency:")
    print(
        bond_summary_df.sort_values("lowEST_bond_saliency_mean", ascending=False)
        .head(10)[[
            "reference_bond_atom_pair",
            "bond_type",
            "lowEST_bond_saliency_mean",
            "lowEST_bond_saliency_std",
        ]]
        .to_string(index=False)
    )

    summary_txt = os.path.join(output_dir, "ensemble_summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write("Randomized SMILES ensemble low-EST saliency summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"input_smi: {input_smi}\n")
        f.write(f"reference_smi: {reference_smi}\n")
        f.write(f"n_enum_smiles: {n_enum_smiles}\n")
        f.write(f"EST_LOW: {est_low}\n")
        f.write(f"EST_HIGH: {est_high}\n")
        f.write(f"SA_FIXED: {sa_value}\n")
        f.write(f"ATOM_AGG_MODE: {atom_agg_mode}\n")
        f.write(f"INCLUDE_TOPOLOGY_ON_ATOMS: {include_topology_on_atoms}\n")
        f.write("\nMain figure:\n")
        f.write("  ensemble_mean_lowEST_atom_bond_topology_saliency.png\n")
        f.write("\nEach randomized SMILES result is saved in enum_XX folders.\n")
        f.write("\nCSV outputs:\n")
        f.write("  ensemble_all_token_scores.csv\n")
        f.write("  ensemble_all_atom_scores_this_smiles.csv\n")
        f.write("  ensemble_all_bond_scores_this_smiles.csv\n")
        f.write("  ensemble_mapped_atom_scores_each_enum.csv\n")
        f.write("  ensemble_mapped_bond_scores_each_enum.csv\n")
        f.write("  ensemble_token_mapping_fraction.csv\n")
        f.write("  ensemble_atom_scores_summary_mean_std.csv\n")
        f.write("  ensemble_bond_scores_summary_mean_std.csv\n")

    print(f"\nSaved summary: {summary_txt}")
    print("\nAnalysis completed.")
    print(f"All results have been saved to: {output_dir}")

    return {
        "enum_smiles_df": enum_df,
        "mapped_atom_df": mapped_atom_df,
        "mapped_bond_df": mapped_bond_df,
        "mapped_fraction_df": mapped_fraction_df,
        "summary_df": summary_df,
        "bond_summary_df": bond_summary_df,
        "structure_atom_mean": structure_atom_mean,
        "bond_mean_dict": bond_mean_dict,
        "reference_smi": reference_smi,
    }
