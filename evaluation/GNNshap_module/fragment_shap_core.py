# -*- coding: utf-8 -*-
"""
Fragment-level Monte Carlo SHAP core.

This is true fragment-level SHAP:
    each fragment slot is treated as one Shapley player.

No config is imported here. All definitions are supplied by the root runner.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from GNNshap_module.shap_common import (
    full_mask,
    predict_with_masks,
    prepare_fragment_dict,
    fragment_names_to_node_mask,
)


def compute_tadf_fragment_shap(
    model,
    tadf_data,
    host_data,
    target_index,
    fragment_slots,
    mc_steps=64,
    random_state=42,
    mask_baseline="zero",
    add_remainder_fragment=True,
):
    """
    True TADF fragment-level Monte Carlo Shapley.

    Each non-None fragment slot is one Shapley player.
    Host/environment graph is fixed as full graph.
    """
    n_tadf = tadf_data.x.size(0)
    n_host = host_data.x.size(0)

    fragment_dict = prepare_fragment_dict(
        fragment_slots_or_dict=fragment_slots,
        num_nodes=n_tadf,
        prefix="TADF",
        add_remainder=add_remainder_fragment,
    )

    if fragment_dict is None:
        raise ValueError(
            "No valid TADF fragments were provided. "
            "Please define TADF_FRAGMENT_SLOTS in the root runner."
        )

    fragment_names = list(fragment_dict.keys())
    n_fragments = len(fragment_names)

    phi = {frag_name: 0.0 for frag_name in fragment_names}
    host_full = full_mask(n_host)
    rng = np.random.RandomState(int(random_state))

    print("\nComputing TADF fragment-level SHAP...")
    print("TADF fragments:", fragment_names)
    print("Number of fragments:", n_fragments)
    print("MC steps:", mc_steps)

    for step in range(int(mc_steps)):
        order = rng.permutation(n_fragments)
        kept_fragments = set()

        current_tadf_mask = fragment_names_to_node_mask(
            num_nodes=n_tadf,
            fragment_dict=fragment_dict,
            kept_fragments=kept_fragments,
        )

        v_prev = predict_with_masks(
            model,
            tadf_data,
            host_data,
            tadf_mask=current_tadf_mask,
            host_mask=host_full,
            target_index=target_index,
            baseline_mode=mask_baseline,
        )

        for frag_idx in order:
            frag_name = fragment_names[frag_idx]
            kept_fragments.add(frag_name)

            new_tadf_mask = fragment_names_to_node_mask(
                num_nodes=n_tadf,
                fragment_dict=fragment_dict,
                kept_fragments=kept_fragments,
            )

            v_new = predict_with_masks(
                model,
                tadf_data,
                host_data,
                tadf_mask=new_tadf_mask,
                host_mask=host_full,
                target_index=target_index,
                baseline_mode=mask_baseline,
            )

            phi[frag_name] += (v_new - v_prev)
            v_prev = v_new

        if (step + 1) % max(1, int(mc_steps) // 10) == 0:
            print("  TADF fragment SHAP progress: {}/{}".format(step + 1, mc_steps))

    for frag_name in fragment_names:
        phi[frag_name] /= float(mc_steps)

    return phi, fragment_dict


def compute_host_fragment_shap(
    model,
    tadf_data,
    host_data,
    target_index,
    fragment_slots,
    mc_steps=64,
    random_state=42,
    mask_baseline="zero",
    add_remainder_fragment=True,
):
    """
    Host/environment fragment-level Monte Carlo Shapley.
    TADF graph is fixed as full graph.
    """
    n_tadf = tadf_data.x.size(0)
    n_host = host_data.x.size(0)

    fragment_dict = prepare_fragment_dict(
        fragment_slots_or_dict=fragment_slots,
        num_nodes=n_host,
        prefix="Host/environment",
        add_remainder=add_remainder_fragment,
    )

    if fragment_dict is None:
        return None, None

    fragment_names = list(fragment_dict.keys())
    n_fragments = len(fragment_names)

    phi = {frag_name: 0.0 for frag_name in fragment_names}
    tadf_full = full_mask(n_tadf)
    rng = np.random.RandomState(int(random_state) + 17)

    print("\nComputing host/environment fragment-level SHAP...")
    print("Host fragments:", fragment_names)
    print("Number of fragments:", n_fragments)
    print("MC steps:", mc_steps)

    for step in range(int(mc_steps)):
        order = rng.permutation(n_fragments)
        kept_fragments = set()

        current_host_mask = fragment_names_to_node_mask(
            num_nodes=n_host,
            fragment_dict=fragment_dict,
            kept_fragments=kept_fragments,
        )

        v_prev = predict_with_masks(
            model,
            tadf_data,
            host_data,
            tadf_mask=tadf_full,
            host_mask=current_host_mask,
            target_index=target_index,
            baseline_mode=mask_baseline,
        )

        for frag_idx in order:
            frag_name = fragment_names[frag_idx]
            kept_fragments.add(frag_name)

            new_host_mask = fragment_names_to_node_mask(
                num_nodes=n_host,
                fragment_dict=fragment_dict,
                kept_fragments=kept_fragments,
            )

            v_new = predict_with_masks(
                model,
                tadf_data,
                host_data,
                tadf_mask=tadf_full,
                host_mask=new_host_mask,
                target_index=target_index,
                baseline_mode=mask_baseline,
            )

            phi[frag_name] += (v_new - v_prev)
            v_prev = v_new

        if (step + 1) % max(1, int(mc_steps) // 10) == 0:
            print("  Host fragment SHAP progress: {}/{}".format(step + 1, mc_steps))

    for frag_name in fragment_names:
        phi[frag_name] /= float(mc_steps)

    return phi, fragment_dict


def save_fragment_shap_csv(fragment_phi, fragment_dict, prefix, out_dir):
    """
    Save fragment-level SHAP values to CSV.
    """
    rows = []

    for frag_name, shap_value in fragment_phi.items():
        atom_indices = fragment_dict[frag_name]

        rows.append({
            "fragment": frag_name,
            "atom_indices": ",".join(map(str, atom_indices)),
            "num_atoms": len(atom_indices),
            "shap_value": float(shap_value),
            "abs_shap_value": float(abs(shap_value)),
            "shap_per_atom": float(shap_value) / max(1, len(atom_indices)),
            "abs_shap_per_atom": float(abs(shap_value)) / max(1, len(atom_indices)),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("abs_shap_value", ascending=False)

    csv_path = os.path.join(out_dir, "%s_fragment_shap.csv" % prefix)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def save_fragment_shap_bar(fragment_phi, prefix, out_dir, target_property):
    """
    Save fragment-level SHAP bar plot.
    """
    df = pd.DataFrame([
        {
            "fragment": frag_name,
            "shap_value": float(shap_value),
            "abs_shap_value": float(abs(shap_value)),
        }
        for frag_name, shap_value in fragment_phi.items()
    ])

    df = df.sort_values("shap_value", ascending=True)

    plt.figure(figsize=(7.5, max(4.5, 0.45 * len(df))))
    plt.barh(df["fragment"], df["shap_value"])
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Fragment SHAP contribution to predicted %s" % target_property)
    plt.ylabel("Fragment")
    plt.title("%s fragment-level SHAP" % prefix)
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "%s_fragment_shap_bar.png" % prefix)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    return fig_path
