# -*- coding: utf-8 -*-
"""
Atom-level Monte Carlo SHAP core.

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
    empty_mask,
    predict_with_masks,
)


def compute_tadf_atom_shap(
    model,
    tadf_data,
    host_data,
    target_index,
    mc_steps=64,
    random_state=42,
    mask_baseline="zero",
):
    """
    TADF atom SHAP with host fixed as full graph.
    Each TADF atom is treated as one Shapley player.
    """
    n_tadf = tadf_data.x.size(0)
    n_host = host_data.x.size(0)

    phi = np.zeros(n_tadf, dtype=np.float64)
    host_full = full_mask(n_host)
    rng = np.random.RandomState(int(random_state))

    print("\nComputing TADF atom-level SHAP...")
    print("TADF atoms:", n_tadf)
    print("MC steps:", mc_steps)

    for step in range(int(mc_steps)):
        order = rng.permutation(n_tadf)
        current_mask = empty_mask(n_tadf)

        v_prev = predict_with_masks(
            model,
            tadf_data,
            host_data,
            tadf_mask=current_mask,
            host_mask=host_full,
            target_index=target_index,
            baseline_mode=mask_baseline,
        )

        for atom_idx in order:
            current_mask[atom_idx] = True

            v_new = predict_with_masks(
                model,
                tadf_data,
                host_data,
                tadf_mask=current_mask,
                host_mask=host_full,
                target_index=target_index,
                baseline_mode=mask_baseline,
            )

            phi[atom_idx] += (v_new - v_prev)
            v_prev = v_new

        if (step + 1) % max(1, int(mc_steps) // 10) == 0:
            print("  TADF atom SHAP progress: {}/{}".format(step + 1, mc_steps))

    phi = phi / float(mc_steps)
    return phi


def compute_host_atom_shap(
    model,
    tadf_data,
    host_data,
    target_index,
    mc_steps=64,
    random_state=42,
    mask_baseline="zero",
):
    """
    Host/environment atom SHAP with TADF fixed as full graph.
    """
    n_tadf = tadf_data.x.size(0)
    n_host = host_data.x.size(0)

    phi = np.zeros(n_host, dtype=np.float64)
    tadf_full = full_mask(n_tadf)
    rng = np.random.RandomState(int(random_state) + 17)

    print("\nComputing host/environment atom-level SHAP...")
    print("Host/environment atoms:", n_host)
    print("MC steps:", mc_steps)

    for step in range(int(mc_steps)):
        order = rng.permutation(n_host)
        current_mask = empty_mask(n_host)

        v_prev = predict_with_masks(
            model,
            tadf_data,
            host_data,
            tadf_mask=tadf_full,
            host_mask=current_mask,
            target_index=target_index,
            baseline_mode=mask_baseline,
        )

        for atom_idx in order:
            current_mask[atom_idx] = True

            v_new = predict_with_masks(
                model,
                tadf_data,
                host_data,
                tadf_mask=tadf_full,
                host_mask=current_mask,
                target_index=target_index,
                baseline_mode=mask_baseline,
            )

            phi[atom_idx] += (v_new - v_prev)
            v_prev = v_new

        if (step + 1) % max(1, int(mc_steps) // 10) == 0:
            print("  Host atom SHAP progress: {}/{}".format(step + 1, mc_steps))

    phi = phi / float(mc_steps)
    return phi


def save_atom_shap_csv(phi, prefix, out_dir):
    df = pd.DataFrame({
        "atom_idx": np.arange(len(phi), dtype=int),
        "shap_value": phi,
        "abs_shap_value": np.abs(phi),
    }).sort_values("abs_shap_value", ascending=False)

    csv_path = os.path.join(out_dir, "%s_atom_shap.csv" % prefix)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path


def save_atom_shap_bar(phi, prefix, out_dir, target_property, topn=30):
    df = pd.DataFrame({
        "atom_idx": np.arange(len(phi), dtype=int),
        "shap_value": phi,
        "abs_shap_value": np.abs(phi),
    }).sort_values("abs_shap_value", ascending=False)

    df_top = df.head(topn).iloc[::-1]

    labels = ["atom_%d" % i for i in df_top["atom_idx"].values]
    values = df_top["shap_value"].values

    plt.figure(figsize=(7.5, max(5, 0.28 * len(df_top))))
    plt.barh(labels, values)
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Atom SHAP contribution to predicted %s" % target_property)
    plt.ylabel("Atom index")
    plt.title("%s atom-level SHAP top %d" % (prefix, len(df_top)))
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "%s_atom_shap_bar.png" % prefix)
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    return fig_path
