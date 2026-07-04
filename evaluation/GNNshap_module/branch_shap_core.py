# -*- coding: utf-8 -*-
"""
Branch-level SHAP core.

No config is imported here. All definitions are supplied by the root runner.
"""

import os
import json
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from GNNshap_module.shap_common import (
    full_mask,
    empty_mask,
    predict_with_masks,
)


def compute_branch_shap(
    model,
    tadf_data,
    host_data,
    target_index,
    target_property="Delta_EST_eV",
    mask_baseline="zero",
):
    """
    Two-player Shapley decomposition:
        player 1 = TADF branch
        player 2 = host/environment branch
    """
    n_tadf = tadf_data.x.size(0)
    n_host = host_data.x.size(0)

    t_empty = empty_mask(n_tadf)
    t_full = full_mask(n_tadf)
    h_empty = empty_mask(n_host)
    h_full = full_mask(n_host)

    v_empty = predict_with_masks(
        model, tadf_data, host_data,
        tadf_mask=t_empty,
        host_mask=h_empty,
        target_index=target_index,
        baseline_mode=mask_baseline,
    )

    v_tadf = predict_with_masks(
        model, tadf_data, host_data,
        tadf_mask=t_full,
        host_mask=h_empty,
        target_index=target_index,
        baseline_mode=mask_baseline,
    )

    v_host = predict_with_masks(
        model, tadf_data, host_data,
        tadf_mask=t_empty,
        host_mask=h_full,
        target_index=target_index,
        baseline_mode=mask_baseline,
    )

    v_both = predict_with_masks(
        model, tadf_data, host_data,
        tadf_mask=t_full,
        host_mask=h_full,
        target_index=target_index,
        baseline_mode=mask_baseline,
    )

    phi_tadf = 0.5 * ((v_tadf - v_empty) + (v_both - v_host))
    phi_host = 0.5 * ((v_host - v_empty) + (v_both - v_tadf))
    interaction = v_both - v_tadf - v_host + v_empty

    result = {
        "target_property": target_property,
        "v_empty": v_empty,
        "v_tadf_only": v_tadf,
        "v_host_only": v_host,
        "v_both_full": v_both,
        "phi_tadf_branch": phi_tadf,
        "phi_host_branch": phi_host,
        "interaction_tadf_host": interaction,
        "branch_additivity_check_phi_sum": phi_tadf + phi_host,
        "branch_additivity_check_v_both_minus_v_empty": v_both - v_empty,
        "branch_additivity_error": (phi_tadf + phi_host) - (v_both - v_empty),
    }

    return result


def save_branch_shap(branch_result, out_dir, target_property):
    df = pd.DataFrame([
        {
            "component": "TADF_branch",
            "value_type": "shap_value",
            "value": branch_result["phi_tadf_branch"],
        },
        {
            "component": "Host_or_environment_branch",
            "value_type": "shap_value",
            "value": branch_result["phi_host_branch"],
        },
        {
            "component": "TADF_Host_nonadditive_interaction",
            "value_type": "interaction",
            "value": branch_result["interaction_tadf_host"],
        },
    ])

    csv_path = os.path.join(out_dir, "branch_shap.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    json_path = os.path.join(out_dir, "branch_values.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(branch_result, f, indent=2, ensure_ascii=False)

    plt.figure(figsize=(6.5, 4.5))
    plt.bar(df["component"], df["value"])
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.ylabel("Contribution to predicted %s" % target_property)
    plt.xticks(rotation=25, ha="right")
    plt.title("Branch-level SHAP and interaction")
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "branch_shap_bar.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    return csv_path, json_path, fig_path
