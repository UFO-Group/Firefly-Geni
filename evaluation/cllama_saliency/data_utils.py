# -*- coding: utf-8 -*-
"""
Data loading and property normalization utilities.
"""

import pandas as pd
import numpy as np


def normalize_value(x, xmin, xmax):
    """
    Normalize a real physical value to [0, 1].
    """
    return max(0.0, min(1.0, (float(x) - xmin) / (xmax - xmin)))


def load_property_ranges(full_csv_path, sa_fixed=None):
    """
    Read the global ranges of Delta_EST and SA score.

    Returns:
        est_min, est_max, sa_min, sa_max, sa_value
    """
    print(f"\nReading global property distribution: {full_csv_path}")
    df_full = pd.read_csv(full_csv_path)

    est_real_dist = df_full["Delta_EST_eV"].dropna().values
    sa_real_dist = df_full["sa_score"].dropna().values

    est_min, est_max = float(est_real_dist.min()), float(est_real_dist.max())
    sa_min, sa_max = float(sa_real_dist.min()), float(sa_real_dist.max())

    if sa_fixed is None:
        sa_value = float(np.median(sa_real_dist))
    else:
        sa_value = float(sa_fixed)

    print(f"EST range: {est_min:.6f} -- {est_max:.6f}")
    print(f"SA  range: {sa_min:.6f} -- {sa_max:.6f}")
    print(f"SA_FIXED = {sa_value:.6f}")

    return est_min, est_max, sa_min, sa_max, sa_value
