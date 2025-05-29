# scripts/simulate_data.py

import numpy as np
import pandas as pd


def simulate_behavior(df, days=30, seed=42):
    np.random.seed(seed)
    df_sim = df.copy()

    # Simulate logins per customer
    df_sim['logins'] = np.random.poisson(lam=5, size=len(df)) * (days / 30)

    # Simulate support contacts
    df_sim['support_contacts'] = np.random.binomial(
        n=days, p=0.03, size=len(df))

    # Simulate data usage in GB
    df_sim['data_usage_gb'] = np.round(np.random.gamma(
        shape=2, scale=1.5, size=len(df)) * (days / 30), 2)

    return df_sim
