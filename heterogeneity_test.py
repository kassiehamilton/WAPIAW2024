import os
import readline

import numpy as np
import pandas as pd
from scipy.stats import chi2

def set_confounds_type(confounds):
    if pd.isna(confounds):
        return 'None'
    else:
        return 'all'

def set_confounds_type_detail(confounds):
    if pd.isna(confounds):
        return 'None'
    else:
        return confounds

def get_all_val(indir):
    vals = pd.DataFrame()
    file_list = [file for file in os.listdir(indir) if file.endswith('.csv')]
    for i in file_list:
        valset = pd.read_csv(os.path.join(indir, i), index_col=0, header=0)
        vals = pd.concat([vals, valset], axis=0, ignore_index=True)
    vals['confounds_type'] = vals['confounds'].apply(set_confounds_type)
    vals['confounds_type_details'] = vals['confounds'].apply(set_confounds_type_detail)
    vals = vals[vals['regressor'].str.contains('IDP')]
    mapping = {
        ('ABCD', 'uppsyssnegativeurgency'): '*ABCD*:uppsyssnegativeurgency,*UKB*:N,*HCP_D*:uppsneg,*HCP_A*:neon,*HCP_YA*:Neuroticism,*ANXPE*:neov',
        ('ANXPE', 'neov'): '*ABCD*:uppsyssnegativeurgency,*UKB*:N,*HCP_D*:uppsneg,*HCP_A*:neon,*HCP_YA*:Neuroticism,*ANXPE*:neov',
        ('HCP_Aging', 'neon'): '*ABCD*:uppsyssnegativeurgency,*UKB*:N,*HCP_D*:uppsneg,*HCP_A*:neon,*HCP_YA*:Neuroticism,*ANXPE*:neov',
        ('HCP_Development', 'uppsneg'): '*ABCD*:uppsyssnegativeurgency,*UKB*:N,*HCP_D*:uppsneg,*HCP_A*:neon,*HCP_YA*:Neuroticism,*ANXPE*:neov',
        ('HCP_YA', 'Neuroticism'): '*ABCD*:uppsyssnegativeurgency,*UKB*:N,*HCP_D*:uppsneg,*HCP_A*:neon,*HCP_YA*:Neuroticism,*ANXPE*:neov',
        ('UKB', 'N'): '*ABCD*:uppsyssnegativeurgency,*UKB*:N,*HCP_D*:uppsneg,*HCP_A*:neon,*HCP_YA*:Neuroticism,*ANXPE*:neov',
        ('ABCD', 'cbclscrdsmdepressr'): '*ABCD*:cbclscrdsmdepressr,*UKB*:rds,*HCP_D*:dsmscale,*HCP_A*:sadRawScore,*HCP_YA*:Sadness,*ANXPE*:hamdtotal',
        ('ANXPE', 'hamdtotal'): '*ABCD*:cbclscrdsmdepressr,*UKB*:rds,*HCP_D*:dsmscale,*HCP_A*:sadRawScore,*HCP_YA*:Sadness,*ANXPE*:hamdtotal',
        ('HCP_Aging', 'sadRawScore'): '*ABCD*:cbclscrdsmdepressr,*UKB*:rds,*HCP_D*:dsmscale,*HCP_A*:sadRawScore,*HCP_YA*:Sadness,*ANXPE*:hamdtotal',
        ('HCP_Development', 'dsmscale'): '*ABCD*:cbclscrdsmdepressr,*UKB*:rds,*HCP_D*:dsmscale,*HCP_A*:sadRawScore,*HCP_YA*:Sadness,*ANXPE*:hamdtotal',
        ('HCP_YA', 'Sadness'): '*ABCD*:cbclscrdsmdepressr,*UKB*:rds,*HCP_D*:dsmscale,*HCP_A*:sadRawScore,*HCP_YA*:Sadness,*ANXPE*:hamdtotal',
        ('UKB', 'rds'): '*ABCD*:cbclscrdsmdepressr,*UKB*:rds,*HCP_D*:dsmscale,*HCP_A*:sadRawScore,*HCP_YA*:Sadness,*ANXPE*:hamdtotal',
        ('ABCD', 'cbclscrsynanxdepr'): '*ABCD*:cbclscrsynanxdepr,*UKB*:phq,*HCP_D*:cbclanxdep',
        ('HCP_Development', 'cbclanxdep'): '*ABCD*:cbclscrsynanxdepr,*UKB*:phq,*HCP_D*:cbclanxdep',
    }
    vals['phe_for_plot'] = vals.apply(lambda row: mapping.get((row['Dataset'], row['phenotype']), ''), axis=1)
    return vals

# --------------------------- Heterogeneity function ---------------------------

def heterogeneity_three_tests(y, v):
    """
    y: 1D array of per-dataset effect estimates
    v: 1D array of per-dataset variances (SE^2)
    Returns: dict(k, Q, df, p_Q, I2 (0..1), tau2_DL)
    """
    y = np.asarray(y, float)
    v = np.asarray(v, float)

    # keep finite and positive-variance entries
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[ok], v[ok]
    k = y.size
    if k < 2:
        return None

    w = 1.0 / v
    sw = np.sum(w)
    mu_fe = np.sum(w * y) / sw

    # Cochran's Q
    Q = float(np.sum(w * (y - mu_fe) ** 2))
    df = int(k - 1)
    p_Q = float(1.0 - chi2.cdf(Q, df))

    # I^2 (proportion)
    I2 = float(max((Q - df) / Q, 0.0)) if Q > 0 and df > 0 else 0.0

    # DerSimonian–Laird tau^2
    c = sw - (np.sum(w**2) / sw)
    tau2_DL = float(max((Q - df) / c, 0.0)) if c > 0 else 0.0

    return {"k": k, "Q": Q, "df": df, "p_Q": p_Q, "I2": I2, "tau2_DL": tau2_DL}

# --------------------------- Configure paths ---------------------------

INPUT_DIR  = ".../"
OUTPUT_DIR = ".../"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------- Load & preprocess ---------------------------

df = get_all_val(INPUT_DIR)
df['IDP_name'] = df['parcel']
df.loc[df['phenotype'] == 'phq', 'Dataset'] = 'UKB_phq'
df = df[(df['phenotype'] != 'phq') &
        (df['phenotype'] != 'cbclscrsynanxdepr') &
        (df['phenotype'] != 'cbclanxdep')]
dataset_list = ['ABCD', 'ANXPE', 'HCP_Aging', 'HCP_Development', 'HCP_YA', 'UKB']
phen_neuroticism = {
    'ABCD': 'uppsyssnegativeurgency',
    'UKB': 'N',
    'HCP_Development': 'uppsneg',
    'HCP_Aging': 'neon',
    'HCP_YA': 'Neuroticism',
    'ANXPE': 'neov'
}
phen_depress = {
    'ABCD': 'cbclscrdsmdepressr',
    'UKB': 'rds',
    'HCP_Development': 'dsmscale',
    'HCP_Aging': 'sadRawScore',
    'HCP_YA': 'Sadness',
    'ANXPE': 'hamdtotal'
}

phen_groups = {
    "Neuroticism": phen_neuroticism,
    "Depression": phen_depress
}

# --------------------------- Run heterogeneity (no covars) ---------------------------

out_rows = []
unique_idps = df['IDP_name'].dropna().unique()

for phen_group_name, phen_group in phen_groups.items():
    for idp_name in unique_idps:
        y, v = [], []

        for dataset in dataset_list:
            phen = phen_group.get(dataset)
            if phen is None:
                continue

            row = df[
                (df['IDP_name'] == idp_name) &
                (df['Dataset'] == dataset) &
                (df['phenotype'] == phen)
            ]
            if row.empty:
                continue

            # pull per-dataset effect and variance
            y.append(row['coef_'].values[0])
            v.append(row['var_coef_'].values[0])

        stats = heterogeneity_three_tests(y, v)
        if stats is None:
            continue

        out_rows.append({
            "IDP_name": idp_name,
            "phen_group": phen_group_name,
            "k": stats["k"],
            "Q": stats["Q"],
            "df": stats["df"],
            "p_Q": stats["p_Q"],
            "I2": stats["I2"],
            "I2_percent": stats["I2"] * 100,
            "tau2_DL": stats["tau2_DL"]
        })

het_df = pd.DataFrame(out_rows).sort_values(["phen_group", "IDP_name"]).reset_index(drop=True)
alpha = 0.05
het_df["sig_Q"] = (het_df["p_Q"] < alpha) & het_df["p_Q"].notna()

def i2_level(p):
    if pd.isna(p):
        return np.nan
    if p < 25:
        return "low"          # ~0–24%
    elif p < 50:
        return "moderate"     # 25–49%
    elif p < 75:
        return "high"         # 50–74%
    else:
        return "very high"    # ≥75%
het_df["I2_level"] = het_df["I2_percent"].apply(i2_level)
het_df["I2_level"] = pd.Categorical(
    het_df["I2_level"],
    categories=["low", "moderate", "high", "very high"],
    ordered=True)

out_csv = os.path.join(OUTPUT_DIR, "heterogeneity_Q_I2_tau2_only.csv")
het_df.to_csv(out_csv, index=False)

