import numpy as np
import pandas as pd
from pathlib import Path
import common

BENIGN_CSV = common.path_file_benign_consolidation
ANOM_CSV   = common.path_file_anomalies_consolidation

print("Loading CSVs...")
df_benign = pd.read_csv(BENIGN_CSV)
df_anom   = pd.read_csv(ANOM_CSV)

def parse_synth_imsi(s: str):
    parts = str(s).split("|")
    base = parts[0]
    fam = parts[2] if len(parts) >= 3 else None
    return base, fam

df_benign["base_imsi"] = df_benign["imsi"].astype(str)

parsed = df_anom["imsi"].astype(str).apply(parse_synth_imsi)
df_anom["base_imsi"]       = parsed.apply(lambda x: x[0])
df_anom["mutation_family"] = parsed.apply(lambda x: x[1])

NON_MODEL_COLS = {
    "minute_ts", "imsi", "datapoint", "base_imsi", "mutation_family"
}

model_cols = [c for c in df_anom.columns if c not in NON_MODEL_COLS]

missing_in_benign = [c for c in model_cols if c not in df_benign.columns]
if missing_in_benign:
    raise ValueError(f"Model columns missing in benign CSV: {missing_in_benign}")

def consensus(row):
    vals = row[model_cols].values
    return np.sum(vals == 1)

df_anom["S"]    = df_anom.apply(consensus, axis=1)
df_benign["S"]  = df_benign.apply(consensus, axis=1)  # not strictly needed

benign_index = df_benign.set_index(["minute_ts", "base_imsi"])

def cosine_distance_decision_space(a: np.ndarray, b: np.ndarray) -> float:
    """
    a, b are 1D arrays of per-model decisions (0/1 or -1 for missing).

    We ignore positions where either side is -1.
    """
    mask = (a != -1) & (b != -1)
    a = a[mask]
    b = b[mask]

    if a.size == 0:
        return np.nan

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan

    cos_sim = float(np.dot(a, b) / (na * nb))
    return 1.0 - cos_sim  # cosine distance

cos_dists = []
matched_flags = []

for idx, row in df_anom.iterrows():
    key = (row["minute_ts"], row["base_imsi"])
    try:
        parent = benign_index.loc[key]
    except KeyError:
        cos_dists.append(np.nan)
        matched_flags.append(False)
        continue

    if isinstance(parent, pd.DataFrame):
        parent = parent.iloc[0]

    a_vec = row[model_cols].to_numpy(dtype=float)
    b_vec = parent[model_cols].to_numpy(dtype=float)

    d = cosine_distance_decision_space(a_vec, b_vec)
    cos_dists.append(d)
    matched_flags.append(True)

df_anom["cos_dist_parent"] = cos_dists
df_anom["matched_benign"]  = matched_flags

print(f"Total synthetic rows: {len(df_anom)}")
print(f"Matched to benign:    {df_anom['matched_benign'].sum()}")

summary = (
    df_anom[df_anom["matched_benign"]]
    .groupby("S")["cos_dist_parent"]
    .agg(["count", "mean", "median"])
    .reset_index()
    .sort_values("S")
)

print("\nCosine distance to benign parent by consensus S:")
print(summary)

OUT_CSV = Path(common.path_results) / "cosine_by_S.csv"
summary.to_csv(OUT_CSV, index=False)
print(f"\nSaved summary to {OUT_CSV}")
