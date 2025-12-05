import common
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable
import numpy as np
import pandas as pd

try:
    from scipy.stats import fisher_exact, chi2_contingency
except Exception:
    fisher_exact = None
    chi2_contingency = None

try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    multipletests = None


def _detect_model_cols(df: pd.DataFrame) -> List[str]:
    meta_cols = {"minute_ts", "imsi", "datapoint", "predicted_label_sum"}
    model_cols = [
        c for c in df.columns
        if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])
    ]
    return model_cols

def load_consolidations(
    model_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    df_b = pd.read_csv(common.path_file_benign_consolidation)
    df_a = pd.read_csv(common.path_file_anomalies_consolidation)
    if model_cols is None:
        model_cols = _detect_model_cols(df_b)
    return df_b, df_a, model_cols


def agreement_tallies(
    df_b: pd.DataFrame, df_a: pd.DataFrame, model_cols: List[str]
) -> Dict[str, pd.DataFrame]:
    M = len(model_cols)

    hist_b = df_b["predicted_label_sum"].value_counts().reindex(range(M + 1), fill_value=0).sort_index()
    hist_a = df_a["predicted_label_sum"].value_counts().reindex(range(M + 1), fill_value=0).sort_index()
    hist = pd.DataFrame({"predicted_label_sum": range(M + 1), "background": hist_b.values, "synthetic": hist_a.values})

    cum_b = hist_b[::-1].cumsum()[::-1]
    cum_a = hist_a[::-1].cumsum()[::-1]
    cum = pd.DataFrame({"k": range(M + 1), "background_ge_k": cum_b.values, "synthetic_ge_k": cum_a.values})

    miss_row_b = (df_b[model_cols] == -1).mean(axis=1)
    miss_row_a = (df_a[model_cols] == -1).mean(axis=1)
    missing_row_summary = pd.DataFrame({
        "set": ["background", "synthetic"],
        "avg_missing_rate_per_row": [miss_row_b.mean(), miss_row_a.mean()],
        "median_missing_rate_per_row": [miss_row_b.median(), miss_row_a.median()],
    })

    miss_model_b = (df_b[model_cols] == -1).mean(axis=0).rename("background_missing_rate")
    miss_model_a = (df_a[model_cols] == -1).mean(axis=0).rename("synthetic_missing_rate")
    missing_model = pd.concat([miss_model_b, miss_model_a], axis=1).reset_index(names=["model"])

    return {
        "hist_S": hist,
        "cum_S_ge_k": cum,
        "missing_row": missing_row_summary,
        "missing_model": missing_model,
    }

def stats_tests_at_k(
    df_b: pd.DataFrame, df_a: pd.DataFrame, model_cols: List[str], fdr_method: str = "fdr_bh"
) -> pd.DataFrame:
    if fisher_exact is None or chi2_contingency is None:
        raise ImportError("scipy is required for stats_tests_at_k")

    M = len(model_cols)
    N_b, N_a = len(df_b), len(df_a)
    rows = []
    for k in range(1, M + 1):
        TP_k = int((df_a["predicted_label_sum"] >= k).sum())
        FP_k = int((df_b["predicted_label_sum"] >= k).sum())
        table = [[TP_k, N_a - TP_k], [FP_k, N_b - FP_k]]
        OR, p_fisher = fisher_exact(table, alternative="two-sided")
        chi2, p_chi, _, _ = chi2_contingency(table, correction=False)
        rows.append(dict(k=k, odds_ratio=OR, p_fisher=p_fisher, p_chi=p_chi))
    out = pd.DataFrame(rows)

    if multipletests is None:
        out["p_fisher_adj"] = np.nan
        out["significant_fdr5"] = np.nan
    else:
        rej, p_adj, _, _ = multipletests(out["p_fisher"].values, alpha=0.05, method=fdr_method)
        out["p_fisher_adj"] = p_adj
        out["significant_fdr5"] = rej
    return out

def top_consensus_examples(df: pd.DataFrame, top_n: int = 10, keep_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if keep_cols is None:
        keep_cols = ["minute_ts", "imsi", "datapoint", "predicted_label_sum"]
    keep = [c for c in keep_cols if c in df.columns]
    return df.sort_values("predicted_label_sum", ascending=False).loc[:, keep].head(top_n).reset_index(drop=True)


def apply_min_coverage(df: pd.DataFrame, model_cols: List[str], q_frac: float = 0.8) -> pd.DataFrame:
    M = len(model_cols)
    nonmiss = (df[model_cols] != -1).sum(axis=1)
    q = int(np.ceil(q_frac * M))
    return df.loc[nonmiss >= q].copy()


def headline_numbers(
    df_b: pd.DataFrame,
    df_a: pd.DataFrame,
    model_cols: List[str],
    k_high: Optional[int] = None,
) -> pd.DataFrame:
    M = len(model_cols)
    cum = agreement_tallies(df_b, df_a, model_cols)["cum_S_ge_k"]
    N_b, N_a = len(df_b), len(df_a)

    share_b_ge1 = cum.loc[cum["k"] == 1, "background_ge_k"].iloc[0] / N_b if N_b else np.nan
    share_a_ge1 = cum.loc[cum["k"] == 1, "synthetic_ge_k"].iloc[0] / N_a if N_a else np.nan

    if k_high is None:
        k_high = int(np.ceil(M / 2))

    count_b_high = int(cum.loc[cum["k"] == k_high, "background_ge_k"].iloc[0])
    count_a_high = int(cum.loc[cum["k"] == k_high, "synthetic_ge_k"].iloc[0])

    rows = [{
        "M_models": M,
        "N_background": N_b,
        "N_synthetic": N_a,
        "share_background_S_ge1": share_b_ge1,
        "share_synthetic_S_ge1": share_a_ge1,
        "k_high": k_high,
        "background_S_ge_k_high": count_b_high,
        "synthetic_S_ge_k_high": count_a_high,
    }]
    return pd.DataFrame(rows)


def compute_all_results(df_b, df_a, model_cols,
    min_coverage_q: Optional[float] = None,
) -> Dict[str, pd.DataFrame | float]:
    out: Dict[str, pd.DataFrame | float] = {}
    out.update(agreement_tallies(df_b, df_a, model_cols))
    out["top_synth"] = top_consensus_examples(df_a)
    out["top_background"] = top_consensus_examples(df_b)
    out["headline"] = headline_numbers(df_b, df_a, model_cols)
    try:
        out["stats_tests"] = stats_tests_at_k(df_b, df_a, model_cols)
    except Exception:
        pass
    return out


if __name__ == "__main__":
    df_b, df_a, model_cols = load_consolidations()
    if common.min_coverage is not None:
        df_b = apply_min_coverage(df_b, model_cols, q_frac=common.min_coverage)
        df_a = apply_min_coverage(df_a, model_cols, q_frac=common.min_coverage)

    tables = compute_all_results(df_b, df_a, model_cols, common.min_coverage)

    # Write outputs
    dir_results = Path(common.path_results)
    dir_results.mkdir(parents=True, exist_ok=True)
    for key, val in tables.items():
        if isinstance(val, pd.DataFrame):
            val.to_csv(f"{common.prefix_results}_{key}.csv", index=False)
        else:
            with open(f"{common.prefix_results}_{key}.txt", "w", encoding="utf-8") as fh:
                fh.write(str(val))
    print("Done")
