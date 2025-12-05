from pathlib import Path
from typing import List, Union
import common

import pandas as pd

def consolidate(
    flag_type: int,
    pred_dir: Union[str, Path] | None = None,
    out_csv: Union[str, Path] | None = None
) -> Path:
    if pred_dir is None:
        pred_dir = common.path_ollama
    else:
        pred_dir = Path(pred_dir)

    if out_csv is None:
        if flag_type == 0:
            out_csv = common.path_models + "/benign_consolidation_" + common.tracker_path_ollama + ".csv"
        else:
            out_csv = common.path_models + "/anomalies_consolidation_" + common.tracker_path_ollama + ".csv"
        print("out_csv =", out_csv)
    else:
        out_csv = Path(out_csv)

    if Path(out_csv).exists():
        merged = pd.read_csv(out_csv)
        model_cols = [
            c for c in merged.columns
            if c not in ("minute_ts", "imsi", "datapoint", "predicted_label_sum")
        ]
        merged[model_cols] = merged[model_cols].astype("Int8")
        print(f"[info] loaded previous consolidation ({len(merged):,} rows)")
    else:
        merged = None
        model_cols: List[str] = []

    for model, _ in common.MODEL_LIMITS.items():
        stub = model.replace(":", "_")
        pred_path = Path(common.path_ollama + "/" + stub + "_predictions.csv")
        base_path = Path(common.path_ollama + "/" + stub + ".csv")

        print(f"\nReading", pred_path)
        if not pred_path.exists():
            print("File not found, skipping...")
            continue
        
        df_pred = pd.read_csv(pred_path)
        if base_path.exists():
            df_base = pd.read_csv(base_path,
                                    usecols=["minute_ts", "imsi", "true_label", "datapoint"])
            df_pred = df_pred.merge(
                df_base, on=["minute_ts", "imsi", "true_label"], how="left"
            )
        else:
            print(str(base_path), "not found! Skipping...")
            continue
        
        df_pred = df_pred[df_pred["true_label"] == flag_type].copy()
        df_pred.reset_index(drop=True, inplace=True)

        col_name = model
        df_pred.rename(columns={"predicted_label": col_name}, inplace=True)
        df_pred = df_pred[["minute_ts", "imsi", "datapoint", col_name]]
        df_pred[col_name] = df_pred[col_name].astype("Int8")

        if merged is None:
            merged = df_pred
            model_cols = [col_name]
        else:
            if col_name not in merged.columns:
                merged[col_name] = -1
                model_cols.append(col_name)

            merged = merged.merge(
                df_pred,
                on=["minute_ts", "imsi", "datapoint"],
                how="outer",
                suffixes=("", "_new"),
            )
            fresh = f"{col_name}_new"
            merged[col_name] = merged[fresh].combine_first(merged[col_name]).astype("Int8")
            merged.drop(columns=fresh, inplace=True)

    if merged is None:
        print("Processing resulted in no CSV!")
        return

    merged[model_cols] = merged[model_cols].fillna(-1).astype("Int8")
    for m, _ in common.MODEL_LIMITS.items():
        if m not in merged.columns:
            merged[m] = -1
            model_cols.append(m)
    merged[model_cols] = merged[model_cols].fillna(-1).astype("Int8")

    merged["predicted_label_sum"] = (
        merged[model_cols].clip(lower=0).sum(axis=1).astype("Int16")
    )

    merged.to_csv(out_csv, index=False)
    print("\nColidated saved at", out_csv)

def consolidate_final(flag_type: int):
    print("\n\nIn the final considation phase now...")

    src_dir   = Path(common.path_models)
    if flag_type == 0:
        pattern   = "benign_consolidation*.csv"
        out_csv   = Path(common.path_file_benign_consolidation)
    else:    
        pattern   = "anomalies_consolidation*.csv"
        out_csv   = Path(common.path_file_anomalies_consolidation)
    keys      = ["minute_ts", "imsi", "datapoint"]

    print("Reading", pattern, "files to dataframes...")
    dfs = [pd.read_csv(p) for p in src_dir.glob(pattern)]
    if not dfs:
        print("No consolidation files matched the pattern!")
        return

    print("Bringing all dataFrames to the same set of columns...")
    model_cols = sorted(
        {c for df in dfs for c in df.columns}
        - set(keys + ["predicted_label_sum"])
    )

    def prepare(df):
        print("Adding missing model columns to be filled with '-1'...")
        missing = set(model_cols) - set(df.columns)
        for m in missing:
            df[m] = -1
        print("Keeping only what we need and ensuring Int8...")
        df = df[keys + model_cols]
        df[model_cols] = df[model_cols].astype("Int8")
        return df

    dfs = [prepare(df) for df in dfs]

    print("Concatenating and collapsing duplicates...")
    cat = pd.concat(dfs, ignore_index=True)
    print("For duplicates, taking element-wise max (1 > 0 > -1) per model column...")
    agg_max = cat.groupby(keys, as_index=False)[model_cols].max()

    print("Re-computing the ensemble vote...")
    agg_max["predicted_label_sum"] = (
        agg_max[model_cols].clip(lower=0).sum(axis=1).astype("Int16")
    )

    print("Saving final consolidation to", out_csv, "...")
    agg_max.to_csv(out_csv, index=False)
    print(f"Saved {len(agg_max):,} unique rows to {out_csv}")
