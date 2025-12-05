import common
import csv, json, random, pathlib
from typing import Dict, List, Tuple
from pathlib import Path
import pandas as pd

def load_benign(jsonl_path: str, csv_path: str) -> Tuple[Dict[str, str],
                                                          List[Tuple[str, int]]]:
    idx, meta = {}, []
    jp, cp = map(pathlib.Path, (jsonl_path, csv_path))

    with jp.open() as fj, cp.open() as fc:
        reader = csv.DictReader(fc)
        for j_line, c_row in zip(fj, reader):
            obj = json.loads(j_line)
            key = f"{obj['minute_ts']}|{obj['imsi']}"
            idx[key] = j_line.rstrip('\n')
            meta.append((key, int(c_row["num_tokens"])))
    return idx, meta

def load_anomaly(jsonl_path: str, csv_path: str) -> Tuple[Dict[str, str],
                                                          List[Tuple[str, int]]]:
    idx, meta, seen = {}, [], set()
    jp, cp = map(pathlib.Path, (jsonl_path, csv_path))

    with jp.open() as fj, cp.open() as fc:
        reader = csv.DictReader(fc)
        for j_line, c_row in zip(fj, reader):
            obj = json.loads(j_line)
            key = (
                f"{obj['minute_ts']}|{obj['imsi']}|"
                f"{c_row['mutation']}|{c_row['field_group']}"
            )
            if key in seen:
                continue
            seen.add(key)
            idx[key] = j_line.rstrip('\n')
            meta.append((key, int(c_row["num_tokens"])))
    return idx, meta

def load_beningn_and_anomalies():
    print("\nIndexing benign & anomaly sources ...")
    benign_idx, benign_meta = load_benign(common.path_file_fused_filtered, 
                                        common.path_file_fused_filtered_lengths)
    print("\nbenign_idx length:", len(benign_idx), "| benign_meta length:", len(benign_meta))
    anomaly_idx, anomaly_meta = load_anomaly(common.path_file_anomalies, 
                                            common.path_file_anomaly_specs)
    print("\nanomaly_idx length:", len(anomaly_idx), "| anomaly_meta length:", len(anomaly_meta))
    return benign_idx, benign_meta, anomaly_idx, anomaly_meta

random.seed(42)

def prepare_llm_datasets():

    benign_idx, benign_meta, anomaly_idx, anomaly_meta = load_beningn_and_anomalies()

    skip_by_model = {m: set() for m in common.MODEL_LIMITS}
    cons_path = Path(common.path_file_anomalies_consolidation)
    if cons_path.exists():
        use_cols = ["minute_ts", "imsi"] + list(common.MODEL_LIMITS.keys())
        df_cons  = pd.read_csv(cons_path, usecols=use_cols)
        df_cons["key"] = df_cons["minute_ts"].astype(str) + "|" + df_cons["imsi"]

        for model in common.MODEL_LIMITS:
            if model in df_cons.columns:
                seen_keys = df_cons.loc[df_cons[model] != -1, "key"]
                skip_by_model[model].update(seen_keys.tolist())

    for model, max_tok in common.MODEL_LIMITS.items():
        print(f"\nPreparing dataset for {model}  (≤ {max_tok:,} tokens)")
        out_csv = pathlib.Path(common.path_ollama + "/" + model.replace(":", "_") + ".csv")
        if out_csv.exists():
            print("Found saved file:", out_csv.name)
            print("Skipping LLM dataset preparation...\n")
            continue
        
        benign_keys = [k for k, n in benign_meta if n <= max_tok and k in benign_idx]
        if not benign_keys:
            print("\n\nNone of the benign records fit the context window!!!\n\n")
            continue

        target_anom = max(1, len(benign_keys) * common.precentage_anomalies // 100)
        anom_keys_pool = [
            k for k, n in anomaly_meta
            if n <= max_tok and k in anomaly_idx and k not in skip_by_model[model]
        ]
        if not anom_keys_pool:
            print("\n\nNo *new* anomalous records fit the context window!!!\n\n")
            continue

        random.shuffle(anom_keys_pool)
        anom_keys = anom_keys_pool[:target_anom]

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="") as fout:
            writer = csv.writer(fout)
            writer.writerow(
                ["minute_ts", "imsi", "true_label", "datapoint"]
            )

            for key in benign_keys:
                ts, imsi = key.split("|", 1)
                writer.writerow([ts, imsi, 0, benign_idx[key]])

            for key in anom_keys:
                ts, imsi = key.split("|", 1)
                writer.writerow([ts, imsi, 1, anomaly_idx[key]])

        print("\nWriting complete for", out_csv.name)

    print("\nAll dataset CSVs stored in", common.path_ollama)
