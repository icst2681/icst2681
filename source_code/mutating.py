import json
import random
import copy
from typing import Any, Dict, List, Tuple, Union
import common
import pandas as pd
from pathlib import Path

__all__ = [
    "FIELDS_GROUPS",
    "swap_field_group",
    "mutate",
]

PathType = List[Union[str, int]]
ParentRef = Tuple[Any, Union[str, int]]

def _collect_nodes(
    current: Any,
    template: PathType,
    parent: Any,
    key: Union[str, int],
    out: List[ParentRef],
) -> None:
    if not template:
        out.append((parent, key))
        return

    seg, *rest = template

    if isinstance(current, list):
        indices = range(len(current)) if seg == "*" else [int(seg)] if seg.isdigit() else []
        for idx in indices:
            if 0 <= idx < len(current):
                _collect_nodes(current[idx], rest, current, idx, out)
        return

    if isinstance(current, dict):
        if seg == "*":
            for k, v in current.items():
                _collect_nodes(v, rest, current, k, out)
        elif seg in current:
            _collect_nodes(current[seg], rest, current, seg, out)
        return

    return


def find_nodes(obj: Any, template: PathType) -> List[ParentRef]:
    results: List[ParentRef] = []
    _collect_nodes(obj, template, None, None, results)
    return results


def swap_field_group(
    rec_a: Dict[str, Any],
    rec_b: Dict[str, Any],
    group_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:

    a = copy.deepcopy(rec_a)
    b = copy.deepcopy(rec_b)

    for path in common.MUTANT_FIELD_GROUPS[group_name]:
        nodes_a = find_nodes(a, path)
        nodes_b = find_nodes(b, path)
        for (parent_a, key_a), (parent_b, key_b) in zip(nodes_a, nodes_b):
            if parent_a is None or parent_b is None:
                continue
            parent_a[key_a], parent_b[key_b] = parent_b[key_b], parent_a[key_a]
    return a, b


def mutate(
    percent: float = 0.05,
    random_seed: int | None = None,
) -> Tuple[str, str]:

    if Path(common.path_file_anomalies).exists() \
        and Path(common.path_file_seeds).exists() \
        and Path(common.path_file_anomaly_specs).exists():
        print("\nFound needed files mentioned below:")
        print(common.path_file_anomalies)
        print(common.path_file_seeds)
        print(common.path_file_anomaly_specs)
        print("Skipping mutation...\n")
        return True
    
    if random_seed is not None:
        random.seed(random_seed)

    original_records = common.get_fused_filtered_data()

    n_total = len(original_records)
    if n_total == 0:
        print("\nEmpty fused data file – nothing to mutate.\n")
        return
    else:
        print(f"\nFused record count: {n_total}\n")

    sample_size = max(1, int(n_total * percent))
    sample_indices = random.sample(range(n_total), sample_size)

    anomalies: List[dict[str, Any]] = []
    seeds: List[dict[str, Any]] = []
    anomaly_specs: List[Dict[str, Any]] = []

    for idx_a in sample_indices:
        idx_b = random.choice([i for i in range(n_total) if i != idx_a])
        mutation = "swap"
        for group_name in common.MUTANT_FIELD_GROUPS:
            rec_a, rec_b = original_records[idx_a], original_records[idx_b]
            try:
                mut_a, mut_b = swap_field_group(rec_a, rec_b, group_name)
            except Exception:
                print("\nCannot apply", mutation, "::", group_name, "to", rec_a.get("minute_ts"),"::", \
                      rec_a.get("imsi"), "and", rec_b.get("minute_ts"),"::", rec_b.get("imsi"))
                continue

            anomalies.append(mut_a)
            seeds.append(rec_a)
            num_tokens_a = common.get_num_tokens(common.enc, mut_a)
            anomaly_specs.append({
                "minute_ts": mut_a.get("minute_ts"),
                "imsi"     : mut_a.get("imsi"),
                "mutation" : mutation,
                "field_group": group_name,
                "num_tokens"    : num_tokens_a
            })

            anomalies.append(mut_b)
            seeds.append(rec_b)
            num_tokens_b = common.get_num_tokens(common.enc, mut_b)
            anomaly_specs.append({
                "minute_ts": mut_b.get("minute_ts"),
                "imsi"     : mut_b.get("imsi"),
                "mutation" : mutation,
                "field_group": group_name,
                "num_tokens"    : num_tokens_b
            })

    with open(common.path_file_anomalies, "w", encoding="utf-8") as fh:
        for rec in anomalies:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(common.path_file_seeds, "w", encoding="utf-8") as fh:
        for rec_seed in seeds:
            fh.write(json.dumps(rec_seed, ensure_ascii=False) + "\n")

    pd.DataFrame(anomaly_specs).to_csv(common.path_file_anomaly_specs, index=False)

    print(f"wrote {len(anomalies)} anomalies at {common.path_file_anomalies}")
    print(f"wrote {len(seeds)} seeds at {common.path_file_seeds}")
    print(f"wrote anomaly specs CSV at {common.path_file_anomaly_specs}")

    return True
