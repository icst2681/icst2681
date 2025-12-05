
from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict

import common

def main() -> None:

    ss7_df, dia_df, gtp_df = common.load_and_return()
    total_ss7 = len(ss7_df)
    total_dia = len(dia_df)
    total_gtp = len(gtp_df)

    lengths_csv = common.path_file_fused_filtered_lengths

    kept_rows = common.get_fused_filtered_data()
    fused_kept_n = len(kept_rows)

    rows_exactly2 = 0
    rows_all3 = 0
    total_ss7_in_kept = 0
    total_dia_in_kept = 0
    total_gtp_in_kept = 0

    for rec in kept_rows:
        ss7 = rec.get("ss7") or []
        dia = rec.get("diameter") or []
        gtp = rec.get("gtp") or []
        total_ss7_in_kept += len(ss7)
        total_dia_in_kept += len(dia)
        total_gtp_in_kept += len(gtp)
        present = (1 if ss7 else 0) + (1 if dia else 0) + (1 if gtp else 0)
        if present == 2:
            rows_exactly2 += 1
        elif present == 3:
            rows_all3 += 1

    cand_rows = common.get_fused_data()
    total_candidates = len(cand_rows)

    avg_tokens = None
    if lengths_csv:
        p = Path(lengths_csv) if not isinstance(lengths_csv, Path) else lengths_csv
        if p.exists():
            total_tokens = 0
            n_rows = 0
            with p.open("r", encoding="utf-8") as f:
                rdr = csv.DictReader(f)
                for row in rdr:
                    val = row.get("num_tokens")
                    if not val:
                        continue
                    try:
                        total_tokens += int(val)
                    except ValueError:
                        total_tokens += float(val)
                    n_rows += 1
            if n_rows:
                avg_tokens = total_tokens / n_rows

    family_counts: Dict[str, int] = {}
    FAMILY_LABELS = {
        "ss7_cg_gt":          "SS7: Calling-Party GT (CGGT)",
        "ss7_cd_gt":          "SS7: Called-Party GT (CDGT)",
        "ss7_point_code":     "SS7: Point Codes (OPC/DPC)",
        "ss7_application_context": "SS7: Application Context (ACN)",
        "dia_user_name":      "Diameter: User-Name (IMSI)",
        "dia_session_id":     "Diameter: Session-Id",
        "dia_origin_host":    "Diameter: Origin-Host",
        "dia_visited_plmn":   "Diameter: Visited-PLMN-Id",
        "dia_apn_service":    "Diameter: APN Service-Selection",
        "gtp_teid":          "GTP: TEIDs (control/user plane)",
        "gtp_apn":            "GTP: APN",
        "gtp_user_loc":            "GTP: User Location Information (ULI)",
        "gtp_pdn_ip":       "GTP: PDN IP address (PAA)",
    }

    anoms_csv = Path(common.path_file_anomaly_specs)
    with anoms_csv.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            key = (row.get("field_group") or "").strip()
            if not key:
                continue
            label = FAMILY_LABELS.get(key, key)
            family_counts[label] = family_counts.get(label, 0) + 1

    print("\n=== TABLE I: Dataset summary ===")
    print(f"Total SS7 fragments: {total_ss7}")
    print(f"Total Diameter fragments: {total_dia}")
    print(f"Total GTP fragments: {total_gtp}")
    print(f"Total fused candidates (all windows): {total_candidates if total_candidates is not None else 'N/A'}")
    print(f"Fused kept (minutes with >=2 protocols): {fused_kept_n}")
    print(f"Rows with exactly 2 protocols: {rows_exactly2}")
    print(f"Rows with all 3 protocols: {rows_all3}")

    if avg_tokens is not None:
        print(f"\nAverage tokens per fused record (from fused_filtered_lengths.csv): {avg_tokens:.1f}")

    if family_counts:
        print("\n=== TABLE II: Synthetic anomalies grouped by mutation ===")
        total = 0
        for k in sorted(family_counts.keys()):
            print(f"{k}: {family_counts[k]}")
            total += family_counts[k]
        print(f"Total: {total}")
    else:
        print("\n[info] File", anoms_csv, "missing. Table II skipped.")

if __name__ == "__main__":
    main()
