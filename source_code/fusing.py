import pandas as pd
import json

def join_by_minute_and_imsi(path_file_fused, ss7_df, dia_df, gtp_df):

    ss7_df = ss7_df.dropna(subset=["imsi", "timestamp"]).copy()
    dia_df = dia_df.dropna(subset=["imsi", "timestamp"]).copy()
    gtp_df = gtp_df.dropna(subset=["imsi", "timestamp"]).copy()

    ss7_df["protocol"] = "ss7"
    dia_df["protocol"] = "diameter"
    gtp_df["protocol"] = "gtp"

    combined_df = pd.concat([ss7_df, dia_df, gtp_df], ignore_index=True)

    combined_df["minute_ts"] = combined_df["timestamp"].dt.floor("T")

    combined_df = combined_df.sort_values(by=["minute_ts", "timestamp"])

    grouped_data = {}

    for row in combined_df.itertuples(index=False):
        key = (row.minute_ts, row.imsi)

        if key not in grouped_data:
            grouped_data[key] = {
                "minute_ts": row.minute_ts.isoformat(),
                "imsi": row.imsi,
                "ss7": [],
                "diameter": [],
                "gtp": []
            }

        record_data = row.raw if hasattr(row, "raw") else row._asdict()
        grouped_data[key][row.protocol].append(record_data)

    fused_records = list(grouped_data.values())

    with open(path_file_fused, "w") as f:
        for record in fused_records:
            f.write(json.dumps(record, default=str) + "\n")

    return fused_records
