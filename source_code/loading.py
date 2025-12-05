import json
import pandas as pd
from datetime import datetime
import os
import glob

def load_all_raw_json_from_csvs(directory):
    json_lines = []
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    for file in csv_files:
        try:
            df = pd.read_csv(file)
            if "_raw" in df.columns:
                json_lines.extend(df["_raw"].dropna().tolist())
        except Exception as e:
            print(f"Failed to read {file}: {e}")
    return json_lines

def load_protocol_data(json_lines, protocol_name):
    records = []
    for line in json_lines:
        try:
            data = json.loads(line)
            imsi = data.get("sid", None) or data.get("protocol", {}).get(protocol_name, {}).get("imsi", None)
            iid = data.get("iid", None)
            ts = data.get("protocol", {}).get(protocol_name, {}).get("timestamp", None)
            if ts:
                ts = datetime.utcfromtimestamp(ts)
            record = {
                "imsi": imsi,
                "iid": iid,
                "timestamp": ts,
                "raw": data.get("protocol", {}).get(protocol_name, {})
            }
            records.append(record)
        except Exception as e:
            print(f"Failed to parse line: {e}")
    return pd.DataFrame(records)

