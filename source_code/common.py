import json
import loading
import os
import fusing
from typing import Dict, List, Union
import csv
import tiktoken

path_main = "icst2681"
path_data = path_main + "/data"
path_models = path_main + "/models"
path_file_fused = path_data + "/fused.jsonl"
path_manual_features_based_scores = path_models + "/manual_features_based_scores.csv"
path_transformer_features_based_scores = path_models + "/transformer_features_based_scores.csv"
path_file_anomalies = path_data + "/anomalies.jsonl"
path_file_seeds = path_data + "/seeds.jsonl"
path_file_anomaly_specs = path_data + "/anomaly_specs.csv"
path_file_fused_filtered = path_data + "/fused_filtered.jsonl"
path_file_fused_filtered_lengths = path_data + "/fused_filtered_lengths.csv"
enc = tiktoken.encoding_for_model("gpt-4o-mini")
precentage_anomalies = 5
threads_embeddings = 4
path_file_anomalies_consolidation = path_models + "/anomalies_consolidation.csv"
path_file_benign_consolidation = path_models + "/benign_consolidation.csv"
tracker_path_ollama = "0"
path_tracker_path_ollama = path_models + "/tracker_path_ollama.txt"
path_ollama = path_models + "/ollama_" + tracker_path_ollama
path_results = path_main + "/results"
prefix_results = path_results + "/results"
min_coverage = None

MUTANT_FIELD_GROUPS: Dict[str, List[List[Union[str, int]]]] = {
    "ss7_cg_gt": [
        ["ss7", "*", "cggt"],
        ["ss7", "*", "cggt_country"],
        ["ss7", "*", "cggt_network"],
        ["ss7", "*", "cggt_tadig"],
        ["ss7", "*", "cggt_mccmnc"],
    ],
    "ss7_cd_gt": [
        ["ss7", "*", "cdgt"],
        ["ss7", "*", "cdgt_country"],
        ["ss7", "*", "cdgt_network"],
        ["ss7", "*", "cdgt_tadig"],
        ["ss7", "*", "cdgt_mccmnc"],
    ],
    "ss7_point_code": [
        ["ss7", "*", "operations", "*", "opc"],
        ["ss7", "*", "operations", "*", "dpc"],
        ["ss7", "*", "opc"],
        ["ss7", "*", "dpc"],
    ],
    "ss7_application_context": [["ss7", "*", "application_context"]],

    "dia_user_name": [["diameter", "*", "req", "avps", "User-Name"]],

    "dia_session_id": [
        ["diameter", "*", "req", "avps", "Session-Id"],
        ["diameter", "*", "answer", "avps", "Session-Id"],
    ],

    "dia_origin_host": [["diameter", "*", "req", "avps", "Origin-Host"]],

    "dia_visited_plmn": [["diameter", "*", "req", "avps", "Visited-PLMN-Id", "mccmnc"]],

    "dia_apn_service": [
        [
            "diameter",
            "*",
            "answer",
            "Subscription-Data",
            "APN-Configuration-Profile",
            "APN-Configuration",
            "array",
            "*",
            "Service-Selection",
        ]
    ],

    "gtp_teid": [
        ["gtp", "*", "operations", "*", "teid"],
        ["gtp", "*", "operations", "*", "teid_cp"],
        ["gtp", "*", "operations", "*", "teid_d1"],
    ],

    "gtp_apn": [["gtp", "*", "operations", "*", "apn"]],

    "gtp_user_loc": [
        ["gtp", "*", "operations", "*", "user_location_infomation", "mcc"],
        ["gtp", "*", "operations", "*", "user_location_infomation", "mnc"],
        ["gtp", "*", "operations", "*", "user_location_infomation", "lac"],
        ["gtp", "*", "operations", "*", "user_location_infomation", "ci"],
    ],

    "gtp_pdn_ip": [["gtp", "*", "operations", "*", "end_user_addr", "ipv4addr"]],
}

MODEL_LIMITS = {
    "nomic-embed-text:v1.5"                 : 2000,
    "bge-m3:567m"                           : 8000,
    "snowflake-arctic-embed:137m"           : 2000,
    "snowflake-arctic-embed2:568m"          : 8000,
    "jina/jina-embeddings-v2-base-en"       : 8000,
    "unclemusclez/jina-embeddings-v2-base-code:q8": 8000
}

def load_and_return():
    ss7_lines = loading.load_all_raw_json_from_csvs(path_data+"/SS7")
    dia_lines = loading.load_all_raw_json_from_csvs(path_data+"/Diameter")
    gtp_lines = loading.load_all_raw_json_from_csvs(path_data+"/GTPC")

    ss7_df = loading.load_protocol_data(ss7_lines, "SS7")
    dia_df = loading.load_protocol_data(dia_lines, "Diameter")
    gtp_df = loading.load_protocol_data(gtp_lines, "GTPv1")

    return ss7_df, dia_df, gtp_df

def get_fused_data():
    if os.path.exists(path_file_fused):
        print("\nFound existing file:", path_file_fused)
        print("Loading...\n")
        with open(path_file_fused, "r") as f:
            fused_records = [json.loads(line) for line in f]
        print(f"\nFused record count: {len(fused_records)}\n")
        return fused_records

    ss7_df, dia_df, gtp_df = load_and_return()
    fused_records = fusing.join_by_minute_and_imsi(path_file_fused, ss7_df, dia_df, gtp_df)
    print(f"\nFused record count: {len(fused_records)}\n")
    return fused_records

def get_fused_filtered_data():
    if os.path.exists(path_file_fused_filtered):
        print("\nFound existing file:", path_file_fused_filtered)
        print("Loading...\n")
        with open(path_file_fused_filtered, "r") as f:
            fused_filtered_records = [json.loads(line) for line in f]
        print(f"\nFused filtered record count: {len(fused_filtered_records)}\n")
        return fused_filtered_records
    
    filter_fused_and_get_length()
    fused_filtered_records = get_fused_filtered_data()
    return fused_filtered_records

def get_num_tokens(enc, sequence):
    input = json.dumps(sequence, sort_keys=True)
    num_tokens = len(enc.encode(input))
    return num_tokens

def filter_fused_and_get_length():
    fused_data = get_fused_data()
    with open(path_file_fused_filtered, "w") as fout, open(path_file_fused_filtered_lengths, "w") as lout:
            writer = csv.writer(lout)
            writer.writerow(["minute_ts", "imsi", "num_tokens"])
            for rec in fused_data:
                values = [rec.get("ss7"), rec.get("diameter"), rec.get("gtp")]
                if sum(bool(v) for v in values) >= 2:
                    fout.write(json.dumps(rec) + "\n")

                    num_tokens = get_num_tokens(enc, rec)
                    writer.writerow([rec["minute_ts"], rec["imsi"], num_tokens])

def set_path_ollama():

    global tracker_path_ollama, path_ollama

    print("\n\nCreating tracker_path_ollama file IF it doesn't exist")
    if not os.path.exists(path_tracker_path_ollama):
        with open(path_tracker_path_ollama, "w") as f:
            print("Writing starting version as '0'")
            f.write("0")

    print("Reading current version")
    with open(path_tracker_path_ollama, "r") as f:
        current_version = int(f.read().strip())

    print("Incrementing version for the run that is about to start")
    new_version = current_version + 1

    print("Writing back the new version in the tracker_path_ollama file")
    with open(path_tracker_path_ollama, "w") as f:
        f.write(str(new_version))

    print("Setting tracker_path_ollama to new value")
    tracker_path_ollama = str(new_version)
    print("tracker_path_ollama has been set to", tracker_path_ollama)

    print("Constructing path_ollama")
    path_ollama = path_models + "/ollama_" + tracker_path_ollama
    print("path_ollama has been set to", path_ollama)
