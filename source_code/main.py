import feature_extracting
import modeling
import mutating
import common
import llm_dataset_preparing
import os, pathlib
import consolidating
import time

def get_predictions():

    for model, _ in common.MODEL_LIMITS.items():
        csv_in = common.path_ollama + "/" + model.replace(":", "_") + ".csv"
        cache_out = common.path_ollama + "/" + model.replace(":", "_") + "_emb.parquet"

        path_predictions = os.path.splitext(csv_in)[0] + "_predictions.csv"
        if pathlib.Path(path_predictions).exists():
            print(path_predictions, "exists. Skipping for model", model)
            continue
        
        emb_df = feature_extracting.build_embedding_dataframe(
            csv_path     = csv_in,
            ollama_model = model,
            cache_path   = cache_out,
        )

        if emb_df is None:
            continue

        modeling.llm_modeling(
                        embedding_df = emb_df,
                        csv_in_path  = csv_in,
                        path_models  = common.path_ollama,
                        model_tag    = model.replace(':', '_'),
                    )

if __name__ == "__main__":

    while True:
        print("\nPerforming the important step of setting up path_ollama...")
        common.set_path_ollama()

        print("\nJoining based on IMSI and timestamp...")
        fused_data = common.get_fused_data()

        print("\nFiltering fused records based on defined criterion...")
        common.get_fused_filtered_data()

        print("\nCreating Anomalies...")
        mutating.mutate(1)

        print("\nPreparing datasets for LLMs...")
        llm_dataset_preparing.prepare_llm_datasets()

        print("\nGetting predictions...")
        get_predictions()

        print("\nConsolidating previous and new predictions...")
        consolidating.consolidate(1)
        consolidating.consolidate(0)

        print("\nConsolidating finally to remove any mismatches...")
        consolidating.consolidate_final(1)
        consolidating.consolidate_final(0)

        print("\nSleeping for 5 minutes...")
        time.sleep(5 * 60)
        print("Waking up...\n")