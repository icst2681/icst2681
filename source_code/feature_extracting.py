import pandas as pd
# import torch
# from transformers import AutoModel, AutoTokenizer
# import os
from tqdm import tqdm
import ollama, csv, json, pathlib
from typing import List, Any
import common
import concurrent.futures as cf

def _flatten_json(obj: Any, parent: str = "") -> List[str]:
    if isinstance(obj, dict):
        items = []
        for k, v in obj.items():
            items.extend(_flatten_json(v, f"{parent}.{k}" if parent else k))
        return items
    elif isinstance(obj, list):
        items = []
        for i, v in enumerate(obj):
            items.extend(_flatten_json(v, f"{parent}[{i}]"))
        return items
    else:
        return [f"{parent} = {obj}"]
    
def record_to_text(json_str: str) -> str:
    obj = json.loads(json_str)
    return ". ".join(_flatten_json(obj))

def build_embedding_dataframe(
    csv_path: str,
    ollama_model: str,
    cache_path: str | None = None,
) -> pd.DataFrame:

    csv_path   = pathlib.Path(csv_path)
    if csv_path.exists() == False:
        return None

    cache_path = pathlib.Path(cache_path) if cache_path else None

    if cache_path and cache_path.exists():
        print("Loading cached embeddings from", cache_path)
        return pd.read_parquet(cache_path)

    models = ollama.list()
    try:
        have = {m["name"] for m in models}
    except (TypeError, KeyError):
        have = {m[0] for m in models}

    if ollama_model not in have:
        print(f"[setup] pulling {ollama_model} …")
        ollama.pull(model=ollama_model)

    def _embed_single(text: str) -> list[float]:
        return ollama.embeddings(
            model   = ollama_model,
            prompt  = text)["embedding"]
    
    rows = []

    with cf.ThreadPoolExecutor(max_workers=common.threads_embeddings) as pool, \
        csv_path.open() as f:
        reader = csv.DictReader(f)
        total = sum(1 for _ in csv_path.open()) - 1
        pbar = tqdm(total=total, desc=f"Embedding {ollama_model}")
        running = set()

        for row in reader:
            text = record_to_text(row["datapoint"])
            fut  = pool.submit(_embed_single, text)
            running.add(fut)

            if len(running) >= common.threads_embeddings:
                done, running = cf.wait(running, return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    rows.append(fut.result())
                pbar.update(len(done))

        for fut in cf.as_completed(running):
            rows.append(fut.result())
            pbar.update(1)

        pbar.close()

    dim = len(rows[0])
    df = pd.DataFrame(rows,
                    columns=[f"e{i}" for i in range(dim)],
                    dtype="float")

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        print("Saved embeddings at", cache_path)

    return df