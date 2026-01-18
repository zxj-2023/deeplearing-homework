import json
import os
import time
from datetime import datetime

import torch
from sentence_transformers import SentenceTransformer

from src.config import LinearRAGConfig
from src.LinearRAG import LinearRAG
from src.utils import setup_logging
import logging


def list_datasets(dataset_root):
    return [
        name
        for name in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, name)) and not name.startswith(".")
    ]


def load_passages(dataset_root, dataset_name):
    chunks_path = os.path.join(dataset_root, dataset_name, "chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    # Keep an index prefix so adjacent passage links remain stable.
    return [f"{idx}:{chunk}" for idx, chunk in enumerate(chunks)]


def load_embedding_model(embedding_model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(embedding_model_path, device=device)


def main():
    dataset_root = "dataset"
    embedding_model_path = "model/all-mpnet-base-v2"
    spacy_model = "en_core_web_trf"
    working_dir = "./import"
    batch_size = 128
    max_workers = 16
    use_vectorized_retrieval = False

    datasets = list_datasets(dataset_root)
    if not datasets:
        raise RuntimeError(f"No datasets found under {dataset_root}")

    embedding_model = load_embedding_model(embedding_model_path)
    os.makedirs("results", exist_ok=True)
    setup_logging("results/index_build.log")
    logger = logging.getLogger(__name__)

    timings = []
    logger.info("Found %d datasets under %s", len(datasets), dataset_root)
    for dataset_name in datasets:
        logger.info("Loading passages for dataset: %s", dataset_name)
        passages = load_passages(dataset_root, dataset_name)
        logger.info("Loaded %d passages for dataset: %s", len(passages), dataset_name)
        rag_config = LinearRAGConfig(
            dataset_name=dataset_name,
            embedding_model=embedding_model,
            llm_model=None,
            spacy_model=spacy_model,
            working_dir=working_dir,
            batch_size=batch_size,
            max_workers=max_workers,
            use_vectorized_retrieval=use_vectorized_retrieval,
        )
        logger.info("Building index for dataset: %s", dataset_name)
        rag_model = LinearRAG(global_config=rag_config)

        start = time.perf_counter()
        rag_model.index(passages)
        elapsed = time.perf_counter() - start
        logger.info("Finished index for dataset: %s in %.3fs", dataset_name, elapsed)

        record = {
            "dataset": dataset_name,
            "seconds": round(elapsed, 3),
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        timings.append(record)
        print(f"[{dataset_name}] index built in {record['seconds']}s")

    summary_path = os.path.join("results", "index_times.json")
    existing = []
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(existing + timings, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
