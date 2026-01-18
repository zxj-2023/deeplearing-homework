import json
import os
import time
from datetime import datetime

import torch
from sentence_transformers import SentenceTransformer

from src.config import LinearRAGConfig
from src.evaluate import Evaluator
from src.LinearRAG import LinearRAG
from src.utils import LLM_Model, setup_logging


def get_config():
    # 本次批量 QA + 评测的默认配置。
    return {
        "dataset_root": "dataset",
        "embedding_model": "model/all-mpnet-base-v2",
        "spacy_model": "en_core_web_trf",
        "llm_model": "qwen-flash",
        "working_dir": "./import",
        "batch_size": 128,
        "max_workers": 16,
        "use_vectorized_retrieval": False,
    }


def list_datasets(dataset_root):
    # 枚举数据集目录，跳过隐藏文件夹。
    return [
        name
        for name in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, name)) and not name.startswith(".")
    ]


def load_dataset(dataset_root, dataset_name):
    # 读取问题列表和 chunk 文本。
    questions_path = os.path.join(dataset_root, dataset_name, "questions.json")
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    chunks_path = os.path.join(dataset_root, dataset_name, "chunks.json")
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    # 保留 chunk 索引前缀，便于相邻段落链接稳定。
    passages = [f"{idx}:{chunk}" for idx, chunk in enumerate(chunks)]
    return questions, passages


def load_embedding_model(embedding_model_path):
    # 自动选择设备并加载向量模型。
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(embedding_model_path, device=device)


def main():
    # 基于已有索引，对所有数据集执行 QA + 评测。
    config = get_config()
    datasets = list_datasets(config["dataset_root"])
    if not datasets:
        raise RuntimeError(f"No datasets found under {config['dataset_root']}")

    # 共享资源初始化（日志、向量模型、LLM）。
    os.makedirs("results", exist_ok=True)
    setup_logging("results/log_qa_eval.txt")
    embedding_model = load_embedding_model(config["embedding_model"])
    llm_model = LLM_Model(config["llm_model"])

    summary = []
    for dataset_name in datasets:
        if dataset_name == "2wikimultihop":
            continue
        # 加载该数据集的输入。
        questions, passages = load_dataset(config["dataset_root"], dataset_name)
        # 复用已有索引（working_dir 下），不再调用 index()。
        rag_config = LinearRAGConfig(
            dataset_name=dataset_name,
            embedding_model=embedding_model,
            llm_model=llm_model,
            spacy_model=config["spacy_model"],
            working_dir=config["working_dir"],
            batch_size=config["batch_size"],
            max_workers=config["max_workers"],
            use_vectorized_retrieval=config["use_vectorized_retrieval"],
        )
        rag_model = LinearRAG(global_config=rag_config)

        # 每次运行创建独立输出目录。
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join("results", dataset_name, time_str)
        os.makedirs(output_dir, exist_ok=True)

        # 重新构建索引（会覆盖内存中的图结构）。
        rag_model.index(passages)
        index_seconds = 0.0

        # 运行 QA（检索 + 生成）并收集预测。
        start_qa = time.perf_counter()
        qa_results = rag_model.qa(questions)
        qa_seconds = time.perf_counter() - start_qa

        predictions_path = os.path.join(output_dir, "predictions.json")
        with open(predictions_path, "w", encoding="utf-8") as f:
            json.dump(qa_results, f, ensure_ascii=False, indent=4)

        # 评测预测结果（LLM accuracy + contain accuracy）。
        evaluator = Evaluator(llm_model=llm_model, predictions_path=predictions_path)
        start_eval = time.perf_counter()
        llm_accuracy, contain_accuracy = evaluator.evaluate(max_workers=config["max_workers"])
        eval_seconds = time.perf_counter() - start_eval

        # 记录该数据集的耗时与准确率。
        summary.append({
            "dataset": dataset_name,
            "predictions_path": predictions_path,
            "llm_accuracy": round(llm_accuracy, 4),
            "contain_accuracy": round(contain_accuracy, 4),
            "index_seconds": round(index_seconds, 3),
            "qa_seconds": round(qa_seconds, 3),
            "eval_seconds": round(eval_seconds, 3),
            "started_at": time_str,
        })

    # 汇总结果写入到 results/qa_eval_summary.json。
    summary_path = os.path.join("results", "qa_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
