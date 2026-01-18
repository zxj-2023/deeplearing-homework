import argparse
import json
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from src.config import LinearRAGConfig
from src.LinearRAG import LinearRAG
import os
import warnings
from src.evaluate import Evaluator
from src.utils import LLM_Model
from src.utils import setup_logging
from datetime import datetime

# 固定使用指定 GPU；如需改变设备策略可修改或移除。
#os.environ["CUDA_VISIBLE_DEVICES"] = "4"
warnings.filterwarnings('ignore')

def get_config():
    # 直接在代码内配置参数，不依赖命令行输入。
    # 模型与数据集路径配置。
    spacy_model = "en_core_web_trf"  # spaCy 模型名称。
    embedding_model = r"D:\code\python\lineargraph\LinearRAG\model\all-mpnet-base-v2"  # 句向量模型本地绝对路径。
    dataset_name = "2wikimultihop"  # 数据集目录名，对应 dataset/<name>/。
    llm_model = "qwen-flash"  # LLM 模型名称，用于生成答案或评测。
    # 运行与检索超参。
    max_workers = 16  # 并行工作线程数。
    max_iterations = 3  # 最大迭代轮数。
    iteration_threshold = 0.4  # 迭代停止阈值。
    passage_ratio = 2  # 候选段落比例。
    top_k_sentence = 3  # 每次检索保留的前 K 句子数量。
    use_vectorized_retrieval = False  # 是否启用向量化矩阵检索（GPU 友好）。
    return {
        "spacy_model": spacy_model,
        "embedding_model": embedding_model,
        "dataset_name": dataset_name,
        "llm_model": llm_model,
        "max_workers": max_workers,
        "max_iterations": max_iterations,
        "iteration_threshold": iteration_threshold,
        "passage_ratio": passage_ratio,
        "top_k_sentence": top_k_sentence,
        "use_vectorized_retrieval": use_vectorized_retrieval,
    }


def load_dataset(dataset_name): 
    # 读取问题与分块语料，并将段落加上索引前缀，便于后续定位原始块。
    questions_path = f"dataset/{dataset_name}/questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    chunks_path = f"dataset/{dataset_name}/chunks.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    # 以“索引:内容”的形式拼接，保持可追溯性。
    passages = [f'{idx}:{chunk}' for idx, chunk in enumerate(chunks)]
    return questions, passages

def load_embedding_model(embedding_model):
    # 在 GPU 上初始化句向量模型用于检索嵌入（SentenceTransformer）。
    embedding_model = SentenceTransformer(embedding_model,device="cuda")
    return embedding_model

def main():
    # 为本次运行生成时间戳目录。
    time = datetime.now()
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    config = get_config()
    embedding_model = load_embedding_model(config["embedding_model"])
    questions,passages = load_dataset(config["dataset_name"])
    # 日志写入 results/<dataset>/<timestamp>/，便于对照每次运行。
    setup_logging(f"results/{config['dataset_name']}/{time_str}/log.txt")
    llm_model = LLM_Model(config["llm_model"])
    # 组装配置：决定检索迭代次数、阈值、passage 比例等关键超参。
    rag_config = LinearRAGConfig(
        dataset_name=config["dataset_name"],
        embedding_model=embedding_model,
        spacy_model=config["spacy_model"],
        max_workers=config["max_workers"],
        llm_model=llm_model,
        max_iterations=config["max_iterations"],
        iteration_threshold=config["iteration_threshold"],
        passage_ratio=config["passage_ratio"],
        top_k_sentence=config["top_k_sentence"],
        use_vectorized_retrieval=config["use_vectorized_retrieval"]
    )
    rag_model = LinearRAG(global_config=rag_config)
    # 构建检索索引并对所有问题执行 QA。
    rag_model.index(passages)
    questions = rag_model.qa(questions)
    # 保存预测结果，供评测或后处理使用。
    os.makedirs(f"results/{config['dataset_name']}/{time_str}", exist_ok=True)
    with open(f"results/{config['dataset_name']}/{time_str}/predictions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=4)
    # 触发评测流程，输出指标与日志。
    evaluator = Evaluator(llm_model=llm_model, predictions_path=f"results/{config['dataset_name']}/{time_str}/predictions.json")
    evaluator.evaluate(max_workers=config["max_workers"])
if __name__ == "__main__":
    main()
