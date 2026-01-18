# LinearRAG: Linear Graph Retrieval-Augmented Generation on Large-scale Corpora

> 线性图检索增强生成（LinearRAG）复现实验。项目聚焦于“无关系图”的构建与多跳检索流程，适合用于课程作业复现与对比分析。

<p align="center">
  <a href="https://arxiv.org/abs/2510.10114" target="_blank">
    <img src="https://img.shields.io/badge/Paper-Arxiv-red?logo=arxiv&style=flat-square" alt="arXiv:2510.10114">
  </a>
  <a href="https://huggingface.co/datasets/Zly0523/linear-rag/tree/main" target="_blank">
    <img src="https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface&style=flat-square" alt="HuggingFace">
  </a>
  <a href="https://github.com/LuyaoZhuang/linear-rag" target="_blank">
    <img src="https://img.shields.io/badge/GitHub-Project-181717?logo=github&style=flat-square" alt="GitHub">
  </a>
</p>

---

## 🚀 Highlights
- **Relation-free Graph**：不依赖显式关系抽取，仅用 NER 与语义相似度构图。
- **Multi-hop Reasoning**：图传播驱动的多跳检索，减少复杂推理步骤。
- **Scalable**：索引构建不消耗 LLM tokens，速度快、成本低。

---

## 🛠 环境与依赖

### 1) 安装依赖
```bash
pip install -r requirements.txt
```

### 2) 安装 spaCy 模型
```bash
python -m spacy download en_core_web_trf
```

> `medical` 数据集可选安装 SciSpacy 模型：
```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_scibert-0.5.3.tar.gz
```

### 3) LLM 评测密钥
当前评测使用 DashScope 兼容接口，通过环境变量读取：
```bash
set DASHSCOPE_API_KEY=your-key
```

---

## 📦 数据与模型准备

### 1) 下载数据集
```bash
git clone https://huggingface.co/datasets/Zly0523/linear-rag
cp -r linear-rag/dataset/* dataset/
```

### 2) 下载 embedding 模型
```bash
python download_embedding_model.py
```

确保模型位于：
```
model/all-mpnet-base-v2/
```

---

## ⚡ 快速开始

### 构建索引（所有数据集）
```bash
python run_build_index_all.py
```

### 单数据集 QA + 评测
直接在 `run.py` 内配置参数后运行：
```bash
python run.py
```

### 多数据集 QA + 评测
```bash
python run_acc.py
```

> 说明：当前脚本使用“代码内配置”，不走命令行参数。请在 `run.py` / `run_acc.py` 中修改 `get_config()`。

---

## 📂 目录结构
- `src/`: LinearRAG 核心实现。
- `dataset/<name>/`: 输入数据（`questions.json`, `chunks.json`）。
- `import/`: 索引与嵌入缓存（`*.parquet`, `*.graphml`）。
- `results/`: 预测、评测、日志与统计。
- `model/`: 本地 embedding 模型。
- `docs/`: 实验记录与统计整理文档。

---

## 📊 输出结果
运行后输出位于 `results/`，典型文件：
- `predictions.json`: QA 输出（含评测字段）。
- `evaluation_results.json`: 评测统计。
- `log.txt` / `log_qa_eval.txt`: 日志。
