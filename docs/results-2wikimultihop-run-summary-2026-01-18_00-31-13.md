# 2wikimultihop 生成结果解读

## 概览

- 样本数量：1000
- 每条样本字段：`question`、`sorted_passage`、`sorted_passage_scores`、`pred_answer`、`gold_answer`、`llm_accuracy`、`contain_accuracy`
- 平均检索段落数：5.0

## 准确率摘要

| 指标 | 数值 |
| --- | --- |
| LLM Accuracy（平均） | 0.653 |
| Contain Accuracy（平均） | 0.738 |
| LLM 判定正确数 | 653 / 1000 |
| Contain 判定正确数 | 738 / 1000 |

## 答案输出特征

| 指标 | 数值 |
| --- | --- |
| 完全匹配数（pred == gold，忽略大小写与空白） | 586 |
| 预测答案平均长度（字符） | 21.2 |
| 预测答案长度中位数（字符） | 16 |

## 文件位置

- `results/2wikimultihop/2026-01-18_00-31-13/predictions.json`
## 字段含义

- `question`：输入问题文本。
- `sorted_passage`：检索到的 passage 列表（按相关性排序）。
- `sorted_passage_scores`：与 `sorted_passage` 对应的相关性分数。
- `pred_answer`：模型生成的答案。
- `gold_answer`：数据集提供的标准答案。
- `llm_accuracy`：由 LLM 评判该样本是否正确（1/0）。
- `contain_accuracy`：基于字符串包含关系的快速判定（1/0）。
## run_acc.py 的检索设置说明

当前使用 LinearRAG 默认配置（`src/config.py`），检索方式为 BFS 迭代图检索。

- `use_vectorized_retrieval=False`：使用 BFS 迭代检索（非矩阵化向量检索）。
- `retrieval_top_k=5`：每个问题返回 5 个候选 passage。
- `max_iterations=3`：图传播最大迭代轮数。
- `top_k_sentence=1`：每轮仅保留相似度最高的 1 个句子参与传播。
- `iteration_threshold=0.5`：低于阈值的传播会被剪枝。
- `passage_ratio=1.5`：融合 dense 检索分数与图得分的比例系数。
- `passage_node_weight=0.05`：passage 节点权重缩放。
- `damping=0.5`：PPR 阻尼系数。
- `batch_size=128`：向量编码批量大小。
- `max_workers=16`：并行线程数。
## 准确率计算方式

准确率在 `src/evaluate.py` 中计算，包含两种指标：

- **LLM Accuracy**：将 `pred_answer` 与 `gold_answer` 交给 LLM 判断，只允许返回 `correct/incorrect`。返回 `correct` 记 1，否则记 0，最后取平均值。
- **Contain Accuracy**：对答案做归一化（小写、去标点、去冠词、压缩空白），判断 `gold_answer` 是否被 `pred_answer` 包含；包含记 1，否则记 0，最后取平均值。

评测会把每条样本的结果写回 `predictions.json`，并在同目录生成 `evaluation_results.json`。
