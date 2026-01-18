# 2wikimultihop 导入产物说明

该目录是 `dataset/2wikimultihop` 构建索引后的产物，用于加速后续检索与图推理。

## 文件说明

- `passage_embedding.parquet`
  - 段落（passage）向量库缓存，包含 `hash_id / text / embedding`。
  - 用于密集检索与计算段落相似度。
- `sentence_embedding.parquet`
  - 句子向量库缓存，来源于 NER 过程中抽取的句子。
  - 用于“实体 ↔ 句子”传播与相似度计算。
- `entity_embedding.parquet`
  - 实体向量库缓存，包含所有识别到的实体。
  - 用于从问题实体出发的图检索与权重传播。
- `ner_results.json`
  - NER 结果缓存，包含：
    - `passage_hash_id_to_entities`：每个 passage 的实体集合
    - `sentence_to_entities`：每个句子的实体集合
  - 用于避免重复 NER 计算。
- `LinearRAG.graphml`
  - 构建完成的图结构文件（igraph GraphML）。
  - 节点包含 passage/entity，边包含 passage-entity 关联与相邻 passage 连接。

## 生成来源

这些文件由 `LinearRAG.index()` 在 `working_dir=./import` 下生成。若删除，会在下次构建索引时重新生成。
