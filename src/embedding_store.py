from copy import deepcopy
from src.utils import compute_mdhash_id
import numpy as np
import pandas as pd
import os

class EmbeddingStore:
    def __init__(self, embedding_model, db_filename, batch_size, namespace):
        # embedding_model: 负责将文本编码为向量的模型（如 SentenceTransformer）。
        # db_filename: 向量库存储路径（Parquet 文件）。
        # batch_size: 批量编码大小，用于控制显存/内存占用。
        # namespace: 不同用途/数据集的命名空间前缀，避免 hash 冲突。
        self.embedding_model = embedding_model
        self.db_filename = db_filename
        self.batch_size = batch_size
        self.namespace = namespace
        
        # 内存中的主数据结构与索引映射。
        self.hash_ids = []
        self.texts = []
        self.embeddings = []
        self.hash_id_to_text = {}
        self.hash_id_to_idx = {}
        self.text_to_hash_id = {}
        
        # 初始化时尝试从磁盘加载已有向量库。
        self._load_data()
    
    def _load_data(self):
        # 如果 parquet 文件存在，则加载并重建索引映射。
        if os.path.exists(self.db_filename):
            df = pd.read_parquet(self.db_filename)
            self.hash_ids = df["hash_id"].values.tolist()
            self.texts = df["text"].values.tolist()
            self.embeddings = df["embedding"].values.tolist()
            
            # 重建 hash_id 与索引/文本的快速映射表。
            self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
            self.hash_id_to_text = {h: t for h, t in zip(self.hash_ids, self.texts)}
            self.text_to_hash_id = {t: h for t, h in zip(self.texts, self.hash_ids)}
            print(f"[{self.namespace}] Loaded {len(self.hash_ids)} records from {self.db_filename}")
        
    def insert_text(self, text_list):
        # 计算每段文本的 hash_id，用于去重与定位。
        nodes_dict = {}
        for text in text_list:
            nodes_dict[compute_mdhash_id(text, prefix=self.namespace + "-")] = {'content': text}
        
        all_hash_ids = list(nodes_dict.keys())
        
        # 仅对未存在的文本进行编码，避免重复计算。
        existing = set(self.hash_ids)
        missing_ids = [h for h in all_hash_ids if h not in existing]      
        texts_to_encode = [nodes_dict[hash_id]["content"] for hash_id in missing_ids]
        # 批量编码并做向量归一化，便于后续余弦相似度检索。
        all_embeddings = self.embedding_model.encode(texts_to_encode,normalize_embeddings=True, show_progress_bar=False,batch_size=self.batch_size)
        
        # 写入内存并持久化到磁盘。
        self._upsert(missing_ids, texts_to_encode, all_embeddings)

    def _upsert(self, hash_ids, texts, embeddings):
        # 将新增数据合并到内存列表。
        self.hash_ids.extend(hash_ids)
        self.texts.extend(texts)
        self.embeddings.extend(embeddings)
        
        # 更新索引映射，保证检索一致性。
        self.hash_id_to_idx = {h: idx for idx, h in enumerate(self.hash_ids)}
        self.hash_id_to_text = {h: t for h, t in zip(self.hash_ids, self.texts)}
        self.text_to_hash_id = {t: h for t, h in zip(self.texts, self.hash_ids)}
        
        # 落盘保存，避免内存数据丢失。
        self._save_data()

    def _save_data(self):
        # 序列化为 Parquet，便于后续快速加载与跨进程使用。
        data_to_save = pd.DataFrame({
            "hash_id": self.hash_ids,
            "text": self.texts,
            "embedding": self.embeddings
        })
        os.makedirs(os.path.dirname(self.db_filename), exist_ok=True)
        data_to_save.to_parquet(self.db_filename, index=False)
      
    def get_hash_id_to_text(self):
        # 返回副本，避免外部修改内部状态。
        return deepcopy(self.hash_id_to_text)
    
    def encode_texts(self, texts):
        # 提供通用编码接口，统一归一化与批量大小设置。
        return self.embedding_model.encode(texts, normalize_embeddings=True, show_progress_bar=False, batch_size=self.batch_size)
    
    def get_embeddings(self, hash_ids):
        # 根据 hash_id 列表提取对应的向量矩阵。
        if not hash_ids:
            return np.array([])
        indices = np.array([self.hash_id_to_idx[h] for h in hash_ids], dtype=np.intp)
        embeddings = np.array(self.embeddings)[indices]
        return embeddings
