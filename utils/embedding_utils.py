"""
Embedding 工具函数 - 批量调用 + 缓存
"""
import hashlib
import logging
from typing import List, Dict, Optional
from functools import lru_cache

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """Embedding 缓存（LRU 策略）"""
    
    def __init__(self, max_size: int = 10000):
        """
        初始化缓存
        :param max_size: 最大缓存条目数
        """
        self.max_size = max_size
        self._cache: Dict[str, List[float]] = {}
    
    def _compute_key(self, text: str) -> str:
        """计算文本哈希作为缓存键"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """获取缓存的 embedding"""
        key = self._compute_key(text)
        return self._cache.get(key)
    
    def set(self, text: str, embedding: List[float]):
        """设置缓存"""
        key = self._compute_key(text)
        
        # LRU：如果满了，删除最旧的
        if len(self._cache) >= self.max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = embedding
    
    def clear(self):
        """清空缓存"""
        self._cache.clear()
    
    def size(self) -> int:
        """返回缓存大小"""
        return len(self._cache)


# 全局缓存实例
_embedding_cache: Optional[EmbeddingCache] = None


def get_embedding_cache(max_size: int = 10000) -> EmbeddingCache:
    """获取全局缓存单例"""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache(max_size)
    return _embedding_cache


async def batch_embed(
    texts: List[str],
    model_name: str = "BAAI/bge-m3",
    use_cache: bool = True,
    batch_size: int = 32
) -> List[List[float]]:
    """
    批量生成 Embedding
    :param texts: 文本列表
    :param model_name: 模型名称
    :param use_cache: 是否使用缓存
    :param batch_size: 批次大小
    :return: Embedding 列表
    """
    from sentence_transformers import SentenceTransformer
    
    cache = get_embedding_cache() if use_cache else None
    
    # 检查缓存
    embeddings = []
    uncached_texts = []
    uncached_indices = []
    
    if cache:
        for i, text in enumerate(texts):
            emb = cache.get(text)
            if emb is not None:
                embeddings.append((i, emb))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
    
    # 加载模型并生成 embedding
    if uncached_texts:
        logger.info(f"Generating embeddings for {len(uncached_texts)} texts")
        
        # TODO: 异步加载模型
        model = SentenceTransformer(model_name)
        
        # 分批处理
        all_new_embeddings = []
        for start in range(0, len(uncached_texts), batch_size):
            end = min(start + batch_size, len(uncached_texts))
            batch = uncached_texts[start:end]
            
            batch_embeddings = model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            all_new_embeddings.extend(batch_embeddings.tolist())
        
        # 填充结果并更新缓存
        result = [None] * len(texts)
        
        # 填充缓存命中的
        for idx, emb in embeddings:
            result[idx] = emb
        
        # 填充新生成的
        for i, (orig_idx, emb) in enumerate(zip(uncached_indices, all_new_embeddings)):
            result[orig_idx] = emb
            if cache:
                cache.set(uncached_texts[i], emb)
        
        return result
    
    # 全部命中缓存
    return [emb for _, emb in sorted(embeddings, key=lambda x: x[0])]


@lru_cache(maxsize=1000)
def embed_single_cached(text: str, model_name: str = "BAAI/bge-m3") -> List[float]:
    """
    单个文本 embedding（带 LRU 缓存）
    使用 Python 内置的 lru_cache
    """
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer(model_name)
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()
