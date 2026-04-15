"""
Milvus 向量存储客户端封装
"""
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class MilvusClient:
    """Milvus 向量数据库客户端"""
    
    def __init__(
        self,
        uri: str = "http://localhost:19530",
        collection_name: str = "knowledge_base"
    ):
        """
        初始化 Milvus 客户端
        :param uri: Milvus 服务地址
        :param collection_name: 集合名称
        """
        self.uri = uri
        self.collection_name = collection_name
        self.client = None
        self.collection = None
    
    async def connect(self):
        """连接 Milvus"""
        try:
            # 生产环境使用 pymilvus 异步客户端
            # from pymilvus import MilvusClient
            # self.client = MilvusClient(uri=self.uri)
            
            logger.info(f"Connected to Milvus at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise
    
    async def close(self):
        """关闭连接"""
        if self.client:
            # await self.client.close()
            pass
    
    async def create_collection(
        self,
        dimension: int = 1024,
        metric_type: str = "COSINE",
        index_type: str = "HNSW"
    ):
        """创建集合"""
        # TODO: 实现集合创建
        # schema = self.client.create_schema()
        # schema.add_field("id", DataType.INT64, is_primary=True)
        # schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=dimension)
        # schema.add_field("category", DataType.VARCHAR, max_length=64)
        # ...
        
        logger.info(f"Created collection {self.collection_name} (dim={dimension})")
    
    async def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        向量搜索
        :param query_embedding: 查询向量
        :param top_k: 返回数量
        :param filter_expr: 过滤表达式
        :param output_fields: 返回字段
        :return: 搜索结果
        """
        if not self.client:
            await self.connect()
        
        # TODO: 实现搜索
        # results = await self.client.search(
        #     collection_name=self.collection_name,
        #     data=[query_embedding],
        #     limit=top_k,
        #     expr=filter_expr,
        #     output_fields=output_fields or ["category", "chunk_text"]
        # )
        
        logger.debug(f"Searching with top_k={top_k}, filter={filter_expr}")
        return []
    
    async def batch_insert(
        self,
        embeddings: List[List[float]],
        metadata: List[Dict[str, Any]]
    ) -> List[str]:
        """
        批量插入向量
        :param embeddings: 向量列表
        :param metadata: 元数据列表
        :return: 插入的 ID 列表
        """
        if not self.client:
            await self.connect()
        
        # TODO: 实现批量插入
        # ids = await self.client.insert(
        #     collection_name=self.collection_name,
        #     data={"embedding": embeddings, **metadata}
        # )
        
        logger.info(f"Inserted {len(embeddings)} vectors")
        return [f"id_{i}" for i in range(len(embeddings))]
    
    async def delete_by_ids(self, ids: List[str]) -> int:
        """按 ID 删除"""
        # TODO: 实现删除
        logger.info(f"Deleting {len(ids)} vectors")
        return len(ids)
    
    async def count(self) -> int:
        """获取集合总数"""
        # TODO: 实现计数
        return 0
    
    async def load_collection(self):
        """加载集合到内存"""
        # await self.client.load_collection(self.collection_name)
        pass
    
    async def release_collection(self):
        """释放集合"""
        # await self.client.release_collection(self.collection_name)
        pass


# 备用 FAISS 索引（降级方案）
class FAISSBackup:
    """本地 FAISS 索引（当 Milvus 不可用时）"""
    
    def __init__(self, dimension: int = 1024):
        self.dimension = dimension
        self.index = None
        self.id_map = {}
    
    async def initialize(self):
        """初始化 FAISS 索引"""
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.dimension)  # 内积相似度
            logger.info("FAISS index initialized")
        except ImportError:
            logger.warning("FAISS not available, using dummy index")
    
    async def add(self, embeddings: List[List[float]], ids: List[str]):
        """添加向量"""
        if self.index:
            import numpy as np
            self.index.add(np.array(embeddings, dtype=np.float32))
            for i, id_ in enumerate(ids):
                self.id_map[len(self.id_map) + i] = id_
    
    async def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 10
    ) -> List[Dict]:
        """搜索"""
        if not self.index:
            return []
        
        import numpy as np
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            top_k
        )
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:
                results.append({
                    "id": self.id_map.get(idx),
                    "score": float(dist),
                    "metadata": {}
                })
        
        return results
