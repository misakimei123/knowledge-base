"""
混合检索编排引擎 - RAG+KG 融合
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """混合检索结果"""
    vector_results: List[Dict[str, Any]]
    kg_results: List[Dict[str, Any]]
    merged_results: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class HybridRAGEngine:
    """RAG+KG 混合检索编排器"""
    
    def __init__(self, milvus_client=None, neo4j_client=None, config: Dict = None):
        """
        初始化混合检索引擎
        :param milvus_client: Milvus 客户端实例
        :param neo4j_client: Neo4j 客户端实例
        :param config: 配置字典
        """
        self.milvus = milvus_client
        self.neo4j = neo4j_client
        self.config = config or {}
        
        # 检索参数
        self.vector_top_k = config.get("vector_top_k", 10) if config else 10
        self.kg_top_k = config.get("kg_top_k", 5) if config else 5
        self.final_top_k = config.get("final_top_k", 8) if config else 8
        self.rrf_k = config.get("rrf_k", 60) if config else 60
        self.rerank_enable = config.get("rerank_enable", False) if config else False
    
    async def retrieve(
        self, 
        query: str, 
        category_filter: Optional[List[str]] = None
    ) -> HybridResult:
        """
        执行混合检索，返回融合结果
        :param query: 查询文本
        :param category_filter: 分类过滤列表
        :return: 混合检索结果
        """
        # 1. 并行执行向量 + 图谱检索
        vector_task = asyncio.create_task(
            self._vector_retrieve(query, category_filter)
        )
        kg_task = asyncio.create_task(
            self._kg_retrieve(query, category_filter)
        )
        
        vector_results, kg_results = await asyncio.gather(
            vector_task, 
            kg_task,
            return_exceptions=True
        )
        
        # 处理异常
        if isinstance(vector_results, Exception):
            logger.error(f"Vector retrieval failed: {vector_results}")
            vector_results = []
        
        if isinstance(kg_results, Exception):
            logger.error(f"KG retrieval failed: {kg_results}")
            kg_results = []
        
        # 2. RRF 融合（Reciprocal Rank Fusion）
        merged = self._reciprocal_rank_fusion(
            vector_results,
            kg_results,
            k=self.rrf_k
        )
        
        # 3. 可选：重排序
        if self.rerank_enable and merged:
            merged = await self._rerank(query, merged)
        
        return HybridResult(
            vector_results=vector_results,
            kg_results=kg_results,
            merged_results=merged[:self.final_top_k],
            metadata={
                "vector_count": len(vector_results),
                "kg_count": len(kg_results),
                "fusion_strategy": "rrf",
                "category_filter": category_filter
            }
        )
    
    async def _vector_retrieve(
        self, 
        query: str, 
        category_filter: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """向量检索"""
        if not self.milvus:
            logger.warning("Milvus client not initialized")
            return []
        
        # TODO: 调用 Milvus 搜索
        # filter_expr = self._build_category_filter(category_filter)
        # results = await self.milvus.search(query, top_k=self.vector_top_k, filter=filter_expr)
        
        logger.debug(f"Vector search for: {query[:50]}...")
        return []
    
    async def _kg_retrieve(
        self, 
        query: str, 
        category_filter: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """知识图谱检索"""
        if not self.neo4j:
            logger.warning("Neo4j client not initialized")
            return []
        
        # TODO: 调用 Neo4j 搜索
        # results = await self.neo4j.retrieve(query, category_filter, top_k=self.kg_top_k)
        
        logger.debug(f"KG search for: {query[:50]}...")
        return []
    
    def _build_category_filter(self, category_list: Optional[List[str]]) -> str:
        """构建 Milvus 标量过滤表达式"""
        if not category_list:
            return ""
        
        # 支持分类树继承：查询"HR"自动包含"HR/Policy", "HR/Recruitment"
        expanded = self._expand_category_tree(category_list)
        
        if len(expanded) == 1:
            return f'category == "{expanded[0]}"'
        else:
            categories_str = ", ".join(f'"{c}"' for c in expanded)
            return f"category in [{categories_str}]"
    
    def _expand_category_tree(self, categories: List[str]) -> List[str]:
        """扩展分类树（支持子分类继承）"""
        expanded = set()
        
        # TODO: 从元数据数据库加载分类树
        # 简单实现：直接返回原分类
        for cat in categories:
            expanded.add(cat)
            # 生产环境：递归添加子分类
        
        return list(expanded)
    
    def _reciprocal_rank_fusion(
        self, 
        vector_results: List[Dict], 
        kg_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """
        RRF 融合算法
        :param vector_results: 向量检索结果
        :param kg_results: KG 检索结果
        :param k: 平滑参数
        :return: 融合后的结果
        """
        score_map = {}
        
        # 向量结果打分
        for rank, item in enumerate(vector_results):
            doc_id = item.get("id", item.get("doc_id"))
            if doc_id not in score_map:
                score_map[doc_id] = {"item": item, "score": 0, "sources": []}
            score_map[doc_id]["score"] += 1.0 / (k + rank + 1)
            score_map[doc_id]["sources"].append("vector")
        
        # KG 结果打分
        for rank, item in enumerate(kg_results):
            doc_id = item.get("id", item.get("entity_id"))
            if doc_id not in score_map:
                score_map[doc_id] = {"item": item, "score": 0, "sources": []}
            score_map[doc_id]["score"] += 1.0 / (k + rank + 1)
            score_map[doc_id]["sources"].append("kg")
        
        # 按分数排序
        merged = sorted(
            score_map.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        # 添加融合分数到结果
        for item in merged:
            item["item"]["fusion_score"] = round(item["score"], 4)
            item["item"]["sources"] = item["sources"]
        
        return [item["item"] for item in merged]
    
    async def _rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """使用 BGE-Reranker 重排序"""
        # TODO: 实现重排序
        # from sentence_transformers import CrossEncoder
        # passages = [r.get("text", r.get("content", "")) for r in results]
        # pairs = [[query, p] for p in passages]
        # scores = self.rerank_model.predict(pairs)
        # ...
        
        logger.debug(f"Reranking {len(results)} results...")
        return results  # 临时直接返回


# 向量检索器封装
class MilvusVectorRetriever:
    """Milvus 向量检索器"""
    
    def __init__(self, collection, embed_model, similarity_top_k: int = 10):
        self.collection = collection
        self.embed_model = embed_model
        self.similarity_top_k = similarity_top_k
    
    async def aretrieve(self, query: str, filters: Optional[str] = None) -> List[Dict]:
        """异步检索"""
        # TODO: 实现
        return []


# KG 检索器封装  
class Neo4jKGRetriever:
    """Neo4j 图谱检索器"""
    
    def __init__(self, driver, entity_extractor, query_template: str, max_hops: int = 2):
        self.driver = driver
        self.entity_extractor = entity_extractor
        self.query_template = query_template
        self.max_hops = max_hops
    
    async def aretrieve(self, query: str, filters: Optional[List[str]] = None) -> List[Dict]:
        """异步检索"""
        # TODO: 实现
        return []
