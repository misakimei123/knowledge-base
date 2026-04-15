"""
文档处理流水线 - 解析→分块→向量化→图谱构建
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """文档处理结果"""
    success: bool
    doc_id: str
    vector_count: int = 0
    entity_count: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentPipeline:
    """文档入库流水线（解析→分块→向量化→图谱构建）"""
    
    def __init__(self, config: Dict = None):
        """
        初始化文档处理流水线
        :param config: 配置字典
        """
        self.config = config or {}
        self.chunk_size = config.get("chunk_size", 512) if config else 512
        self.chunk_overlap = config.get("chunk_overlap", 50) if config else 50
        
        # 延迟初始化组件
        self._parser = None
        self._entity_extractor = None
    
    async def process(
        self, 
        doc_id: str, 
        file_path: str, 
        category: str,
        visible_roles: Optional[List[str]] = None
    ) -> ProcessingResult:
        """
        异步处理单个文档
        :param doc_id: 文档 ID
        :param file_path: 文件路径
        :param category: 文档分类
        :param visible_roles: 可见角色列表
        :return: 处理结果
        """
        try:
            logger.info(f"Processing document: {doc_id} ({category})")
            
            # 1. 解析文档
            documents = await self._parse_document(file_path)
            
            if not documents:
                return ProcessingResult(
                    success=False,
                    doc_id=doc_id,
                    error="Failed to parse document"
                )
            
            # 2. 语义分块
            nodes = self._split_documents(documents)
            
            # 3. 写入 Milvus（批量）
            milvus_ids = await self._batch_insert_milvus(
                nodes, doc_id, category, visible_roles
            )
            
            # 4. 实体抽取 + 写入 Neo4j
            entities = await self._extract_entities(documents, category)
            kg_ids = await self._batch_insert_neo4j(entities, doc_id, category)
            
            logger.info(
                f"Document {doc_id} processed: "
                f"{len(milvus_ids)} vectors, {len(kg_ids)} entities"
            )
            
            return ProcessingResult(
                success=True,
                doc_id=doc_id,
                vector_count=len(milvus_ids),
                entity_count=len(kg_ids),
                metadata={
                    "category": category,
                    "chunk_count": len(nodes),
                    "file_path": file_path
                }
            )
            
        except Exception as e:
            logger.error(f"Process doc {doc_id} failed: {e}", exc_info=True)
            return ProcessingResult(
                success=False,
                doc_id=doc_id,
                error=str(e)
            )
    
    async def process_batch(
        self,
        documents: List[Dict[str, str]],
        max_concurrent: int = 5
    ) -> List[ProcessingResult]:
        """
        批量处理文档
        :param documents: 文档列表 [{"doc_id": ..., "file_path": ..., "category": ...}]
        :param max_concurrent: 最大并发数
        :return: 处理结果列表
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(doc):
            async with semaphore:
                return await self.process(
                    doc_id=doc["doc_id"],
                    file_path=doc["file_path"],
                    category=doc.get("category", "general"),
                    visible_roles=doc.get("visible_roles")
                )
        
        tasks = [process_with_semaphore(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 转换异常为 ProcessingResult
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ProcessingResult(
                    success=False,
                    doc_id=documents[i]["doc_id"],
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _parse_document(self, file_path: str) -> List[Any]:
        """解析文档"""
        # TODO: 使用 LlamaIndex Readers + PyPDF2
        # from llama_index import SimpleDirectoryReader
        # reader = SimpleDirectoryReader(input_files=[file_path])
        # documents = reader.load_data()
        
        logger.debug(f"Parsing document: {file_path}")
        return [{"text": "Sample content", "metadata": {"source": file_path}}]
    
    def _split_documents(self, documents: List[Any]) -> List[Any]:
        """语义分块"""
        # TODO: 使用 SemanticSplitterNodeParser
        # node_parser = SemanticSplitterNodeParser(
        #     chunk_size=self.chunk_size,
        #     chunk_overlap=self.chunk_overlap,
        #     embed_model=self.config.get("embedding_model")
        # )
        # nodes = node_parser.get_nodes_from_documents(documents)
        
        nodes = []
        for doc in documents:
            text = doc.get("text", "")
            # 简单按长度分块
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i:i + self.chunk_size]
                nodes.append({
                    "text": chunk,
                    "metadata": doc.get("metadata", {})
                })
        
        return nodes
    
    async def _batch_insert_milvus(
        self,
        nodes: List[Dict],
        doc_id: str,
        category: str,
        visible_roles: Optional[List[str]]
    ) -> List[str]:
        """批量写入 Milvus"""
        # TODO: 调用 Milvus 客户端批量插入
        # embeddings = [node.get("embedding") for node in nodes]
        # metadata = [
        #     {
        #         "doc_id": doc_id,
        #         "category": category,
        #         "chunk_text": node["text"][:2000],
        #         "visible_roles": visible_roles or ["all"]
        #     }
        #     for node in nodes
        # ]
        # ids = await self.milvus_client.batch_insert(embeddings, metadata)
        
        logger.debug(f"Inserting {len(nodes)} chunks to Milvus")
        return [f"vec_{i}" for i in range(len(nodes))]
    
    async def _extract_entities(
        self,
        documents: List[Any],
        category: str
    ) -> List[Dict]:
        """实体抽取"""
        # TODO: 使用 NER 模型或规则抽取
        from .kg_engine import SimpleEntityExtractor
        
        extractor = SimpleEntityExtractor()
        all_entities = []
        
        for doc in documents:
            text = doc.get("text", "")
            entities = await extractor.extract(text, category)
            
            for entity in entities:
                all_entities.append({
                    "id": f"{category}_{entity['text']}_{hash(entity['text']) % 10000}",
                    "type": entity["type"],
                    "properties": {"text": entity["text"]},
                    "doc_id": doc.get("metadata", {}).get("source", ""),
                    "category": category
                })
        
        return all_entities
    
    async def _batch_insert_neo4j(
        self,
        entities: List[Dict],
        doc_id: str,
        category: str
    ) -> List[str]:
        """批量写入 Neo4j"""
        # TODO: 调用 Neo4j 客户端批量插入
        # ids = await self.neo4j_client.batch_insert_entities(entities)
        
        logger.debug(f"Inserting {len(entities)} entities to Neo4j")
        return [f"ent_{i}" for i in range(len(entities))]


# 工具函数
def get_category_roles(category: str) -> List[str]:
    """获取分类对应的可见角色"""
    # TODO: 从元数据数据库加载
    role_mapping = {
        "HR": ["hr_admin", "manager", "employee"],
        "HR_Policy": ["hr_admin", "manager"],
        "Finance": ["finance_admin", "manager"],
        "Technical": ["engineer", "manager"],
        "General": ["all"]
    }
    return role_mapping.get(category, ["all"])
