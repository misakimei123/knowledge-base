"""
Neo4j 图数据库客户端封装
"""
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j 图数据库客户端"""
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str = "password"
    ):
        """
        初始化 Neo4j 客户端
        :param uri: Neo4j URI
        :param user: 用户名
        :param password: 密码
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
    
    async def connect(self):
        """连接 Neo4j"""
        try:
            # 生产环境使用 neo4j 官方异步驱动
            # from neo4j import AsyncGraphDatabase
            # self.driver = AsyncGraphDatabase.driver(
            #     self.uri, 
            #     auth=(self.user, self.password)
            # )
            
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    async def close(self):
        """关闭连接"""
        if self.driver:
            # await self.driver.close()
            pass
    
    async def execute_cypher(
        self, 
        cypher: str, 
        parameters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        执行 Cypher 查询
        :param cypher: Cypher 语句
        :param parameters: 参数
        :return: 查询结果
        """
        if not self.driver:
            await self.connect()
        
        # TODO: 实现异步执行
        # async with self.driver.session() as session:
        #     result = await session.run(cypher, parameters=parameters or {})
        #     return [record.data() for record in result]
        
        logger.debug(f"Executing Cypher: {cypher[:100]}...")
        return []
    
    async def create_constraints(self):
        """创建唯一约束"""
        constraints = [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT relation_id IF NOT EXISTS FOR ()-[r:RELATION]-() REQUIRE r.id IS UNIQUE"
        ]
        
        for constraint in constraints:
            await self.execute_cypher(constraint)
        
        logger.info("Created Neo4j constraints")
    
    async def create_indexes(self):
        """创建索引"""
        indexes = [
            "CREATE INDEX entity_category IF NOT EXISTS FOR (e:Entity) ON (e.category)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX relation_type IF NOT EXISTS FOR ()-[r:RELATION]-() ON type(r)"
        ]
        
        for index in indexes:
            await self.execute_cypher(index)
        
        logger.info("Created Neo4j indexes")
    
    async def insert_entity(
        self,
        entity_id: str,
        entity_type: str,
        properties: Dict[str, Any],
        doc_id: str,
        category: str
    ):
        """插入实体"""
        cypher = f"""
        MERGE (e:`{entity_type}` {{id: $entity_id}})
        SET e += $properties
        SET e.doc_id = $doc_id
        SET e.category = $category
        SET e.created_at = datetime()
        """
        
        await self.execute_cypher(cypher, {
            "entity_id": entity_id,
            "properties": properties,
            "doc_id": doc_id,
            "category": category
        })
    
    async def insert_relation(
        self,
        from_id: str,
        to_id: str,
        relation_type: str,
        properties: Dict[str, Any] = None
    ):
        """插入关系"""
        cypher = f"""
        MATCH (a), (b)
        WHERE a.id = $from_id AND b.id = $to_id
        MERGE (a)-[r:`{relation_type}`]->(b)
        SET r += $properties
        SET r.created_at = datetime()
        """
        
        await self.execute_cypher(cypher, {
            "from_id": from_id,
            "to_id": to_id,
            "properties": properties or {}
        })
    
    async def batch_insert_entities(self, entities: List[Dict]) -> List[str]:
        """批量插入实体"""
        ids = []
        for entity in entities:
            await self.insert_entity(
                entity_id=entity["id"],
                entity_type=entity["type"],
                properties=entity.get("properties", {}),
                doc_id=entity.get("doc_id", ""),
                category=entity.get("category", "")
            )
            ids.append(entity["id"])
        return ids
    
    async def query_by_entity(
        self, 
        entity_id: str, 
        max_hops: int = 2
    ) -> List[Dict]:
        """按实体 ID 查询及其关联关系"""
        cypher = f"""
        MATCH (e)
        WHERE e.id = $entity_id
        OPTIONAL MATCH (e)-[r*1..{max_hops}]-(connected)
        RETURN e, r, connected
        """
        
        results = await self.execute_cypher(cypher, {"entity_id": entity_id})
        return results
    
    async def query_by_category(
        self, 
        category: str, 
        limit: int = 100
    ) -> List[Dict]:
        """按分类查询实体"""
        cypher = """
        MATCH (e)
        WHERE e.category = $category
        RETURN e
        LIMIT $limit
        """
        
        results = await self.execute_cypher(cypher, {
            "category": category,
            "limit": limit
        })
        return results
    
    async def delete_entity(self, entity_id: str) -> bool:
        """删除实体及其关系"""
        cypher = """
        MATCH (e)
        WHERE e.id = $entity_id
        DETACH DELETE e
        """
        
        await self.execute_cypher(cypher, {"entity_id": entity_id})
        return True
    
    async def get_stats(self) -> Dict:
        """获取图谱统计信息"""
        stats_cypher = """
        MATCH ()-[r]->()
        RETURN type(r) as relation_type, count(*) as count
        """
        
        entity_stats_cypher = """
        MATCH (e)
        RETURN labels(e) as types, count(*) as count
        """
        
        relations = await self.execute_cypher(stats_cypher)
        entities = await self.execute_cypher(entity_stats_cypher)
        
        return {
            "relations": relations,
            "entities": entities
        }
