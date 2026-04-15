#!/usr/bin/env python3
"""
Neo4j 索引和约束初始化脚本
创建图谱索引、约束和基础标签
"""

from neo4j import GraphDatabase
import os


class Neo4jInitializer:
    """Neo4j 数据库初始化工具"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def close(self):
        """关闭连接"""
        self.driver.close()
    
    def create_constraints(self):
        """创建唯一性约束"""
        constraints = [
            # 实体唯一性约束
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT entity_name_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE",
            
            # 文档唯一性约束
            "CREATE CONSTRAINT doc_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.doc_id IS UNIQUE",
            
            # 关系类型索引
            "CREATE INDEX relation_type_index IF NOT EXISTS FOR ()-[r:RELATED]-() ON (r.type)",
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                try:
                    result = session.run(constraint)
                    result.consume()
                    print(f"✅ 约束/索引创建：{constraint[:60]}...")
                except Exception as e:
                    print(f"⚠️  约束已存在或创建失败：{e}")
    
    def create_labels(self):
        """创建基础标签和属性索引"""
        indexes = [
            # Entity 标签索引
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_category_index IF NOT EXISTS FOR (e:Entity) ON (e.category)",
            "CREATE INDEX entity_confidence_index IF NOT EXISTS FOR (e:Entity) ON (e.confidence)",
            
            # Document 标签索引
            "CREATE INDEX doc_category_index IF NOT EXISTS FOR (d:Document) ON (d.category)",
            "CREATE INDEX doc_created_index IF NOT EXISTS FOR (d:Document) ON (d.created_at)",
            
            # Relationship 索引
            "CREATE INDEX rel_source_index IF NOT EXISTS FOR ()-[r]->() ON (r.source_doc)",
            "CREATE INDEX rel_timestamp_index IF NOT EXISTS FOR ()-[r]->() ON (r.created_at)",
        ]
        
        with self.driver.session() as session:
            for index in indexes:
                try:
                    result = session.run(index)
                    result.consume()
                    print(f"✅ 索引创建：{index[:60]}...")
                except Exception as e:
                    print(f"⚠️  索引创建失败：{e}")
    
    def create_base_schema(self):
        """创建基础图谱模式"""
        schema_cyphers = [
            # 创建示例分类节点
            """
            MERGE (hr:Category {name: 'HR', path: 'HR'})
            MERGE (tech:Category {name: 'Technical', path: 'Technical'})
            MERGE (finance:Category {name: 'Finance', path: 'Finance'})
            MERGE (compliance:Category {name: 'Compliance', path: 'Compliance'})
            """,
            
            # 创建分类层级关系
            """
            MATCH (parent:Category {name: 'HR'})
            MERGE (policy:Category {name: 'HR_Policy', path: 'HR/Policy'})
            MERGE (recruit:Category {name: 'HR_Recruitment', path: 'HR/Recruitment'})
            MERGE (policy)-[:SUB_CATEGORY_OF]->(parent)
            MERGE (recruit)-[:SUB_CATEGORY_OF]->(parent)
            """,
        ]
        
        with self.driver.session() as session:
            for cypher in schema_cyphers:
                try:
                    result = session.run(cypher)
                    result.consume()
                    print("✅ 基础图谱模式创建成功")
                except Exception as e:
                    print(f"⚠️  模式创建失败：{e}")
    
    def verify_setup(self):
        """验证初始化结果"""
        with self.driver.session() as session:
            # 统计节点数
            node_count = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            
            # 统计关系数
            rel_count = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            
            # 列出所有标签
            labels = session.run("CALL db.labels() YIELD label RETURN collect(label) as labels").single()["labels"]
            
            # 列出所有关系类型
            rel_types = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) as types").single()["types"]
            
            print("\n📊 Neo4j 图谱统计:")
            print(f"   - 节点总数：{node_count}")
            print(f"   - 关系总数：{rel_count}")
            print(f"   - 标签类型：{labels}")
            print(f"   - 关系类型：{rel_types}")


def main():
    """主函数"""
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "your_secure_password")
    
    print("🚀 开始初始化 Neo4j...")
    
    try:
        initializer = Neo4jInitializer(uri, user, password)
        
        print("\n1️⃣  创建约束...")
        initializer.create_constraints()
        
        print("\n2️⃣  创建索引...")
        initializer.create_labels()
        
        print("\n3️⃣  创建基础图谱模式...")
        initializer.create_base_schema()
        
        print("\n4️⃣  验证初始化结果...")
        initializer.verify_setup()
        
        initializer.close()
        print("\n✨ Neo4j 初始化完成！")
        
    except Exception as e:
        print(f"\n❌ 初始化失败：{e}")
        print("💡 请确保 Neo4j 服务已启动且认证信息正确")
        raise


if __name__ == "__main__":
    main()
