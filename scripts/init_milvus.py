#!/usr/bin/env python3
"""
Milvus 集合初始化脚本
创建向量集合并设置索引
"""

import asyncio
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility,
)


async def init_milvus(
    host: str = "localhost",
    port: int = 19530,
    collection_name: str = "enterprise_knowledge",
):
    """初始化 Milvus 集合"""
    
    # 连接 Milvus
    connections.connect(host=host, port=port)
    
    # 检查集合是否已存在
    if utility.has_collection(collection_name):
        print(f"⚠️  集合 '{collection_name}' 已存在，删除后重建...")
        utility.drop_collection(collection_name)
    
    # 定义字段
    fields = [
        FieldSchema(
            name="id", 
            dtype=DataType.INT64, 
            is_primary=True, 
            auto_id=True
        ),
        FieldSchema(
            name="embedding", 
            dtype=DataType.FLOAT_VECTOR, 
            dim=1024,  # BGE-M3 输出维度
            description="文档块向量"
        ),
        FieldSchema(
            name="doc_id", 
            dtype=DataType.VARCHAR, 
            max_length=128,
            description="文档 ID"
        ),
        FieldSchema(
            name="category", 
            dtype=DataType.VARCHAR, 
            max_length=64,
            description="文档分类"
        ),
        FieldSchema(
            name="chunk_text", 
            dtype=DataType.VARCHAR, 
            max_length=8192,
            description="文档块内容"
        ),
        FieldSchema(
            name="section", 
            dtype=DataType.VARCHAR, 
            max_length=256,
            description="章节标题"
        ),
        FieldSchema(
            name="visible_roles", 
            dtype=DataType.VARCHAR, 
            max_length=512,
            description="可见角色列表（JSON）"
        ),
        FieldSchema(
            name="created_at", 
            dtype=DataType.INT64,
            description="创建时间戳"
        ),
    ]
    
    # 创建 schema
    schema = CollectionSchema(
        fields=fields,
        description="企业知识库向量集合",
    )
    
    # 创建集合
    collection = Collection(name=collection_name, schema=schema)
    print(f"✅ 集合 '{collection_name}' 创建成功")
    
    # 创建索引
    index_params = {
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "params": {"M": 16, "efConstruction": 200},
    }
    
    collection.create_index(
        field_name="embedding",
        index_params=index_params,
    )
    print("✅ HNSW 索引创建成功")
    
    # 加载集合到内存
    collection.load()
    print("✅ 集合已加载到内存")
    
    # 验证集合信息
    print(f"\n📊 集合信息:")
    print(f"   - 名称：{collection_name}")
    print(f"   - 实体数：{collection.num_entities}")
    print(f"   - 维度：1024")
    print(f"   - 度量类型：COSINE")
    
    return collection


def main():
    """主函数"""
    import os
    
    host = os.getenv("MILVUS_HOST", "localhost")
    port = int(os.getenv("MILVUS_PORT", "19530"))
    
    print("🚀 开始初始化 Milvus...")
    try:
        collection = asyncio.run(init_milvus(host, port))
        print("\n✨ Milvus 初始化完成！")
    except Exception as e:
        print(f"\n❌ 初始化失败：{e}")
        print("💡 请确保 Milvus 服务已启动且可访问")
        raise


if __name__ == "__main__":
    main()
