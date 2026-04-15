# Enterprise Knowledge Base

企业级知识库系统，采用 RAG+知识图谱混合检索，内置置信度评估模块。

## 核心特性

- 🔍 **混合检索**: Milvus 向量检索 + Neo4j 知识图谱
- 📊 **置信度评估**: 三维度打分（向量/图谱/一致性）+ 概率校准
- 🔐 **权限隔离**: 按分类管理文档，检索时自动过滤
- ⚡ **高性能**: 异步处理，支持日均 3000 文档

## 快速启动

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务（需要 Milvus + Neo4j + Ollama）
python main.py

# 3. 访问 API
# - API: http://localhost:7866/docs
# - Gradio UI: http://localhost:7866
```

## 项目结构

```
enterprise-knowledge-base/
├── main.py                    # 主入口（FastAPI + Gradio）
├── config/
│   ├── settings.py           # Pydantic 配置
│   └── confidence_rules.yaml # 置信度规则
├── core/
│   ├── rag_engine.py         # 混合检索编排
│   ├── kg_engine.py          # Neo4j 图谱操作
│   ├── pipeline.py           # 文档处理流水线
│   └── confidence/           # 置信度评估模块
├── storage/
│   ├── milvus_client.py      # Milvus 封装
│   ├── neo4j_client.py       # Neo4j 封装
│   └── metadata_db.py        # SQLite 元数据
├── utils/
│   ├── file_utils.py         # 文件工具
│   ├── embedding_utils.py    # Embedding 工具
│   └── llm_utils.py          # LLM 调用工具
└── tests/
    └── test_confidence.py    # 单元测试
```

## API 接口

### 查询接口
```bash
POST /api/query
{
  "query": "员工年假政策是什么？",
  "category": "HR",
  "user_id": "user123"
}
```

### 上传接口
```bash
POST /api/upload
FormData:
- file: [PDF/DOCX/TXT]
- category: HR
- user_id: admin
```

## 置信度评估

系统从三个维度评估答案可信度：

| 维度 | 说明 | 权重 |
|------|------|------|
| **向量检索** | 语义相似度评分 | 40% |
| **知识图谱** | 实体关系路径质量 | 40% |
| **一致性验证** | 答案与上下文一致性 | 20% |

输出包含：
- `confidence`: 置信分数 (0-1)
- `level`: high/medium/low
- `explanation`: 自然语言解释
- `audit_id`: 审计追踪 ID

## 开发路线图

- [x] Phase 1: 基础架构搭建
- [ ] Phase 2: 置信度模块开发
- [ ] Phase 3: 企业增强与测试

## 技术栈

- **框架**: FastAPI + Gradio
- **向量库**: Milvus 2.4+
- **图数据库**: Neo4j 5.x
- **Embedding**: BGE-M3
- **LLM**: Qwen2.5-14B-Chat (via Ollama)

## License

MIT
