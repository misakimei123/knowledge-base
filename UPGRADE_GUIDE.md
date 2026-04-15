# 企业级知识库升级指南

## 📋 目录

1. [架构演进路线](#架构演进路线)
2. [Phase 2: 分布式扩展](#phase-2-分布式扩展)
3. [Phase 3: 多租户支持](#phase-3-多租户支持)
4. [性能优化建议](#性能优化建议)
5. [监控与运维](#监控与运维)

---

## 架构演进路线

```
当前版本 (v1.0) → Phase 2 → Phase 3
单机部署         分布式      多租户 SaaS
• Milvus Standalone  • Milvus Cluster   • 租户隔离
• Neo4j Single       • Neo4j Causal     • 资源配额
• 单 Ollama 实例     • 多模型负载均衡    • 计费系统
```

---

## Phase 2: 分布式扩展

### 2.1 Milvus 集群化

**目标**: 支持亿级向量检索，QPS > 1000

```yaml
# docker-compose.cluster.yml
services:
  milvus-etcd:
    image: quay.io/coreos/etcd:v3.5.5
    replicas: 3
  
  milvus-minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    mode: distributed
  
  milvus-querynode:
    image: milvusdb/milvus:v2.4.0
    replicas: 3
    deploy:
      resources:
        limits:
          memory: 8G
  
  milvus-datanode:
    image: milvusdb/milvus:v2.4.0
    replicas: 2
```

**迁移步骤**:
1. 备份现有数据：`python scripts/backup_milvus.py`
2. 部署集群：`docker-compose -f docker-compose.cluster.yml up -d`
3. 数据迁移：使用 Milvus Backup 工具
4. 验证一致性：运行集成测试

### 2.2 Neo4j 因果集群

**目标**: 高可用 + 读写分离

```yaml
services:
  neo4j-core:
    image: neo4j:5.15-enterprise
    environment:
      NEO4J_dbms_mode: CORE
    replicas: 3
  
  neo4j-read-replica:
    image: neo4j:5.15-enterprise
    environment:
      NEO4J_dbms_mode: READ_REPLICA
    replicas: 2
```

**代码调整**:
```python
# storage/neo4j_client.py
class Neo4jClient:
    def __init__(self, cluster_config: dict):
        # 读写分离路由
        self.driver = GraphDatabase.driver(
            cluster_config["bolt_uri"],
            auth=(cluster_config["user"], cluster_config["password"]),
            routing=True  # 启用路由
        )
    
    def read_query(self, cypher: str):
        with self.driver.session(access_mode=READ) as session:
            return session.run(cypher)
    
    def write_query(self, cypher: str):
        with self.driver.session(access_mode=WRITE) as session:
            return session.run(cypher)
```

### 2.3 模型服务负载均衡

**目标**: 支持并发 > 100 QPS

```python
# utils/llm_utils.py - 增强版
class LLMRouter:
    """LLM 请求路由"""
    
    def __init__(self, endpoints: List[str]):
        self.endpoints = endpoints
        self.health_checker = HealthChecker(endpoints)
        self.load_balancer = RoundRobinLB(endpoints)
    
    async def chat(self, messages: list) -> str:
        # 健康检查 + 负载均衡
        healthy = await self.health_checker.get_healthy()
        endpoint = self.load_balancer.select(healthy)
        return await self._call_endpoint(endpoint, messages)
```

---

## Phase 3: 多租户支持

### 3.1 数据隔离策略

**方案 A: 逻辑隔离（推荐初期）**
```python
# storage/milvus_client.py
class MultiTenantMilvus:
    def __init__(self):
        self.collection_prefix = "tenant_"
    
    def search(self, tenant_id: str, query_vector: list, **kwargs):
        collection = f"{self.collection_prefix}{tenant_id}"
        # 每个租户独立集合
        return self.client.search(collection, query_vector, **kwargs)
```

**方案 B: 物理隔离（高安全要求）**
```yaml
# 每个租户独立 Milvus 实例
services:
  tenant-a-milvus:
    image: milvusdb/milvus:v2.4.0
    ports: ["19531:19530"]
  
  tenant-b-milvus:
    image: milvusdb/milvus:v2.4.0
    ports: ["19532:19530"]
```

### 3.2 资源配额管理

```python
# core/quota_manager.py
class QuotaManager:
    """租户资源配额管理"""
    
    def __init__(self, config_path: str):
        self.config = load_yaml(config_path)
    
    def check_quota(self, tenant_id: str, resource: str) -> bool:
        quota = self.config["tenants"][tenant_id]["quotas"]
        current = self._get_current_usage(tenant_id, resource)
        return current < quota[resource]
    
    def record_usage(self, tenant_id: str, resource: str, amount: int):
        # 记录到 Redis 计数器
        redis.incrby(f"quota:{tenant_id}:{resource}", amount)
```

**配置示例**:
```yaml
# config/quotas.yaml
tenants:
  enterprise:
    quotas:
      daily_docs: 10000
      monthly_tokens: 10000000
      max_concurrent_queries: 50
    features:
      kg_search: true
      confidence_eval: true
  
  standard:
    quotas:
      daily_docs: 1000
      monthly_tokens: 1000000
      max_concurrent_queries: 10
    features:
      kg_search: false
      confidence_eval: true
```

---

## 性能优化建议

### 4.1 向量检索优化

```python
# storage/milvus_client.py - 索引优化
index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW",  # 或 IVF_FLAT for 大数据量
    "params": {
        "M": 16,           # 增加 M 提升精度
        "efConstruction": 256,  # 增加构建时间换取更好索引
    }
}

search_params = {
    "metric_type": "COSINE",
    "params": {"ef": 64}  # 搜索时 ef 越大越准但越慢
}
```

### 4.2 缓存策略

```python
# utils/cache.py
from functools import lru_cache
import redis

class HybridCache:
    """LRU + Redis 混合缓存"""
    
    def __init__(self):
        self.local_cache = {}
        self.redis_client = redis.Redis(host='localhost', port=6379)
    
    @lru_cache(maxsize=1000)
    def get_embedding(self, text_hash: str):
        # 本地缓存热门查询
        cached = self.redis_client.get(f"emb:{text_hash}")
        if cached:
            return pickle.loads(cached)
        
        # 计算 embedding...
        embedding = compute_embedding(text)
        
        # 写入缓存
        self.redis_client.setex(
            f"emb:{text_hash}", 
            3600, 
            pickle.dumps(embedding)
        )
        return embedding
```

### 4.3 异步批处理

```python
# core/pipeline.py - 批量处理优化
async def batch_process_documents(self, docs: List[Document], batch_size: int = 32):
    """批量处理文档"""
    batches = [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]
    
    tasks = []
    for batch in batches:
        task = asyncio.create_task(self._process_batch(batch))
        tasks.append(task)
        
        # 控制并发数
        if len(tasks) >= 10:
            await asyncio.gather(*tasks)
            tasks = []
    
    await asyncio.gather(*tasks)
```

---

## 监控与运维

### 5.1 Prometheus 指标

```python
# utils/monitoring.py
from prometheus_client import Counter, Histogram, Gauge

# 定义指标
REQUEST_COUNT = Counter('kb_request_total', '总请求数', ['category', 'status'])
REQUEST_LATENCY = Histogram('kb_request_latency_seconds', '请求延迟')
CONFIDENCE_SCORE = Gauge('kb_confidence_score', '置信度分数', ['level'])
VECTOR_SEARCH_LATENCY = Histogram('kb_vector_search_latency', '向量检索延迟')
KG_SEARCH_LATENCY = Histogram('kb_kg_search_latency', '图谱检索延迟')

# 埋点
@REQUEST_LATENCY.time()
async def retrieve(query: str, category: str):
    try:
        result = await engine.retrieve(query, category)
        REQUEST_COUNT.labels(category=category, status='success').inc()
        return result
    except Exception as e:
        REQUEST_COUNT.labels(category=category, status='error').inc()
        raise
```

### 5.2 Grafana 仪表盘

导入以下 Dashboard ID:
- **应用监控**: 自定义 JSON (见 `monitoring/grafana_dashboard.json`)
- **Milvus 监控**: Milvus 官方 Dashboard
- **Neo4j 监控**: Neo4j Official Dashboard

### 5.3 告警规则

```yaml
# monitoring/alerts.yml
groups:
  - name: knowledge_base_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(kb_request_total{status="error"}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "错误率超过 5%"
      
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(kb_request_latency_seconds_bucket[5m])) > 3
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "P95 延迟超过 3 秒"
      
      - alert: LowConfidenceSpike
        expr: rate(kb_confidence_score{level="low"}[10m]) > 0.3
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "低置信度比例异常升高"
```

---

## 版本兼容性矩阵

| 组件 | v1.0 | v2.0 (分布式) | v3.0 (多租户) |
|------|------|---------------|---------------|
| Milvus | 2.4.x | 2.4.x+ | 2.4.x+ |
| Neo4j | 5.x Community | 5.x Enterprise | 5.x Enterprise |
| Python | 3.10+ | 3.10+ | 3.10+ |
| Docker | 20.10+ | 20.10+ | 24.0+ |

---

## 回滚方案

```bash
# 快速回滚脚本
#!/bin/bash
# rollback.sh

VERSION=$1

echo "🔄 回滚到版本：$VERSION"

# 1. 停止当前服务
docker-compose down

# 2. 恢复数据库备份
python scripts/restore_backup.py --version $VERSION

# 3. 启动旧版本
git checkout $VERSION
docker-compose up -d

echo "✅ 回滚完成"
```

---

## 联系与支持

- **技术文档**: `/README.md`
- **API 文档**: `/API_SPEC.md`
- **问题反馈**: GitHub Issues
- **紧急支持**: 运维团队联系方式

