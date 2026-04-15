"""
Pydantic 配置类 - 类型安全的配置管理
支持 YAML 热更新
"""
import os
import yaml
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class VectorConfig(BaseModel):
    """向量检索配置"""
    metric_type: str = "COSINE"
    radius_threshold: float = 0.65
    topk_weights: str = "exponential"
    decay_factor: float = 0.7


class KGConfig(BaseModel):
    """知识图谱配置"""
    min_path_score: float = 0.3
    relation_weights: Dict[str, float] = Field(default_factory=lambda: {
        "employed_by": 1.0,
        "located_in": 0.9,
        "mentioned_in": 0.6
    })
    entity_type_bonus: Dict[str, float] = Field(default_factory=lambda: {
        "Policy": 0.1,
        "Department": 0.05
    })
    path_decay: float = 0.85


class FaithfulnessConfig(BaseModel):
    """一致性验证配置"""
    strategy: str = "nli"
    nli_model_path: str = "models/bge-reranker-v2-m3.onnx"
    keyword_min_overlap: float = 0.3


class FusionConfig(BaseModel):
    """融合策略配置"""
    default_weights: Dict[str, float] = Field(default_factory=lambda: {
        "vector": 0.4,
        "kg": 0.4,
        "faithfulness": 0.2
    })
    category_rules: Dict[str, Dict[str, float]] = Field(default_factory=dict)
    conflict_rules: Dict[str, float] = Field(default_factory=lambda: {
        "high_retrieval_low_faithfulness": 0.7,
        "kg_vector_mismatch": 0.85
    })


class CalibrationConfig(BaseModel):
    """校准配置"""
    enabled: bool = True
    method: str = "temperature_scaling"
    params_path: str = "models/calibration_params.json"
    retrain_schedule: str = "monthly"


class DecisionConfig(BaseModel):
    """决策阈值配置"""
    thresholds: Dict[str, float] = Field(default_factory=lambda: {
        "high": 0.75,
        "medium": 0.5
    })
    actions: Dict[str, List[str]] = Field(default_factory=lambda: {
        "high": ["return_answer", "cache"],
        "medium": ["return_answer", "show_sources", "log_for_review"],
        "low": ["refuse_answer", "suggest_search", "create_ticket"]
    })


class Settings(BaseSettings):
    """应用主配置"""
    
    # 应用配置
    app_name: str = "Enterprise Knowledge Base"
    debug: bool = False
    log_level: str = "INFO"
    
    # 服务地址
    milvus_uri: str = "http://localhost:19530"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    ollama_base_url: str = "http://localhost:11434"
    
    # 模型配置
    embedding_model: str = "BAAI/bge-m3"
    llm_model: str = "qwen2.5-14b-chat"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    
    # 检索配置
    vector_top_k: int = 10
    kg_top_k: int = 5
    final_top_k: int = 8
    rerank_enable: bool = True
    rrf_k: int = 60
    
    # 文档处理配置
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_concurrent_docs: int = 10
    
    # 置信度配置
    confidence: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        env_file = ".env"
        env_prefix = "EKB_"
        extra = "allow"


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    if not os.path.exists(config_path):
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str = "config/confidence_rules.yaml") -> Dict[str, Any]:
    """
    加载配置（支持热更新）
    :param config_path: YAML 配置文件路径
    :return: 配置字典
    """
    config = load_yaml_config(config_path)
    
    # 合并默认配置
    default_config = {
        "vector": {
            "metric_type": "COSINE",
            "radius_threshold": 0.65,
            "topk_weights": "exponential",
            "decay_factor": 0.7
        },
        "kg": {
            "min_path_score": 0.3,
            "relation_weights": {"employed_by": 1.0, "located_in": 0.9, "mentioned_in": 0.6},
            "entity_type_bonus": {"Policy": 0.1, "Department": 0.05},
            "path_decay": 0.85
        },
        "faithfulness": {
            "strategy": "nli",
            "nli_model_path": "models/bge-reranker-v2-m3.onnx",
            "keyword_min_overlap": 0.3
        },
        "fusion": {
            "default_weights": {"vector": 0.4, "kg": 0.4, "faithfulness": 0.2},
            "category_rules": {},
            "conflict_rules": {"high_retrieval_low_faithfulness": 0.7, "kg_vector_mismatch": 0.85}
        },
        "calibration": {
            "enabled": True,
            "method": "temperature_scaling",
            "params_path": "models/calibration_params.json"
        },
        "decision": {
            "thresholds": {"high": 0.75, "medium": 0.5},
            "actions": {
                "high": ["return_answer", "cache"],
                "medium": ["return_answer", "show_sources", "log_for_review"],
                "low": ["refuse_answer", "suggest_search", "create_ticket"]
            }
        }
    }
    
    # 递归合并
    def merge(base: dict, override: dict) -> dict:
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge(result[key], value)
            else:
                result[key] = value
        return result
    
    return merge(default_config, config)


# 全局配置实例
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """获取全局配置单例"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
