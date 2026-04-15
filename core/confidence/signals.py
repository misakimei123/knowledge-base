"""
三维度信号计算模块
- VectorSignalCalculator: 向量检索置信度
- KGSignalCalculator: 知识图谱置信度  
- FaithfulnessCalculator: 生成一致性验证
"""
import asyncio
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

from .types import ConfidenceSignal


class VectorSignalCalculator:
    """向量检索置信度计算"""
    
    def __init__(self, config: Dict[str, Any]):
        self.metric_type = config.get("metric_type", "COSINE")
        self.radius_threshold = config.get("radius_threshold", 0.65)
        self.topk_weights = config.get("topk_weights", "exponential")
        self.decay_factor = config.get("decay_factor", 0.7)
    
    async def calculate(self, query: str, results: List[Dict[str, Any]]) -> ConfidenceSignal:
        """
        计算向量检索置信度
        :param query: 查询文本
        :param results: 检索结果列表，每项包含 score 等字段
        :return: 置信度信号
        """
        if not results:
            return ConfidenceSignal(
                score=0.0,
                metadata={"reason": "no_results", "timestamp": datetime.now().isoformat()}
            )
        
        # 1. 分数归一化 + 阈值过滤
        normalized = []
        for r in results:
            score = self._normalize_score(r.get("score", 0))
            if score >= self.radius_threshold:
                normalized.append(score)
        
        if not normalized:
            return ConfidenceSignal(
                score=0.0,
                metadata={
                    "reason": "below_threshold",
                    "threshold": self.radius_threshold,
                    "max_score": max(r.get("score", 0) for r in results) if results else 0
                }
            )
        
        # 2. Top-K 加权融合
        weights = self._get_weights(len(normalized))
        topk_scores = normalized[:10]  # 最多取前 10 个
        topk_weights = weights[:len(topk_scores)]
        
        if len(topk_weights) > 0:
            vector_conf = float(np.average(topk_scores, weights=topk_weights))
        else:
            vector_conf = float(np.mean(topk_scores))
        
        # 3. 统计元数据
        category_dist = self._count_categories(results)
        
        return ConfidenceSignal(
            score=round(vector_conf, 4),
            metadata={
                "top1_score": normalized[0] if normalized else 0,
                "valid_count": len(normalized),
                "total_count": len(results),
                "category_distribution": category_dist,
                "threshold": self.radius_threshold
            }
        )
    
    def _normalize_score(self, score: float) -> float:
        """根据度量类型归一化分数到 [0, 1]"""
        if self.metric_type == "COSINE":
            # Cosine 相似度 [-1, 1] → [0, 1]
            return (score + 1) / 2
        elif self.metric_type == "L2":
            # L2 距离 → 相似度（指数衰减）
            return np.exp(-score)
        else:
            # 假设已经是 [0, 1] 范围
            return min(max(score, 0), 1)
    
    def _get_weights(self, n: int) -> np.ndarray:
        """生成排名权重"""
        if self.topk_weights == "uniform":
            return np.ones(n) / n
        elif self.topk_weights == "exponential":
            # 指数衰减权重
            weights = np.array([self.decay_factor ** i for i in range(n)])
            return weights / weights.sum()
        else:
            # 默认倒数权重
            weights = 1.0 / np.arange(1, n + 1)
            return weights / weights.sum()
    
    def _count_categories(self, results: List[Dict]) -> Dict[str, int]:
        """统计分类分布"""
        dist = {}
        for r in results:
            cat = r.get("category", "unknown")
            dist[cat] = dist.get(cat, 0) + 1
        return dist


class KGSignalCalculator:
    """知识图谱置信度计算"""
    
    def __init__(self, config: Dict[str, Any]):
        self.min_path_score = config.get("min_path_score", 0.3)
        self.relation_weights = config.get("relation_weights", {})
        self.entity_type_bonus = config.get("entity_type_bonus", {})
        self.path_decay = config.get("path_decay", 0.85)
    
    async def calculate(self, query: str, kg_results: List[Dict[str, Any]]) -> ConfidenceSignal:
        """
        计算知识图谱置信度
        :param query: 查询文本
        :param kg_results: KG 检索结果，包含 path_weight, hops, entity_confidence 等
        :return: 置信度信号
        """
        if not kg_results:
            return ConfidenceSignal(
                score=0.0,
                metadata={"reason": "no_kg_path"}
            )
        
        scores = []
        entity_types = set()
        
        for item in kg_results:
            # 路径评分 = 关系权重乘积 × 长度衰减
            path_weight = item.get("path_weight", 1.0)
            hops = item.get("hops", 1)
            path_score = path_weight * (self.path_decay ** hops)
            
            # 实体可信度（人工标注/来源质量）
            entity_conf = item.get("entity_confidence", 0.7)
            
            # 时效性加成
            update_time = item.get("update_time")
            time_bonus = self._calc_recency_bonus(update_time)
            
            # 实体类型加分
            entity_type = item.get("type", "")
            type_bonus = 1.0 + self.entity_type_bonus.get(entity_type, 0)
            
            composite = path_score * entity_conf * time_bonus * type_bonus
            
            if composite >= self.min_path_score:
                scores.append(composite)
                entity_types.add(entity_type)
        
        if not scores:
            return ConfidenceSignal(
                score=0.0,
                metadata={
                    "reason": "below_min_score",
                    "threshold": self.min_path_score,
                    "path_count": len(kg_results)
                }
            )
        
        kg_conf = float(np.mean(scores))
        
        return ConfidenceSignal(
            score=round(kg_conf, 4),
            metadata={
                "path_count": len(scores),
                "total_paths": len(kg_results),
                "max_path_score": round(max(scores), 4),
                "avg_hops": round(np.mean([r.get("hops", 1) for r in kg_results]), 2),
                "entity_types": list(entity_types)
            }
        )
    
    def _calc_recency_bonus(self, update_time: Any) -> float:
        """计算时效性加分（越新越高）"""
        if update_time is None:
            return 1.0
        
        try:
            if isinstance(update_time, str):
                from datetime import datetime
                update_time = datetime.fromisoformat(update_time.replace('Z', '+00:00'))
            
            days_ago = (datetime.now() - update_time).days
            if days_ago <= 30:
                return 1.2
            elif days_ago <= 90:
                return 1.1
            elif days_ago <= 365:
                return 1.0
            else:
                return 0.9
        except Exception:
            return 1.0


class FaithfulnessCalculator:
    """生成一致性验证（抗幻觉核心）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.strategy = config.get("strategy", "nli")
        self.nli_model_path = config.get("nli_model_path")
        self.keyword_min_overlap = config.get("keyword_min_overlap", 0.3)
        self._nli_model = None
    
    async def calculate(self, answer: str, contexts: List[str]) -> ConfidenceSignal:
        """
        计算生成一致性置信度
        :param answer: 生成的答案
        :param contexts: 参考上下文列表
        :return: 置信度信号
        """
        if not contexts or not answer:
            return ConfidenceSignal(
                score=0.0,
                metadata={"reason": "empty_input"}
            )
        
        if self.strategy == "keyword":
            score = self._calc_keyword_overlap(answer, contexts)
        elif self.strategy == "nli":
            score = await self._nli_verify(answer, contexts)
        elif self.strategy == "fact":
            score = await self._fact_consistency(answer, contexts)
        else:
            score = 0.5  # 默认降级
        
        return ConfidenceSignal(
            score=round(score, 4),
            metadata={
                "strategy": self.strategy,
                "context_count": len(contexts),
                "answer_length": len(answer)
            }
        )
    
    def _calc_keyword_overlap(self, answer: str, contexts: List[str]) -> float:
        """关键词重叠率计算"""
        import re
        
        # 简单分词（中文按字，英文按词）
        def tokenize(text: str) -> set:
            text = text.lower()
            words = re.findall(r'[\w\u4e00-\u9fff]+', text)
            return set(words)
        
        answer_tokens = tokenize(answer)
        if not answer_tokens:
            return 0.0
        
        context_tokens = set()
        for ctx in contexts[:3]:  # 最多用 3 个上下文
            context_tokens.update(tokenize(ctx))
        
        overlap = len(answer_tokens & context_tokens)
        overlap_rate = overlap / len(answer_tokens) if answer_tokens else 0
        
        return min(overlap_rate / self.keyword_min_overlap, 1.0)
    
    async def _nli_verify(self, answer: str, contexts: List[str]) -> float:
        """
        使用 NLI 模型判断：上下文 ⊨ 答案？
        生产环境需加载 ONNX 模型
        """
        # TODO: 加载 ONNX 模型并推理
        # premise = " ".join(contexts[:3])[:2048]
        # raw_score = self._get_nli_model().predict(premise, answer)
        # return (raw_score + 1) / 2
        
        # 临时降级方案：使用关键词重叠
        return self._calc_keyword_overlap(answer, contexts)
    
    async def _fact_consistency(self, answer: str, contexts: List[str]) -> float:
        """基于事实一致性的验证（需要 LLM 辅助）"""
        # TODO: 调用 LLM 进行事实验证
        return 0.5  # 临时降级
    
    def _get_nli_model(self):
        """懒加载 NLI 模型"""
        if self._nli_model is None and self.nli_model_path:
            # TODO: 加载 ONNX 模型
            pass
        return self._nli_model
