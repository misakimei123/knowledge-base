"""
置信度融合与校准模块
- ConfidenceFusionEngine: 融合策略 + 规则引擎
- ConfidenceCalibrator: 概率校准器
"""
import numpy as np
from typing import Dict, Any, Optional

from .types import ConfidenceSignal, RequestContext


class ConfidenceFusionEngine:
    """置信度融合引擎（支持动态权重 + 规则）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.default_weights = config.get("default_weights", {
            "vector": 0.4,
            "kg": 0.4,
            "faithfulness": 0.2
        })
        self.category_rules = config.get("category_rules", {})
        self.conflict_rules = config.get("conflict_rules", {})
    
    def fuse(self, signals: Dict[str, ConfidenceSignal], ctx: RequestContext) -> float:
        """
        融合多维度信号
        :param signals: 各维度置信度信号
        :param ctx: 请求上下文
        :return: 融合后的原始置信度
        """
        # 1. 动态权重（按分类/问题类型）
        weights = self._get_weights(ctx.category, ctx.query_type)
        
        # 2. 基础加权融合
        base_score = 0.0
        total_weight = 0.0
        
        for key in ["vector", "kg", "faithfulness"]:
            if key in signals and signals[key].score > 0:
                base_score += signals[key].score * weights.get(key, 0)
                total_weight += weights.get(key, 0)
        
        if total_weight > 0:
            base_score /= total_weight
        
        # 3. 业务规则修正（企业风控核心）
        adjusted = self._apply_rules(base_score, signals, ctx)
        
        return adjusted
    
    def _get_weights(self, category: str, query_type: str) -> Dict[str, float]:
        """根据分类和查询类型获取动态权重"""
        weights = self.default_weights.copy()
        
        # 按分类调整
        if category in self.category_rules:
            cat_weights = self.category_rules[category]
            for key, value in cat_weights.items():
                if key in weights:
                    weights[key] = value
        
        # 按查询类型调整
        if query_type == "entity":
            weights["kg"] = min(weights.get("kg", 0.4) + 0.15, 0.6)
            weights["vector"] = max(weights.get("vector", 0.4) - 0.15, 0.2)
        elif query_type == "relation":
            weights["kg"] = min(weights.get("kg", 0.4) + 0.2, 0.7)
        
        # 归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _apply_rules(self, score: float, signals: Dict[str, ConfidenceSignal], 
                     ctx: RequestContext) -> float:
        """应用企业级业务规则"""
        adjusted = score
        
        # 规则 1：关键分类强制高阈值
        critical_categories = ["Compliance", "Finance", "Legal", "HR_Policy"]
        if ctx.category in critical_categories and adjusted < 0.8:
            # 记录日志但不强行提升，让决策层处理
            pass
        
        # 规则 2：信号冲突检测（向量高但一致性低 → 可能幻觉）
        vector_score = signals.get("vector", ConfidenceSignal(0)).score
        faithfulness_score = signals.get("faithfulness", ConfidenceSignal(0)).score
        
        if vector_score > 0.85 and faithfulness_score < 0.5:
            conflict_factor = self.conflict_rules.get("high_retrieval_low_faithfulness", 0.7)
            adjusted *= conflict_factor
        
        # 规则 3：KG 与 Vector 严重不一致
        kg_score = signals.get("kg", ConfidenceSignal(0)).score
        if abs(vector_score - kg_score) > 0.4 and min(vector_score, kg_score) > 0:
            mismatch_factor = self.conflict_rules.get("kg_vector_mismatch", 0.85)
            adjusted *= mismatch_factor
        
        # 规则 4：新用户/低频问题保守策略
        if ctx.user_query_count < 10 and adjusted < 0.7:
            adjusted *= 0.9
        
        return max(0.0, min(1.0, adjusted))


class ConfidenceCalibrator:
    """概率校准器（解决分数虚高问题）"""
    
    def __init__(self, config: Dict[str, Any]):
        self.enabled = config.get("enabled", True)
        self.method = config.get("method", "temperature_scaling")
        self.params_path = config.get("params_path")
        self.params = self._load_calibration_params(config.get("params", {}))
    
    def calibrate(self, raw_score: float) -> float:
        """
        校准原始置信度分数
        :param raw_score: 原始分数 [0, 1]
        :return: 校准后分数 [0, 1]
        """
        if not self.enabled:
            return raw_score
        
        raw_score = max(0.001, min(0.999, raw_score))  # 避免数值问题
        
        if self.method == "temperature_scaling":
            return self._temperature_scaling(raw_score)
        elif self.method == "platt_scaling":
            return self._platt_scaling(raw_score)
        else:
            return raw_score
    
    def _temperature_scaling(self, raw_score: float) -> float:
        """温度缩放校准"""
        temperature = self.params.get("temperature", 1.2)
        
        # p_calibrated = sigmoid(logit(p_raw) / T)
        logit = np.log(raw_score / (1 - raw_score + 1e-7))
        calibrated = 1 / (1 + np.exp(-logit / temperature))
        
        return float(calibrated)
    
    def _platt_scaling(self, raw_score: float) -> float:
        """Platt Scaling 校准"""
        A = self.params.get("A", -1.0)
        B = self.params.get("B", 0.0)
        
        # p = 1 / (1 + exp(A*f + B))
        calibrated = 1 / (1 + np.exp(A * raw_score + B))
        
        return float(calibrated)
    
    def _load_calibration_params(self, default_params: Dict) -> Dict:
        """加载校准参数"""
        # 生产环境从文件加载
        # if self.params_path and os.path.exists(self.params_path):
        #     with open(self.params_path) as f:
        #         return json.load(f)
        
        # 默认参数
        if self.method == "temperature_scaling":
            return {"temperature": 1.2}
        elif self.method == "platt_scaling":
            return {"A": -1.0, "B": 0.0}
        return default_params
    
    def train(self, predictions: list, labels: list) -> None:
        """
        使用黄金数据集训练校准参数
        :param predictions: 模型预测的置信度
        :param labels: 真实标签（是否正确）
        """
        # TODO: 实现校准参数训练逻辑
        # 可以使用 scipy.optimize 最小化负对数似然
        pass


def map_confidence_level(confidence: float, ctx: RequestContext) -> str:
    """将置信度映射到等级"""
    thresholds = {
        "high": 0.75,
        "medium": 0.5
    }
    
    # 按分类调整阈值
    critical_categories = ["Compliance", "Finance", "Legal"]
    if ctx.category in critical_categories:
        thresholds["high"] = 0.85
        thresholds["medium"] = 0.6
    
    if confidence >= thresholds["high"]:
        return "high"
    elif confidence >= thresholds["medium"]:
        return "medium"
    else:
        return "low"


def generate_suggestions(level: str, signals: Dict[str, ConfidenceSignal]) -> list:
    """根据置信度等级和信号生成建议"""
    suggestions = []
    
    if level == "low":
        suggestions.append("建议人工审核此回答")
        suggestions.append("可尝试重新表述问题以获得更准确结果")
        
        # 分析哪个维度得分最低
        min_signal = min(signals.items(), key=lambda x: x[1].score)
        if min_signal[0] == "vector":
            suggestions.append("检索到的相关文档较少，建议扩充知识库")
        elif min_signal[0] == "kg":
            suggestions.append("知识图谱中缺少相关信息，建议完善实体关系")
        elif min_signal[0] == "faithfulness":
            suggestions.append("答案与参考内容一致性较低，可能存在幻觉")
    
    elif level == "medium":
        suggestions.append("已提供参考来源，请结合专业判断使用")
    
    return suggestions
