"""
置信度评估主入口
"""
import asyncio
import logging
from typing import List, Dict, Any
from datetime import datetime

from .types import ConfidenceSignal, ConfidenceResult, RequestContext
from .signals import VectorSignalCalculator, KGSignalCalculator, FaithfulnessCalculator
from .fusion import ConfidenceFusionEngine, ConfidenceCalibrator, map_confidence_level, generate_suggestions
from config.settings import load_config

logger = logging.getLogger(__name__)


class ConfidenceEvaluator:
    """企业级置信度评估主入口"""
    
    def __init__(self, config_path: str = "config/confidence_rules.yaml"):
        """
        初始化置信度评估器
        :param config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        
        # 初始化各维度计算器
        self.vector_calc = VectorSignalCalculator(self.config["vector"])
        self.kg_calc = KGSignalCalculator(self.config["kg"])
        self.faithfulness_calc = FaithfulnessCalculator(self.config["faithfulness"])
        
        # 融合与校准
        self.fusion = ConfidenceFusionEngine(self.config["fusion"])
        self.calibrator = ConfidenceCalibrator(self.config["calibration"])
    
    async def evaluate(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        kg_results: List[Dict[str, Any]],
        answer: str,
        contexts: List[str],
        request_context: RequestContext
    ) -> ConfidenceResult:
        """
        执行完整置信度评估流程
        :param query: 用户查询
        :param vector_results: 向量检索结果
        :param kg_results: 知识图谱检索结果
        :param answer: 生成的答案
        :param contexts: 参考上下文
        :param request_context: 请求上下文
        :return: 结构化置信度结果（含可解释报告）
        """
        try:
            # 1. 并行计算三维度信号
            vector_signal_task = asyncio.create_task(
                self.vector_calc.calculate(query, vector_results)
            )
            kg_signal_task = asyncio.create_task(
                self.kg_calc.calculate(query, kg_results)
            )
            faithfulness_signal_task = asyncio.create_task(
                self.faithfulness_calc.calculate(answer, contexts)
            )
            
            vector_signal, kg_signal, faithfulness_signal = await asyncio.gather(
                vector_signal_task,
                kg_signal_task,
                faithfulness_signal_task
            )
            
            signals = {
                "vector": vector_signal,
                "kg": kg_signal,
                "faithfulness": faithfulness_signal
            }
            
            # 2. 融合 + 校准
            raw_confidence = self.fusion.fuse(signals, request_context)
            calibrated_confidence = self.calibrator.calibrate(raw_confidence)
            
            # 3. 确定置信等级
            level = map_confidence_level(calibrated_confidence, request_context)
            
            # 4. 生成可解释报告
            explanation = self._generate_explanation(
                signals, 
                calibrated_confidence, 
                request_context
            )
            
            # 5. 生成建议
            suggestions = generate_suggestions(level, signals)
            
            # 6. 生成审计 ID
            audit_id = self._generate_audit_id()
            
            return ConfidenceResult(
                confidence=round(calibrated_confidence, 4),
                level=level,
                explanation=explanation,
                audit_id=audit_id,
                signals=signals,
                suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Confidence evaluation failed: {e}", exc_info=True)
            # 降级返回中等置信度
            return ConfidenceResult(
                confidence=0.5,
                level="medium",
                explanation=f"置信度评估失败：{str(e)}",
                audit_id=self._generate_audit_id(),
                signals={},
                suggestions=["系统异常，建议人工审核"]
            )
    
    def _generate_explanation(
        self, 
        signals: Dict[str, ConfidenceSignal], 
        confidence: float,
        ctx: RequestContext
    ) -> str:
        """生成自然语言解释报告"""
        parts = []
        
        # 总体评价
        if confidence >= 0.75:
            parts.append("回答可信度高。")
        elif confidence >= 0.5:
            parts.append("回答可信度中等，建议结合来源判断。")
        else:
            parts.append("回答可信度较低，请谨慎使用。")
        
        # 各维度分析
        vector_score = signals.get("vector", ConfidenceSignal(0)).score
        kg_score = signals.get("kg", ConfidenceSignal(0)).score
        faithfulness_score = signals.get("faithfulness", ConfidenceSignal(0)).score
        
        if vector_score > 0.7:
            parts.append(f"向量检索匹配良好 (相似度：{vector_score:.2f})。")
        elif vector_score > 0:
            parts.append(f"向量检索匹配一般 (相似度：{vector_score:.2f})。")
        else:
            parts.append("未检索到相关文档。")
        
        if kg_score > 0.5:
            parts.append(f"知识图谱支持充分 (图谱置信度：{kg_score:.2f})。")
        elif kg_score > 0:
            parts.append(f"知识图谱部分支持 (图谱置信度：{kg_score:.2f})。")
        else:
            parts.append("知识图谱中无相关信息。")
        
        if faithfulness_score > 0.7:
            parts.append(f"答案与参考内容一致性高 (一致性：{faithfulness_score:.2f})。")
        elif faithfulness_score > 0:
            parts.append(f"答案与参考内容有一定一致性 (一致性：{faithfulness_score:.2f})。")
        
        # 分类特殊说明
        if ctx.category in ["Compliance", "Finance", "Legal"]:
            parts.append("【重要】此问题涉及关键业务领域，建议专业审核。")
        
        return "".join(parts)
    
    def _generate_audit_id(self) -> str:
        """生成审计追踪 ID"""
        import uuid
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"CONF_{timestamp}_{unique_id}"
    
    def evaluate_sync(
        self,
        query: str,
        vector_results: List[Dict[str, Any]],
        kg_results: List[Dict[str, Any]],
        answer: str,
        contexts: List[str],
        request_context: RequestContext
    ) -> ConfidenceResult:
        """同步版本（用于非异步环境）"""
        return asyncio.run(self.evaluate(
            query, vector_results, kg_results, answer, contexts, request_context
        ))
