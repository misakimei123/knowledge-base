#!/usr/bin/env python3
"""
置信度模块单元测试
测试三维度信号计算、融合策略和校准功能
"""

import pytest
import asyncio
from datetime import datetime
from pathlib import Path

# 导入被测试模块
from core.confidence.types import ConfidenceSignal, ConfidenceResult
from core.confidence.signals import (
    VectorSignalCalculator,
    KGSignalCalculator,
    FaithfulnessCalculator,
)
from core.confidence.fusion import ConfidenceFusionEngine, ConfidenceCalibrator
from core.confidence.evaluator import ConfidenceEvaluator


class TestConfidenceSignal:
    """测试 ConfidenceSignal 数据类型"""
    
    def test_signal_creation(self):
        """测试信号创建"""
        signal = ConfidenceSignal(score=0.85, metadata={"reason": "test"})
        assert signal.score == 0.85
        assert signal.metadata["reason"] == "test"
        assert isinstance(signal.timestamp, datetime)
    
    def test_signal_default_values(self):
        """测试默认值"""
        signal = ConfidenceSignal(score=0.5)
        assert signal.score == 0.5
        assert signal.metadata == {}
        assert isinstance(signal.timestamp, datetime)


class TestVectorSignalCalculator:
    """测试向量检索置信度计算"""
    
    @pytest.fixture
    def calculator(self):
        config = {
            "metric_type": "COSINE",
            "radius_threshold": 0.65,
            "topk_weights": "exponential",
        }
        return VectorSignalCalculator(config)
    
    def test_no_results(self, calculator):
        """测试无结果情况"""
        result = asyncio.run(calculator.calculate("query", []))
        assert result.score == 0.0
        assert result.metadata["reason"] == "no_results"
    
    def test_below_threshold(self, calculator):
        """测试低于阈值"""
        # Cosine 相似度会进行归一化：(score + 1) / 2
        # 所以原始分数 0.2 → 归一化后 0.6 < 0.65，会被过滤
        results = [{"score": 0.2}, {"score": 0.1}]
        result = asyncio.run(calculator.calculate("query", results))
        assert result.score == 0.0
        assert result.metadata["reason"] == "below_threshold"
    
    def test_valid_results(self, calculator):
        """测试有效结果"""
        # Cosine 相似度归一化：(score + 1) / 2
        # 0.8 → 0.9, 0.7 → 0.85, 0.6 → 0.8
        results = [
            {"score": 0.8, "metadata": {"category": "HR"}},
            {"score": 0.7, "metadata": {"category": "HR"}},
            {"score": 0.6, "metadata": {"category": "Technical"}},
        ]
        result = asyncio.run(calculator.calculate("query", results))
        assert result.score > 0.0
        assert result.score <= 1.0
        assert "top1_score" in result.metadata
        assert result.metadata["top1_score"] == 0.9  # (0.8 + 1) / 2 = 0.9
    
    def test_mixed_scores(self, calculator):
        """测试混合分数（部分低于阈值）"""
        # 0.8 → 0.9 (通过), 0.2 → 0.6 (过滤), 0.6 → 0.8 (通过)
        results = [
            {"score": 0.8},
            {"score": 0.2},  # 低于阈值，应被过滤
            {"score": 0.6},
        ]
        result = asyncio.run(calculator.calculate("query", results))
        assert result.score > 0.0
        assert result.metadata["valid_count"] == 2


class TestKGSignalCalculator:
    """测试知识图谱置信度计算"""
    
    @pytest.fixture
    def calculator(self):
        config = {
            "min_path_score": 0.3,
            "relation_weights": {
                "employed_by": 1.0,
                "located_in": 0.9,
            },
        }
        return KGSignalCalculator(config)
    
    def test_no_kg_results(self, calculator):
        """测试无图谱结果"""
        result = asyncio.run(calculator.calculate("query", []))
        assert result.score == 0.0
        assert result.metadata["reason"] == "no_kg_path"
    
    def test_valid_paths(self, calculator):
        """测试有效路径"""
        results = [
            {
                "path_weight": 0.9,
                "hops": 1,
                "entity_confidence": 0.95,
                "update_time": datetime.now(),
                "type": "Policy",
            },
            {
                "path_weight": 0.7,
                "hops": 2,
                "entity_confidence": 0.8,
                "update_time": datetime.now(),
                "type": "Department",
            },
        ]
        result = asyncio.run(calculator.calculate("query", results))
        assert result.score > 0.0
        assert result.score <= 1.0
        assert result.metadata["path_count"] == 2
    
    def test_low_quality_paths(self, calculator):
        """测试低质量路径（低于阈值）"""
        results = [
            {
                "path_weight": 0.2,  # 太低
                "hops": 3,
                "entity_confidence": 0.5,
            },
        ]
        result = asyncio.run(calculator.calculate("query", results))
        assert result.score == 0.0


class TestFaithfulnessCalculator:
    """测试一致性验证"""
    
    @pytest.fixture
    def calculator(self):
        config = {
            "strategy": "keyword",
            "keyword_min_overlap": 0.3,
        }
        return FaithfulnessCalculator(config)
    
    def test_high_overlap(self, calculator):
        """测试高重叠率"""
        answer = "员工年假为 15 天"
        contexts = ["根据公司政策，全职员工每年享有 15 天带薪年假。"]
        result = asyncio.run(calculator.calculate(answer, contexts))
        assert result.score > 0.5
    
    def test_low_overlap(self, calculator):
        """测试低重叠率"""
        answer = "完全无关的内容"
        contexts = ["公司政策关于年假的规定。"]
        result = asyncio.run(calculator.calculate(answer, contexts))
        assert result.score < 0.5
    
    def test_empty_contexts(self, calculator):
        """测试空上下文"""
        answer = "某个答案"
        contexts = []
        result = asyncio.run(calculator.calculate(answer, contexts))
        assert result.score == 0.0 or result.score < 0.3


class TestConfidenceFusionEngine:
    """测试置信度融合引擎"""
    
    @pytest.fixture
    def fusion_engine(self):
        config = {
            "default_weights": {
                "vector": 0.4,
                "kg": 0.4,
                "faithfulness": 0.2,
            },
            "category_rules": {},
            "conflict_rules": {
                "high_retrieval_low_faithfulness": 0.7,
            },
        }
        return ConfidenceFusionEngine(config)
    
    def test_basic_fusion(self, fusion_engine):
        """测试基础融合"""
        signals = {
            "vector": ConfidenceSignal(score=0.85),
            "kg": ConfidenceSignal(score=0.75),
            "faithfulness": ConfidenceSignal(score=0.90),
        }
        
        class MockContext:
            category = "HR_Policy"
            query_type = "factual"
            user_query_count = 50
        
        score = fusion_engine.fuse(signals, MockContext())
        assert 0.0 <= score <= 1.0
        # 期望：0.4*0.85 + 0.4*0.75 + 0.2*0.90 = 0.34 + 0.30 + 0.18 = 0.82
        assert abs(score - 0.82) < 0.01
    
    def test_conflict_detection(self, fusion_engine):
        """测试冲突检测（向量高但一致性低）"""
        signals = {
            "vector": ConfidenceSignal(score=0.95),
            "kg": ConfidenceSignal(score=0.80),
            "faithfulness": ConfidenceSignal(score=0.30),  # 低一致性
        }
        
        class MockContext:
            category = "HR_Policy"
            query_type = "factual"
            user_query_count = 50
        
        score = fusion_engine.fuse(signals, MockContext())
        # 应触发冲突规则，分数降级
        assert score < 0.82  # 低于正常融合结果


class TestConfidenceCalibrator:
    """测试置信度校准器"""
    
    @pytest.fixture
    def calibrator(self):
        config = {
            "enabled": True,
            "method": "temperature_scaling",
            "params": {"temperature": 1.2},
        }
        return ConfidenceCalibrator(config)
    
    def test_temperature_scaling(self, calibrator):
        """测试温度缩放"""
        raw_score = 0.7
        calibrated = calibrator.calibrate(raw_score)
        assert 0.0 <= calibrated <= 1.0
        # 温度>1 时，极端分数应向 0.5 收缩
        assert abs(calibrated - 0.5) < abs(raw_score - 0.5)
    
    def test_disabled_calibration(self):
        """测试禁用校准"""
        config = {
            "enabled": False,
            "method": "temperature_scaling",
            "params": {"temperature": 1.2},
        }
        calibrator = ConfidenceCalibrator(config)
        raw_score = 0.7
        calibrated = calibrator.calibrate(raw_score)
        assert calibrated == raw_score


class TestConfidenceEvaluator:
    """测试完整评估流程"""
    
    @pytest.fixture
    def evaluator(self, tmp_path):
        # 创建临时配置文件
        config_path = tmp_path / "test_config.yaml"
        config_content = """
vector:
  metric_type: COSINE
  radius_threshold: 0.65
  topk_weights: exponential

kg:
  min_path_score: 0.3
  relation_weights:
    employed_by: 1.0

faithfulness:
  strategy: keyword
  keyword_min_overlap: 0.3

fusion:
  default_weights:
    vector: 0.4
    kg: 0.4
    faithfulness: 0.2

calibration:
  enabled: true
  method: temperature_scaling
  params:
    temperature: 1.2
"""
        config_path.write_text(config_content)
        return ConfidenceEvaluator(str(config_path))
    
    def test_full_evaluation(self, evaluator):
        """测试完整评估流程"""
        query = "员工年假有多少天？"
        vector_results = [
            {"score": 0.9, "metadata": {"category": "HR"}},
            {"score": 0.85, "metadata": {"category": "HR"}},
        ]
        kg_results = [
            {
                "path_weight": 0.85,
                "hops": 1,
                "entity_confidence": 0.9,
                "update_time": datetime.now(),
                "type": "Policy",
            },
        ]
        answer = "员工年假为 15 天"
        contexts = ["根据公司政策，全职员工每年享有 15 天带薪年假。"]
        
        class MockContext:
            category = "HR_Policy"
            query_type = "factual"
            user_query_count = 50
        
        result = asyncio.run(
            evaluator.evaluate(
                query, vector_results, kg_results, answer, contexts, MockContext()
            )
        )
        
        assert isinstance(result, ConfidenceResult)
        assert 0.0 <= result.confidence <= 1.0
        assert result.level in ["high", "medium", "low"]
        assert len(result.explanation) > 0
        assert len(result.audit_id) > 0
        assert "vector" in result.signals
        assert "kg" in result.signals
        assert "faithfulness" in result.signals


def run_tests():
    """运行所有测试"""
    pytest.main([__file__, "-v"])


if __name__ == "__main__":
    run_tests()
