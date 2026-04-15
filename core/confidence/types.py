"""
置信度评估数据类型定义
"""
from dataclasses import dataclass, field
from typing import Literal, Optional, Dict, Any, List
from datetime import datetime


@dataclass
class ConfidenceSignal:
    """单维度置信度信号"""
    score: float  # [0, 1]
    metadata: Dict[str, Any] = field(default_factory=dict)  # 可解释性元数据
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RequestContext:
    """请求上下文信息"""
    user_id: str
    category: str = ""  # 文档分类
    query_type: str = "general"  # 问题类型：general / entity / relation
    user_query_count: int = 0  # 用户历史查询次数
    role: str = "user"  # 用户角色
    request_id: str = ""  # 请求 ID
    
    def __post_init__(self):
        if not self.request_id:
            import uuid
            self.request_id = str(uuid.uuid4())[:8]


@dataclass
class ConfidenceResult:
    """最终置信度评估结果"""
    confidence: float  # [0, 1]
    level: Literal["high", "medium", "low"]
    explanation: str  # 自然语言解释
    audit_id: str  # 审计追踪 ID
    signals: Dict[str, ConfidenceSignal]  # 各维度原始信号
    suggestions: List[str] = field(default_factory=list)  # 低置信时的建议
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式（用于 API 响应）"""
        return {
            "confidence": self.confidence,
            "level": self.level,
            "explanation": self.explanation,
            "audit_id": self.audit_id,
            "signals": {
                k: {"score": v.score, "metadata": v.metadata}
                for k, v in self.signals.items()
            },
            "suggestions": self.suggestions
        }
