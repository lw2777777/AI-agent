"""
信号融合工具
将多个来源的信号进行融合，生成综合交易信号
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    NEUTRAL = "neutral"


@dataclass
class SignalQuality:
    """信号质量"""
    source: str
    signal_type: SignalType
    confidence: float  # 0-1
    timestamp: datetime
    weight: float = 1.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class FusedSignal:
    """融合后的信号"""
    signal_type: SignalType
    confidence: float  # 0-1
    strength: float  # -1 to 1, 负值为卖出信号
    timestamp: datetime
    sources: List[str]
    source_signals: Dict[str, SignalQuality]
    reasoning: str
    metadata: Dict = field(default_factory=dict)


class SignalFusion:
    """信号融合器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.name = "SignalFusion"
        
        # 融合策略参数
        self.fusion_method = config.get('fusion_method', 'weighted_voting')
        self.consensus_threshold = config.get('consensus_threshold', 0.6)
        self.min_sources = config.get('min_sources', 2)
        
        # 信号源权重
        self.source_weights = config.get('source_weights', {
            'technical': 0.30,
            'fundamental': 0.20,
            'sentiment': 0.20,
            'momentum': 0.15,
            'volatility': 0.15
        })
        
        # 历史信号
        self.signal_history = []
        
    def add_signal(self, source: str, signal_type: SignalType, 
                   confidence: float, **kwargs) -> SignalQuality:
        """添加单个信号"""
        quality = SignalQuality(
            source=source,
            signal_type=signal_type,
            confidence=min(max(confidence, 0), 1),
            timestamp=datetime.now(),
            weight=self.source_weights.get(source, 1.0),
            metadata=kwargs
        )
        return quality
    
    def fuse_signals(self, signals: List[SignalQuality]) -> FusedSignal:
        """
        融合多个信号
        
        Args:
            signals: 信号列表
            
        Returns:
            FusedSignal: 融合后的信号
        """
        if not signals:
            return self._empty_fused_signal()
        
        # 根据融合方法处理
        if self.fusion_method == 'weighted_voting':
            result = self._fuse_weighted_voting(signals)
        elif self.fusion_method == 'consensus':
            result = self._fuse_consensus(signals)
        elif self.fusion_method == 'bayesian':
            result = self._fuse_bayesian(signals)
        else:
            result = self._fuse_weighted_voting(signals)
        
        # 记录历史
        self.signal_history.append(result)
        
        return result
    
    def _fuse_weighted_voting(self, signals: List[SignalQuality]) -> FusedSignal:
        """加权投票融合"""
        # 统计各信号类型的加权分数
        scores = {
            SignalType.BUY: 0,
            SignalType.SELL: 0,
            SignalType.HOLD: 0,
            SignalType.NEUTRAL: 0
        }
        
        total_weight = 0
        source_names = []
        
        for signal in signals:
            weight = signal.weight * signal.confidence
            scores[signal.signal_type] += weight
            total_weight += weight
            source_names.append(signal.source)
        
        if total_weight == 0:
            return self._empty_fused_signal()
        
        # 归一化
        for key in scores:
            scores[key] /= total_weight
        
        # 选择最高分的信号类型
        max_score = max(scores.values())
        # 检查是否有多个相同最高分
        max_types = [k for k, v in scores.items() if v == max_score]
        
        if len(max_types) > 1:
            # 如果有多个最高分，选择hold或neutral
            if SignalType.HOLD in scores and scores[SignalType.HOLD] == max_score:
                signal_type = SignalType.HOLD
            else:
                signal_type = SignalType.NEUTRAL
        else:
            signal_type = max(scores, key=lambda k: scores[k])
        
        # 计算置信度
        confidence = max_score
        
        # 计算强度（-1到1）
        strength = scores[SignalType.BUY] - scores[SignalType.SELL]
        
        # 生成推理
        reasoning = self._generate_reasoning(scores, signal_type)
        
        # 创建源信号字典
        source_signals = {s.source: s for s in signals}
        
        return FusedSignal(
            signal_type=signal_type,
            confidence=confidence,
            strength=strength,
            timestamp=datetime.now(),
            sources=list(set(source_names)),
            source_signals=source_signals,
            reasoning=reasoning,
            metadata={'scores': scores}
        )
    
    def _fuse_consensus(self, signals: List[SignalQuality]) -> FusedSignal:
        """共识融合 - 要求至少一定比例的信号一致"""
        # 统计信号类型
        counts = {
            SignalType.BUY: 0,
            SignalType.SELL: 0,
            SignalType.HOLD: 0,
            SignalType.NEUTRAL: 0
        }
        
        for signal in signals:
            counts[signal.signal_type] += 1
        
        total = len(signals)
        
        # 检查是否有信号类型达到共识阈值
        for signal_type, count in counts.items():
            if count / total >= self.consensus_threshold:
                # 达到共识
                return FusedSignal(
                    signal_type=signal_type,
                    confidence=count / total,
                    strength=(counts[SignalType.BUY] - counts[SignalType.SELL]) / total,
                    timestamp=datetime.now(),
                    sources=[s.source for s in signals],
                    source_signals={s.source: s for s in signals},
                    reasoning=f"Consensus reached: {count}/{total} signals agree on {signal_type.value}",
                    metadata={'counts': counts}
                )
        
        # 没有达到共识，返回neutral
        return FusedSignal(
            signal_type=SignalType.NEUTRAL,
            confidence=max(counts.values()) / total,
            strength=(counts[SignalType.BUY] - counts[SignalType.SELL]) / total,
            timestamp=datetime.now(),
            sources=[s.source for s in signals],
            source_signals={s.source: s for s in signals},
            reasoning=f"No consensus reached. Best: {max(counts, key=counts.get).value} ({max(counts.values())}/{total})",
            metadata={'counts': counts}
        )
    
    def _fuse_bayesian(self, signals: List[SignalQuality]) -> FusedSignal:
        """贝叶斯融合"""
        # 先验概率
        priors = {
            SignalType.BUY: 0.3,
            SignalType.SELL: 0.3,
            SignalType.HOLD: 0.3,
            SignalType.NEUTRAL: 0.1
        }
        
        # 更新后验
        posteriors = priors.copy()
        
        for signal in signals:
            # 似然函数（根据信号类型和置信度）
            for st in SignalType:
                if signal.signal_type == st:
                    likelihood = signal.confidence * signal.weight
                else:
                    likelihood = (1 - signal.confidence) * (1 - signal.weight * 0.5)
                
                posteriors[st] *= likelihood
        
        # 归一化
        total = sum(posteriors.values())
        for key in posteriors:
            posteriors[key] /= total
        
        # 选择后验概率最高的
        signal_type = max(posteriors, key=lambda k: posteriors[k])
        confidence = posteriors[signal_type]
        
        # 计算强度
        strength = posteriors[SignalType.BUY] - posteriors[SignalType.SELL]
        
        return FusedSignal(
            signal_type=signal_type,
            confidence=confidence,
            strength=strength,
            timestamp=datetime.now(),
            sources=[s.source for s in signals],
            source_signals={s.source: s for s in signals},
            reasoning=f"Bayesian fusion: highest posterior for {signal_type.value} ({confidence:.2%})",
            metadata={'posteriors': posteriors, 'priors': priors}
        )
    
    def _generate_reasoning(self, scores: Dict[SignalType, float], 
                           signal_type: SignalType) -> str:
        """生成推理说明"""
        reasons = []
        
        # 排序信号类型
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        reasons.append(f"Primary signal: {signal_type.value} (score: {scores[signal_type]:.2%})")
        
        # 添加第二和第三选择
        if len(sorted_scores) > 1:
            reasons.append(f"Secondary: {sorted_scores[1][0].value} ({sorted_scores[1][1]:.2%})")
        if len(sorted_scores) > 2:
            reasons.append(f"Tertiary: {sorted_scores[2][0].value} ({sorted_scores[2][1]:.2%})")
        
        # 计算优势比
        if signal_type == SignalType.BUY:
            advantage = scores[SignalType.BUY] / max(scores[SignalType.SELL], 0.01)
            reasons.append(f"Buy/Sell ratio: {advantage:.2f}x")
        elif signal_type == SignalType.SELL:
            advantage = scores[SignalType.SELL] / max(scores[SignalType.BUY], 0.01)
            reasons.append(f"Sell/Buy ratio: {advantage:.2f}x")
        
        return " | ".join(reasons)
    
    def _empty_fused_signal(self) -> FusedSignal:
        """空融合信号"""
        return FusedSignal(
            signal_type=SignalType.NEUTRAL,
            confidence=0,
            strength=0,
            timestamp=datetime.now(),
            sources=[],
            source_signals={},
            reasoning="No signals to fuse",
            metadata={}
        )
    
    def get_signal_quality(self, signal: FusedSignal) -> str:
        """获取信号质量评价"""
        if signal.confidence >= 0.8:
            return "Strong"
        elif signal.confidence >= 0.6:
            return "Moderate"
        elif signal.confidence >= 0.4:
            return "Weak"
        else:
            return "Very Weak"
    
    def should_trade(self, signal: FusedSignal) -> Tuple[bool, str]:
        """判断是否应该交易"""
        if signal.confidence < self.consensus_threshold:
            return False, f"Confidence too low: {signal.confidence:.2%}"
        
        if abs(signal.strength) < 0.1:
            return False, f"Signal strength too weak: {signal.strength:.3f}"
        
        if signal.signal_type in [SignalType.NEUTRAL, SignalType.HOLD]:
            return False, f"Signal type: {signal.signal_type.value}"
        
        if len(signal.sources) < self.min_sources:
            return False, f"Insufficient sources: {len(signal.sources)}"
        
        return True, "All conditions met"