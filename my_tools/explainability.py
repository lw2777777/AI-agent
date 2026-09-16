"""
可解释性工具
为交易决策提供详细的解释和分析
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class DecisionExplanation:
    """决策解释"""
    decision_type: str  # 'buy', 'sell', 'hold'
    timestamp: datetime
    summary: str
    detailed_reasons: List[str]
    key_factors: Dict[str, float]  # 因素名 -> 重要性
    contributing_agents: List[str]
    risk_justification: str
    alternative_actions: List[str]
    confidence_level: str  # 'high', 'medium', 'low'
    supporting_data: Dict[str, Any]
    uncertainty_factors: List[str]


class ExplainabilityEngine:
    """可解释性引擎"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.name = "ExplainabilityEngine"
        self.explanation_history = []
        
    def explain_decision(self, decision: Dict, context: Dict, 
                         agent_results: Dict) -> DecisionExplanation:
        """
        解释决策
        
        Args:
            decision: 最终决策
            context: 上下文信息
            agent_results: 各Agent的结果
            
        Returns:
            DecisionExplanation: 决策解释
        """
        try:
            # 提取关键信息
            decision_type = decision.get('action', 'hold')
            
            # 生成关键因素
            key_factors = self._extract_key_factors(decision, context, agent_results)
            
            # 生成详细原因
            detailed_reasons = self._generate_detailed_reasons(
                decision, context, agent_results
            )
            
            # 生成摘要
            summary = self._generate_summary(decision, detailed_reasons, key_factors)
            
            # 风险评估
            risk_justification = self._justify_risk(decision, context)
            
            # 替代方案
            alternatives = self._generate_alternatives(decision, context)
            
            # 置信度
            confidence_level = self._assess_confidence(decision, agent_results)
            
            # 不确定性因素
            uncertainty_factors = self._identify_uncertainties(
                decision, context, agent_results
            )
            
            explanation = DecisionExplanation(
                decision_type=decision_type,
                timestamp=datetime.now(),
                summary=summary,
                detailed_reasons=detailed_reasons,
                key_factors=key_factors,
                contributing_agents=list(agent_results.keys()),
                risk_justification=risk_justification,
                alternative_actions=alternatives,
                confidence_level=confidence_level,
                supporting_data={
                    'market_data_summary': self._summarize_market_data(context.get('data', {})),
                    'agent_votes': {k: v.get('action', 'unknown') for k, v in agent_results.items()},
                    'risk_assessment': context.get('risk_assessment', {})
                },
                uncertainty_factors=uncertainty_factors
            )
            
            self.explanation_history.append(explanation)
            return explanation
            
        except Exception as e:
            logger.error(f"Explanation generation error: {e}")
            return self._default_explanation(decision)
    
    def _extract_key_factors(self, decision: Dict, context: Dict, 
                             agent_results: Dict) -> Dict[str, float]:
        """提取关键因素及其重要性"""
        factors = {}
        
        # 技术指标因素
        if 'technical' in agent_results:
            tech = agent_results['technical']
            if 'rsi' in tech:
                factors['RSI'] = min(abs(tech['rsi'] - 50) / 50, 1)
            if 'momentum' in tech:
                factors['Momentum'] = abs(tech['momentum'])
            if 'trend' in tech:
                factors['Trend'] = 0.8 if tech['trend'] == 'bullish' else 0.6
        
        # 市场因素
        if 'market_condition' in context:
            factors['Market_Condition'] = context['market_condition'].get('score', 0.5)
        
        # 风险因素
        if 'risk_assessment' in context:
            risk = context['risk_assessment']
            factors['Risk'] = 1 - risk.get('risk_score', 0.5) / 10
        
        # Agent一致性
        votes = [v.get('action', 'neutral') for v in agent_results.values()]
        consensus = sum(1 for v in votes if v == decision.get('action', 'hold')) / len(votes) if votes else 0
        factors['Agent_Consensus'] = consensus
        
        # 归一化
        total = sum(factors.values()) if factors else 1
        for key in factors:
            factors[key] = factors[key] / total
        
        return factors
    
    def _generate_detailed_reasons(self, decision: Dict, context: Dict,
                                   agent_results: Dict) -> List[str]:
        """生成详细原因"""
        reasons = []
        
        decision_type = decision.get('action', 'hold')
        confidence = decision.get('confidence', 0.5)
        
        # 主要决策原因
        if decision_type == 'buy':
            reasons.append(f"Bullish signal with {confidence:.1%} confidence")
            reasons.append("Technical indicators suggest upward momentum")
            if context.get('market_condition', {}).get('trend') == 'bullish':
                reasons.append("Market trend is favorable")
        elif decision_type == 'sell':
            reasons.append(f"Bearish signal with {confidence:.1%} confidence")
            reasons.append("Technical indicators suggest downward momentum")
            if context.get('market_condition', {}).get('trend') == 'bearish':
                reasons.append("Market trend is unfavorable")
        else:
            reasons.append("Decision to hold based on mixed signals")
            reasons.append("Insufficient confidence to enter new position")
        
        # Agent贡献
        for agent_name, result in agent_results.items():
            if 'reasoning' in result:
                reasons.append(f"{agent_name}: {result['reasoning']}")
        
        # 风险管理
        if 'risk_assessment' in context:
            risk = context['risk_assessment']
            if risk.get('risk_level') == 'low':
                reasons.append("Risk level is low, enabling position taking")
            elif risk.get('risk_level') == 'high':
                reasons.append("High risk level detected, cautious approach recommended")
        
        return reasons
    
    def _generate_summary(self, decision: Dict, reasons: List[str],
                         factors: Dict[str, float]) -> str:
        """生成摘要"""
        decision_type = decision.get('action', 'hold')
        confidence = decision.get('confidence', 0.5)
        
        # 主要因素
        top_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)[:3]
        factor_str = ", ".join([f"{k}({v:.1%})" for k, v in top_factors])
        
        summary = f"Decision: {decision_type.upper()} (conf: {confidence:.1%}). " \
                  f"Key factors: {factor_str}. " \
                  f"Based on {len(reasons)} considerations."
        
        return summary
    
    def _justify_risk(self, decision: Dict, context: Dict) -> str:
        """证明风险合理性"""
        if 'risk_assessment' not in context:
            return "Risk assessment not available"
        
        risk = context['risk_assessment']
        risk_level = risk.get('risk_level', 'medium')
        risk_score = risk.get('risk_score', 5)
        
        if risk_level == 'low':
            return f"Risk is acceptable (score: {risk_score}/10). Position size within limits."
        elif risk_level == 'medium':
            return f"Moderate risk (score: {risk_score}/10). Stop-loss in place to manage downside."
        else:
            return f"High risk (score: {risk_score}/10). This decision is risk-limited with strict stop-loss."
    
    def _generate_alternatives(self, decision: Dict, context: Dict) -> List[str]:
        """生成替代方案"""
        alternatives = []
        decision_type = decision.get('action', 'hold')
        
        if decision_type == 'buy':
            alternatives.append("Wait for pullback before entering")
            alternatives.append("Reduce position size by 50%")
            alternatives.append("Set tighter stop-loss")
        elif decision_type == 'sell':
            alternatives.append("Partial position exit")
            alternatives.append("Wait for higher price before selling")
            alternatives.append("Use trailing stop instead of market sell")
        else:  # hold
            alternatives.append("Enter with small position for testing")
            alternatives.append("Wait for clearer signal")
            alternatives.append("Look for other opportunities")
        
        return alternatives
    
    def _assess_confidence(self, decision: Dict, agent_results: Dict) -> str:
        """评估置信度级别"""
        confidence = decision.get('confidence', 0.5)
        
        if confidence >= 0.8:
            return 'high'
        elif confidence >= 0.6:
            return 'medium'
        else:
            return 'low'
    
    def _identify_uncertainties(self, decision: Dict, context: Dict,
                                agent_results: Dict) -> List[str]:
        """识别不确定性因素"""
        uncertainties = []
        
        # 数据不足
        if context.get('data_quality', 'good') != 'good':
            uncertainties.append("Limited market data available")
        
        # Agent不一致
        votes = [v.get('action', 'neutral') for v in agent_results.values()]
        if len(set(votes)) > 1:
            uncertainties.append("Agents have conflicting signals")
        
        # 市场不确定性
        if context.get('market_condition', {}).get('volatility', 0) > 0.03:
            uncertainties.append("High market volatility")
        
        # 置信度低
        if decision.get('confidence', 0.5) < 0.7:
            uncertainties.append("Low decision confidence")
        
        if not uncertainties:
            uncertainties.append("No significant uncertainties identified")
        
        return uncertainties
    
    def _summarize_market_data(self, data: Dict) -> Dict:
        """汇总市场数据"""
        if not data:
            return {'status': 'Not available'}
        
        return {
            'price': data.get('price', 0),
            'volume': data.get('volume', 0),
            'change_24h': data.get('change_24h', 0),
            'timestamp': datetime.now().isoformat()
        }
    
    def _default_explanation(self, decision: Dict) -> DecisionExplanation:
        """默认解释"""
        return DecisionExplanation(
            decision_type=decision.get('action', 'hold'),
            timestamp=datetime.now(),
            summary="Decision explanation not available",
            detailed_reasons=["Unable to generate detailed reasons"],
            key_factors={},
            contributing_agents=[],
            risk_justification="Risk justification not available",
            alternative_actions=["Consult manual for alternatives"],
            confidence_level='low',
            supporting_data={},
            uncertainty_factors=["Unknown uncertainties"]
        )
    
    def format_explanation(self, explanation: DecisionExplanation) -> str:
        """格式化输出"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"📋 决策解释 - {explanation.decision_type.upper()}")
        lines.append("=" * 70)
        
        lines.append(f"\n📝 摘要: {explanation.summary}")
        lines.append(f"\n🟢 置信度: {explanation.confidence_level.upper()}")
        
        lines.append("\n🔍 详细原因:")
        for i, reason in enumerate(explanation.detailed_reasons, 1):
            lines.append(f"  {i}. {reason}")
        
        lines.append("\n📊 关键因素:")
        for factor, importance in sorted(explanation.key_factors.items(), 
                                        key=lambda x: x[1], reverse=True):
            bars = "█" * int(importance * 20)
            lines.append(f"  {factor}: {bars} {importance:.1%}")
        
        lines.append(f"\n🛡️ 风险管理: {explanation.risk_justification}")
        
        lines.append("\n🔄 替代方案:")
        for i, alt in enumerate(explanation.alternative_actions, 1):
            lines.append(f"  {i}. {alt}")
        
        if explanation.uncertainty_factors:
            lines.append("\n⚠️ 不确定性因素:")
            for factor in explanation.uncertainty_factors:
                lines.append(f"  • {factor}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)