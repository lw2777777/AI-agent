"""
Guardrails Module - 输入/输出护栏

功能:
1. PII检测和脱敏 (个人信息保护)
2. Prompt注入检测
3. 交易合规检查 (仓位限制、禁止交易)
4. 输出验证 (格式、范围、质量)

设计理念:
- 多层防御: 输入检查 + 输出检查
- 可配置: 根据不同场景调整严格程度
- 可审计: 记录所有违规事件
"""

from typing import Dict, Any, Optional, List, Tuple
import re
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class GuardrailViolation:
    """护栏违规记录"""
    rule: str
    message: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data: Dict[str, Any] = field(default_factory=dict)


class FinancialPIIGuardrail:
    """
    PII检测和脱敏
    
    检测:
    - SSN (社会安全号码)
    - 信用卡号
    - 银行账号
    - API密钥
    - 邮箱地址
    - 电话号码
    """
    
    PATTERNS = {
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'ssn_no_dash': r'\b\d{9}\b',
        'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        'bank_account': r'\b\d{8,17}\b',
        'api_key': r'\b(sk-|pk-|api[-_]?key)[\w-]{20,}\b',
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    }
    
    def __init__(self, redact: bool = True, block: bool = False):
        self.redact = redact
        self.block = block
        self.violations: List[GuardrailViolation] = []
    
    def check(self, text: str) -> Dict[str, Any]:
        """
        检查文本中的PII
        
        Returns:
            {
                'passed': bool,
                'pii_found': List[str],
                'redacted_text': str,
                'violations': List[GuardrailViolation]
            }
        """
        pii_found = []
        redacted_text = text
        violations = []
        
        for pii_type, pattern in self.PATTERNS.items():
            matches = list(re.finditer(pattern, text))
            if matches:
                pii_found.append(pii_type)
                if self.redact:
                    for match in matches:
                        redacted_text = redacted_text.replace(
                            match.group(),
                            f'[REDACTED_{pii_type.upper()}]'
                        )
                violations.append(GuardrailViolation(
                    rule=f'pii_{pii_type}',
                    message=f'Found {len(matches)} instance(s) of {pii_type}',
                    severity='high' if pii_type in ['ssn', 'credit_card'] else 'medium',
                    data={'type': pii_type, 'count': len(matches)}
                ))
        
        self.violations.extend(violations)
        
        return {
            'passed': len(pii_found) == 0 or not self.block,
            'pii_found': pii_found,
            'redacted_text': redacted_text if self.redact else text,
            'violations': violations
        }
    
    def get_violations(self) -> List[GuardrailViolation]:
        """获取所有违规记录"""
        return self.violations


class PromptInjectionGuardrail:
    """
    Prompt注入检测
    
    检测:
    - 指令覆盖尝试 ("ignore previous instructions")
    - 角色扮演 ("you are now")
    - 系统提示覆盖 ("system:")
    - 越狱模式
    """
    
    INJECTION_PATTERNS = [
        (r'(?i)ignore\s+(previous|all|above)\s+instructions?', 'instruction_override'),
        (r'(?i)disregard\s+(previous|all|above)', 'instruction_override'),
        (r'(?i)forget\s+(everything|all|previous)', 'memory_override'),
        (r'(?i)you\s+are\s+now\s+', 'role_override'),
        (r'(?i)act\s+as\s+(if\s+you\s+are|a)', 'role_override'),
        (r'(?i)pretend\s+(to\s+be|you\s+are)', 'role_override'),
        (r'(?i)new\s+instructions?:', 'instruction_override'),
        (r'(?i)system\s*:\s*', 'system_override'),
        (r'(?i)\[system\]', 'system_override'),
        (r'(?i)override\s+(mode|instructions?|rules?)', 'override'),
        (r'(?i)jailbreak|dan\s+mode|developer\s+mode', 'jailbreak'),
    ]
    
    def __init__(self, block: bool = True):
        self.block = block
        self.violations: List[GuardrailViolation] = []
    
    def check(self, text: str) -> Dict[str, Any]:
        """检查Prompt注入"""
        matched = []
        violations = []
        
        for pattern, rule in self.INJECTION_PATTERNS:
            if re.search(pattern, text):
                matched.append(rule)
                violations.append(GuardrailViolation(
                    rule=rule,
                    message=f'Detected injection pattern: {rule}',
                    severity='critical',
                    data={'pattern': pattern}
                ))
        
        self.violations.extend(violations)
        
        return {
            'passed': len(matched) == 0 or not self.block,
            'injection_detected': len(matched) > 0,
            'matched_patterns': matched,
            'violations': violations
        }
    
    def get_violations(self) -> List[GuardrailViolation]:
        return self.violations


class TradingComplianceGuardrail:
    """
    交易合规检查
    
    检查:
    - 仓位大小限制
    - 禁止交易列表
    - 市场操纵模式
    - 内幕交易指示
    """
    
    PROHIBITED_PATTERNS = [
        r'(?i)insider\s+(trading|information|tip)',
        r'(?i)pump\s+and\s+dump',
        r'(?i)front\s*run(ning)?',
        r'(?i)wash\s+trad(e|ing)',
        r'(?i)spoof(ing)?',
        r'(?i)layering',
        r'(?i)manipulat(e|ion|ing)\s+(market|price|stock)',
    ]
    
    def __init__(
        self,
        max_position_pct: float = 0.10,
        prohibited_symbols: List[str] = None,
        block: bool = True
    ):
        self.max_position_pct = max_position_pct
        self.prohibited_symbols = prohibited_symbols or []
        self.block = block
        self.violations: List[GuardrailViolation] = []
    
    def check(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """检查合规性"""
        violations = []
        warnings = []
        
        # 1. 禁止模式检测
        for pattern in self.PROHIBITED_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append(GuardrailViolation(
                    rule='prohibited_activity',
                    message=f'Prohibited activity pattern detected',
                    severity='critical',
                    data={'pattern': pattern}
                ))
        
        # 2. 禁止交易标的
        text_upper = text.upper()
        for symbol in self.prohibited_symbols:
            if symbol.upper() in text_upper:
                violations.append(GuardrailViolation(
                    rule='prohibited_symbol',
                    message=f'Prohibited symbol: {symbol}',
                    severity='high',
                    data={'symbol': symbol}
                ))
        
        # 3. 仓位大小检查
        if context:
            portfolio_value = context.get('portfolio_value', 0)
            proposed_value = context.get('proposed_trade_value', 0)
            
            if portfolio_value > 0 and proposed_value > 0:
                position_pct = proposed_value / portfolio_value
                if position_pct > self.max_position_pct:
                    violations.append(GuardrailViolation(
                        rule='position_size_limit',
                        message=f'Position size {position_pct*100:.1f}% exceeds limit {self.max_position_pct*100:.1f}%',
                        severity='medium',
                        data={'position_pct': position_pct, 'limit': self.max_position_pct}
                    ))
        
        self.violations.extend(violations)
        
        return {
            'passed': len(violations) == 0 or not self.block,
            'violations': violations,
            'warnings': warnings
        }
    
    def get_violations(self) -> List[GuardrailViolation]:
        return self.violations


class OutputValidationGuardrail:
    """
    输出验证
    
    检查:
    - 必要字段是否存在
    - 数值范围是否合理
    - 置信度是否在有效范围
    """
    
    def __init__(
        self,
        required_fields: List[str] = None,
        numeric_ranges: Dict[str, Tuple[float, float]] = None,
        max_confidence: float = 1.0
    ):
        self.required_fields = required_fields or []
        self.numeric_ranges = numeric_ranges or {}
        self.max_confidence = max_confidence
        self.violations: List[GuardrailViolation] = []
    
    def check(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """验证输出"""
        violations = []
        warnings = []
        
        # 1. 必要字段检查
        for field in self.required_fields:
            if field not in output or output[field] is None:
                violations.append(GuardrailViolation(
                    rule='missing_field',
                    message=f'Missing required field: {field}',
                    severity='medium',
                    data={'field': field}
                ))
        
        # 2. 数值范围检查
        for field, (min_val, max_val) in self.numeric_ranges.items():
            if field in output and output[field] is not None:
                val = output[field]
                if isinstance(val, (int, float)):
                    if val < min_val or val > max_val:
                        violations.append(GuardrailViolation(
                            rule='range_violation',
                            message=f'{field}={val} outside range [{min_val}, {max_val}]',
                            severity='medium',
                            data={'field': field, 'value': val, 'min': min_val, 'max': max_val}
                        ))
        
        # 3. 置信度检查
        if 'confidence' in output:
            conf = output['confidence']
            if isinstance(conf, (int, float)):
                if conf > self.max_confidence:
                    warnings.append(f'Confidence {conf} exceeds max {self.max_confidence}')
        
        self.violations.extend(violations)
        
        return {
            'passed': len(violations) == 0,
            'missing_fields': [v.data.get('field') for v in violations if v.rule == 'missing_field'],
            'range_violations': [v.data for v in violations if v.rule == 'range_violation'],
            'warnings': warnings,
            'violations': violations
        }


class GuardrailsModule:
    """
    综合护栏模块
    
    组合所有护栏，提供统一接口
    """
    
    def __init__(self):
        self.input_guardrails: List[Any] = []
        self.output_guardrails: List[Any] = []
        self.all_violations: List[GuardrailViolation] = []
        self._enabled = True
    
    def add_pii_protection(self, redact: bool = True, block: bool = False) -> 'GuardrailsModule':
        """添加PII保护"""
        self.input_guardrails.append(FinancialPIIGuardrail(redact=redact, block=block))
        return self
    
    def add_injection_protection(self, block: bool = True) -> 'GuardrailsModule':
        """添加注入保护"""
        self.input_guardrails.append(PromptInjectionGuardrail(block=block))
        return self
    
    def add_trading_compliance(
        self,
        max_position_pct: float = 0.10,
        prohibited_symbols: List[str] = None
    ) -> 'GuardrailsModule':
        """添加交易合规"""
        self.input_guardrails.append(TradingComplianceGuardrail(
            max_position_pct=max_position_pct,
            prohibited_symbols=prohibited_symbols
        ))
        return self
    
    def add_output_validation(
        self,
        required_fields: List[str] = None,
        numeric_ranges: Dict[str, Tuple[float, float]] = None
    ) -> 'GuardrailsModule':
        """添加输出验证"""
        self.output_guardrails.append(OutputValidationGuardrail(
            required_fields=required_fields,
            numeric_ranges=numeric_ranges
        ))
        return self
    
    def check_input(self, text: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """检查输入"""
        if not self._enabled:
            return {'passed': True, 'text': text, 'violations': []}
        
        all_violations = []
        current_text = text
        
        for guardrail in self.input_guardrails:
            if hasattr(guardrail, 'check'):
                if isinstance(guardrail, TradingComplianceGuardrail):
                    result = guardrail.check(current_text, context)
                else:
                    result = guardrail.check(current_text)
                
                violations = result.get('violations', [])
                all_violations.extend(violations)
                
                if result.get('redacted_text'):
                    current_text = result['redacted_text']
        
        self.all_violations.extend(all_violations)
        
        return {
            'passed': len(all_violations) == 0,
            'text': current_text,
            'violations': all_violations
        }
    
    def check_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """检查输出"""
        if not self._enabled:
            return {'passed': True, 'violations': [], 'warnings': []}
        
        all_violations = []
        warnings = []
        
        for guardrail in self.output_guardrails:
            if hasattr(guardrail, 'check'):
                result = guardrail.check(output)
                all_violations.extend(result.get('violations', []))
                warnings.extend(result.get('warnings', []))
        
        self.all_violations.extend(all_violations)
        
        return {
            'passed': len(all_violations) == 0,
            'violations': all_violations,
            'warnings': warnings
        }
    
    def enable(self) -> 'GuardrailsModule':
        """启用护栏"""
        self._enabled = True
        return self
    
    def disable(self) -> 'GuardrailsModule':
        """禁用护栏"""
        self._enabled = False
        return self
    
    def get_violations(self) -> List[GuardrailViolation]:
        """获取所有违规记录"""
        return self.all_violations.copy()
    
    def clear_violations(self) -> None:
        """清空违规记录"""
        self.all_violations = []
    
    def to_agent_config(self) -> Dict[str, Any]:
        """转换为Agent配置"""
        return {
            'guardrails_enabled': self._enabled,
            'input_guardrails_count': len(self.input_guardrails),
            'output_guardrails_count': len(self.output_guardrails)
        }
    
    @classmethod
    def default_financial(cls) -> 'GuardrailsModule':
        """默认金融护栏"""
        return (cls()
            .add_pii_protection(redact=True, block=False)
            .add_injection_protection(block=True)
            .add_trading_compliance(max_position_pct=0.10)
            .add_output_validation(
                required_fields=['symbol', 'action', 'reasoning'],
                numeric_ranges={
                    'confidence': (0.0, 1.0),
                    'quantity': (0.0, float('inf')),
                    'price': (0.0, float('inf'))
                }
            ))