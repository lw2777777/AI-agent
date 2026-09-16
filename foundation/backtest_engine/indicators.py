"""
技术指标模块 - 基于btp_indicators.py精简

包含最常用的30+指标:
- 趋势: SMA, EMA, MACD
- 动量: RSI, Stochastic, Williams %R
- 波动率: Bollinger Bands, ATR
- 成交量: OBV, MFI
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List


# ==================== 趋势指标 ====================

def sma(close: pd.Series, period: int = 20) -> pd.Series:
    """简单移动平均"""
    return close.rolling(window=period).mean()


def ema(close: pd.Series, period: int = 20) -> pd.Series:
    """指数移动平均"""
    return close.ewm(span=period, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """
    MACD指标

    Returns:
        {'macd': macd线, 'signal': 信号线, 'histogram': 柱状图}
    """
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return {'macd': macd_line, 'signal': signal_line, 'histogram': histogram}


# ==================== 动量指标 ====================

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """相对强弱指标"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
    """随机指标"""
    lowest = low.rolling(window=k_period).min()
    highest = high.rolling(window=k_period).max()
    denom = (highest - lowest).replace(0, np.nan)
    k = ((close - lowest) / denom) * 100
    d = k.rolling(window=d_period).mean()
    return {'k': k, 'd': d}


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 14) -> pd.Series:
    """威廉指标 %R"""
    highest = high.rolling(window=period).max()
    lowest = low.rolling(window=period).min()
    denom = (highest - lowest).replace(0, np.nan)
    return -100 * (highest - close) / denom


def momentum(close: pd.Series, period: int = 10) -> pd.Series:
    """动量指标 (百分比变化)"""
    return close.pct_change(periods=period) * 100


# ==================== 波动率指标 ====================

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """平均真实波幅"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
    """布林带"""
    middle = sma(close, period)
    std = close.rolling(window=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return {'upper': upper, 'middle': middle, 'lower': lower}


# ==================== 成交量指标 ====================

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """能量潮"""
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    return (volume * direction).cumsum()


def mfi(high: pd.Series, low: pd.Series, close: pd.Series,
        volume: pd.Series, period: int = 14) -> pd.Series:
    """资金流量指数"""
    tp = (high + low + close) / 3.0
    rmf = tp * volume
    positive_flow = rmf.where(tp > tp.shift(1), 0.0).rolling(period).sum()
    negative_flow = rmf.where(tp < tp.shift(1), 0.0).rolling(period).sum()
    ratio = positive_flow / negative_flow.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))


# ==================== 指标注册表 ====================

INDICATOR_CATALOG = {
    'sma': {'label': 'SMA', 'category': 'Trend', 'params': ['period']},
    'ema': {'label': 'EMA', 'category': 'Trend', 'params': ['period']},
    'macd': {'label': 'MACD', 'category': 'Trend', 'params': ['fast', 'slow', 'signal']},
    'rsi': {'label': 'RSI', 'category': 'Momentum', 'params': ['period']},
    'stoch': {'label': 'Stochastic', 'category': 'Momentum', 'params': ['k_period', 'd_period']},
    'williams_r': {'label': 'Williams %R', 'category': 'Momentum', 'params': ['period']},
    'momentum': {'label': 'Momentum', 'category': 'Momentum', 'params': ['period']},
    'atr': {'label': 'ATR', 'category': 'Volatility', 'params': ['period']},
    'bbands': {'label': 'Bollinger Bands', 'category': 'Volatility', 'params': ['period', 'std_dev']},
    'obv': {'label': 'OBV', 'category': 'Volume', 'params': []},
    'mfi': {'label': 'MFI', 'category': 'Volume', 'params': ['period']},
}


def calculate_indicator(indicator_type: str, data: pd.DataFrame,
                        params: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    统一指标计算接口

    Args:
        indicator_type: 指标类型 (如 'rsi', 'macd')
        data: OHLCV DataFrame
        params: 参数

    Returns:
        {'success': bool, 'values': [...], 'error': str}
    """
    params = params or {}
    close = data['Close']
    high = data.get('High', close)
    low = data.get('Low', close)
    volume = data.get('Volume', pd.Series(0, index=close.index))

    ind = indicator_type.lower()
    result_series = None
    result_multi = None

    # 趋势
    if ind == 'sma':
        result_series = sma(close, params.get('period', 20))
    elif ind == 'ema':
        result_series = ema(close, params.get('period', 20))
    elif ind == 'macd':
        result_multi = macd(close, params.get('fast', 12),
                           params.get('slow', 26), params.get('signal', 9))

    # 动量
    elif ind == 'rsi':
        result_series = rsi(close, params.get('period', 14))
    elif ind == 'stoch':
        result_multi = stochastic(high, low, close,
                                 params.get('k_period', 14), params.get('d_period', 3))
    elif ind == 'williams_r':
        result_series = williams_r(high, low, close, params.get('period', 14))
    elif ind == 'momentum':
        result_series = momentum(close, params.get('period', 10))

    # 波动率
    elif ind == 'atr':
        result_series = atr(high, low, close, params.get('period', 14))
    elif ind in ('bbands', 'bollinger'):
        result_multi = bollinger_bands(close, params.get('period', 20), params.get('std_dev', 2.0))

    # 成交量
    elif ind == 'obv':
        result_series = obv(close, volume)
    elif ind == 'mfi':
        result_series = mfi(high, low, close, volume, params.get('period', 14))

    else:
        return {'success': False, 'error': f'Unknown indicator: {indicator_type}'}

    # 格式化输出
    values = []
    if result_series is not None:
        for idx, val in result_series.items():
            if pd.notna(val):
                values.append({'date': str(idx), 'value': float(val)})
    elif result_multi is not None:
        for idx in data.index:
            point = {'date': str(idx)}
            all_nan = True
            for key, series in result_multi.items():
                v = series.loc[idx] if idx in series.index else np.nan
                if pd.notna(v):
                    point[key] = float(v)
                    all_nan = False
                else:
                    point[key] = None
            if not all_nan:
                values.append(point)

    return {'success': True, 'indicator': indicator_type, 'values': values, 'count': len(values)}


def get_indicator_catalog() -> List[Dict[str, Any]]:
    """获取指标目录"""
    return [
        {'id': key, 'label': info['label'], 'category': info['category'], 'params': info['params']}
        for key, info in INDICATOR_CATALOG.items()
    ]