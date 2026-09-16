# foundation/technical_indicators/trend_indicators.py
"""
趋势指标 - 完整实现
"""

import pandas as pd
import numpy as np
from typing import Dict, Union, Optional


def sma(close: pd.Series, period: int = 20) -> pd.Series:
    """简单移动平均线"""
    return close.rolling(window=period).mean()


def ema(close: pd.Series, period: int = 20) -> pd.Series:
    """指数移动平均线"""
    return close.ewm(span=period, adjust=False).mean()


def wma(close: pd.Series, period: int = 20) -> pd.Series:
    """加权移动平均线"""
    weights = np.arange(1, period + 1, dtype=float)
    return close.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def hma(close: pd.Series, period: int = 20) -> pd.Series:
    """赫尔移动平均线 (Hull Moving Average)"""
    half = int(period / 2)
    sqrt_p = int(np.sqrt(period))
    wmaf = wma(close, half)
    wmas = wma(close, period)
    diff = 2 * wmaf - wmas
    return wma(diff, sqrt_p)


def dema(close: pd.Series, period: int = 20) -> pd.Series:
    """双指数移动平均线"""
    e = ema(close, period)
    return 2 * e - ema(e, period)


def tema(close: pd.Series, period: int = 20) -> pd.Series:
    """三指数移动平均线"""
    e1 = ema(close, period)
    e2 = ema(e1, period)
    e3 = ema(e2, period)
    return 3 * e1 - 3 * e2 + e3


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Dict[str, pd.Series]:
    """
    MACD (Moving Average Convergence Divergence)
    
    Returns:
        {
            'macd': MACD线,
            'signal': 信号线,
            'histogram': 柱状图
        }
    """
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14
) -> Dict[str, pd.Series]:
    """
    ADX (Average Directional Index)
    
    Returns:
        {
            'adx': ADX值,
            'plus_di': +DI,
            'minus_di': -DI
        }
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    
    # True Range
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    
    # Directional Movement
    plus_dm = (high - prev_high).where(
        (high - prev_high) > (prev_low - low), 0.0
    ).clip(lower=0)
    
    minus_dm = (prev_low - low).where(
        (prev_low - low) > (high - prev_high), 0.0
    ).clip(lower=0)
    
    # Average True Range
    atr_val = tr.ewm(span=period, adjust=False).mean()
    
    # Directional Indicators
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_val.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_val.replace(0, np.nan))
    
    # ADX
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    adx_val = dx.ewm(span=period, adjust=False).mean()
    
    return {
        'adx': adx_val,
        'plus_di': plus_di,
        'minus_di': minus_di
    }


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52
) -> Dict[str, pd.Series]:
    """
    一目均衡表 (Ichimoku Cloud)
    
    Returns:
        {
            'tenkan_sen': 转换线,
            'kijun_sen': 基准线,
            'senkou_a': 先行带A,
            'senkou_b': 先行带B,
            'chikou': 延迟线
        }
    """
    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    senkou_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun)
    senkou_b_val = ((high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2).shift(kijun)
    chikou = close.shift(-kijun)
    
    return {
        'tenkan_sen': tenkan_sen,
        'kijun_sen': kijun_sen,
        'senkou_a': senkou_a,
        'senkou_b': senkou_b_val,
        'chikou': chikou
    }


def parabolic_sar(
    high: pd.Series,
    low: pd.Series,
    af_step: float = 0.02,
    af_max: float = 0.2
) -> pd.Series:
    """
    抛物线SAR (Parabolic SAR)
    """
    length = len(high)
    sar = np.zeros(length)
    af = af_step
    bull = True
    ep = low.iloc[0]
    hp = high.iloc[0]
    lp = low.iloc[0]
    sar[0] = high.iloc[0]
    
    for i in range(1, length):
        if bull:
            sar[i] = sar[i-1] + af * (hp - sar[i-1])
            sar[i] = min(sar[i], low.iloc[i-1])
            if i >= 2:
                sar[i] = min(sar[i], low.iloc[i-2])
            
            if low.iloc[i] < sar[i]:
                bull = False
                sar[i] = hp
                lp = low.iloc[i]
                af = af_step
            else:
                if high.iloc[i] > hp:
                    hp = high.iloc[i]
                    af = min(af + af_step, af_max)
        else:
            sar[i] = sar[i-1] + af * (lp - sar[i-1])
            sar[i] = max(sar[i], high.iloc[i-1])
            if i >= 2:
                sar[i] = max(sar[i], high.iloc[i-2])
            
            if high.iloc[i] > sar[i]:
                bull = True
                sar[i] = lp
                hp = high.iloc[i]
                af = af_step
            else:
                if low.iloc[i] < lp:
                    lp = low.iloc[i]
                    af = min(af + af_step, af_max)
    
    return pd.Series(sar, index=high.index)


def calculate_all_trend_indicators(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """计算所有趋势指标"""
    result = df.copy()
    
    # SMA
    result['sma_20'] = sma(df['close'], 20)
    result['sma_50'] = sma(df['close'], 50)
    result['sma_200'] = sma(df['close'], 200)
    
    # EMA
    result['ema_12'] = ema(df['close'], 12)
    result['ema_26'] = ema(df['close'], 26)
    
    # MACD
    macd_result = macd(df['close'])
    result['macd'] = macd_result['macd']
    result['macd_signal'] = macd_result['signal']
    result['macd_hist'] = macd_result['histogram']
    
    # ADX
    if 'high' in df and 'low' in df:
        adx_result = adx(df['high'], df['low'], df['close'])
        result['adx'] = adx_result['adx']
        result['plus_di'] = adx_result['plus_di']
        result['minus_di'] = adx_result['minus_di']
    
    return result