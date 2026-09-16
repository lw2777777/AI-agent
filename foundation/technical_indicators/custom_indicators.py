# foundation/technical_indicators/custom_indicators.py
"""
自定义指标 - 高级策略指标
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


def super_trend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0
) -> Dict[str, pd.Series]:
    """
    SuperTrend 指标
    
    Returns:
        {
            'trend': 趋势方向 (1=上升, -1=下降),
            'line': SuperTrend线,
            'upper': 上轨,
            'lower': 下轨
        }
    """
    # 计算ATR
    atr = _atr(high, low, close, period)
    
    # 计算上轨和下轨
    hl2 = (high + low) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr
    
    # 初始化
    trend = pd.Series(1, index=close.index)
    line = pd.Series(0.0, index=close.index)
    
    for i in range(1, len(close)):
        if close.iloc[i] > upper.iloc[i-1]:
            trend.iloc[i] = 1
        elif close.iloc[i] < lower.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]
        
        if trend.iloc[i] == 1:
            line.iloc[i] = lower.iloc[i]
            if upper.iloc[i] < upper.iloc[i-1]:
                upper.iloc[i] = upper.iloc[i-1]
        else:
            line.iloc[i] = upper.iloc[i]
            if lower.iloc[i] > lower.iloc[i-1]:
                lower.iloc[i] = lower.iloc[i-1]
    
    return {
        'trend': trend,
        'line': line,
        'upper': upper,
        'lower': lower
    }


def vwap_bands(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    num_bands: int = 3,
    std_mult: float = 1.0
) -> Dict[str, pd.Series]:
    """
    VWAP 带宽带 (VWAP Bands)
    
    Returns:
        {
            'vwap': VWAP,
            'upper_bands': 上轨列表,
            'lower_bands': 下轨列表
        }
    """
    # 计算VWAP
    typical = (high + low + close) / 3
    vwap = (typical * volume).cumsum() / volume.cumsum().replace(0, np.nan)
    
    # 计算标准差
    diff = (typical - vwap) ** 2
    weighted_var = (diff * volume).cumsum() / volume.cumsum().replace(0, np.nan)
    std = np.sqrt(weighted_var)
    
    # 生成带宽带
    upper_bands = []
    lower_bands = []
    
    for i in range(1, num_bands + 1):
        upper_bands.append(vwap + i * std_mult * std)
        lower_bands.append(vwap - i * std_mult * std)
    
    return {
        'vwap': vwap,
        'upper_bands': upper_bands,
        'lower_bands': lower_bands
    }


def pivot_points(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    method: str = 'standard'
) -> Dict[str, pd.Series]:
    """
    枢轴点 (Pivot Points)
    
    Methods:
        - standard: 标准枢轴点
        - fibonacci: 斐波那契枢轴点
        - camrilla: Camrilla枢轴点
    
    Returns:
        {
            'pivot': 枢轴点,
            'r1': 阻力1,
            'r2': 阻力2,
            'r3': 阻力3,
            's1': 支撑1,
            's2': 支撑2,
            's3': 支撑3
        }
    """
    # 前一天的高低收盘
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)
    
    if method == 'standard':
        pivot = (prev_high + prev_low + prev_close) / 3
        r1 = 2 * pivot - prev_low
        r2 = pivot + (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot - prev_low)
        s1 = 2 * pivot - prev_high
        s2 = pivot - (prev_high - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)
    
    elif method == 'fibonacci':
        pivot = (prev_high + prev_low + prev_close) / 3
        r1 = pivot + 0.382 * (prev_high - prev_low)
        r2 = pivot + 0.618 * (prev_high - prev_low)
        r3 = pivot + 1.000 * (prev_high - prev_low)
        s1 = pivot - 0.382 * (prev_high - prev_low)
        s2 = pivot - 0.618 * (prev_high - prev_low)
        s3 = pivot - 1.000 * (prev_high - prev_low)
    
    elif method == 'camrilla':
        # Camrilla枢轴点
        r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
        r2 = prev_close + (prev_high - prev_low) * 1.1 / 6
        r3 = prev_close + (prev_high - prev_low) * 1.1 / 4
        r4 = prev_close + (prev_high - prev_low) * 1.1 / 2
        s1 = prev_close - (prev_high - prev_low) * 1.1 / 12
        s2 = prev_close - (prev_high - prev_low) * 1.1 / 6
        s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
        s4 = prev_close - (prev_high - prev_low) * 1.1 / 2
        
        return {
            'pivot': prev_close,
            'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4,
            's1': s1, 's2': s2, 's3': s3, 's4': s4
        }
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        'pivot': pivot,
        'r1': r1, 'r2': r2, 'r3': r3,
        's1': s1, 's2': s2, 's3': s3
    }


def market_profile(
    price: pd.Series,
    volume: pd.Series,
    num_bins: int = 20
) -> Dict[str, pd.Series]:
    """
    市场轮廓 (Market Profile)
    
    Returns:
        {
            'poc': 控制点 (Point of Control),
            'value_area_high': 价值区上沿,
            'value_area_low': 价值区下沿,
            'bins': 价格区间,
            'volume_distribution': 成交量分布
        }
    """
    # 创建价格区间
    price_min = price.min()
    price_max = price.max()
    bin_width = (price_max - price_min) / num_bins
    bins = np.arange(price_min, price_max + bin_width, bin_width)
    
    # 分配成交量到价格区间
    bin_indices = np.digitize(price, bins)
    volume_distribution = pd.Series(0.0, index=range(len(bins)))
    
    for i, vol in enumerate(volume):
        bin_idx = bin_indices.iloc[i] - 1
        if 0 <= bin_idx < len(bins):
            volume_distribution.iloc[bin_idx] += vol
    
    # 找出控制点 (POC)
    poc_idx = volume_distribution.idxmax()
    poc = bins[poc_idx]
    
    # 计算价值区 (70%成交量)
    total_volume = volume_distribution.sum()
    sorted_indices = volume_distribution.sort_values(ascending=False).index
    cumulative_volume = 0
    value_area_bins = []
    
    for idx in sorted_indices:
        cumulative_volume += volume_distribution.iloc[idx]
        value_area_bins.append(idx)
        if cumulative_volume >= total_volume * 0.70:
            break
    
    value_area_high = max(bins[i] for i in value_area_bins)
    value_area_low = min(bins[i] for i in value_area_bins)
    
    return {
        'poc': poc,
        'value_area_high': value_area_high,
        'value_area_low': value_area_low,
        'bins': bins,
        'volume_distribution': volume_distribution
    }


def volume_profile(
    price: pd.Series,
    volume: pd.Series,
    num_bins: int = 50
) -> Dict[str, np.ndarray]:
    """
    成交量轮廓 (Volume Profile)
    
    Returns:
        {
            'price_levels': 价格水平,
            'volume_at_price': 各价格成交量,
            'poc': 控制点
        }
    """
    # 创建价格网格
    price_min = price.min()
    price_max = price.max()
    price_levels = np.linspace(price_min, price_max, num_bins)
    volume_at_price = np.zeros(num_bins)
    
    # 分配成交量
    for i in range(len(price)):
        # 找到最近的价格水平
        idx = np.argmin(np.abs(price_levels - price.iloc[i]))
        volume_at_price[idx] += volume.iloc[i]
    
    # 找出POC
    poc_idx = np.argmax(volume_at_price)
    poc = price_levels[poc_idx]
    
    return {
        'price_levels': price_levels,
        'volume_at_price': volume_at_price,
        'poc': poc
    }


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """计算ATR (内部辅助函数)"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# 便捷函数
def calculate_all_custom_indicators(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """计算所有自定义指标"""
    result = df.copy()
    
    # SuperTrend
    st = super_trend(
        result['high'], result['low'], result['close'],
        period=kwargs.get('st_period', 10),
        multiplier=kwargs.get('st_multiplier', 3.0)
    )
    result['st_trend'] = st['trend']
    result['st_line'] = st['line']
    
    # VWAP Bands
    if 'volume' in result.columns:
        vw = vwap_bands(
            result['high'], result['low'],
            result['close'], result['volume']
        )
        result['vwap'] = vw['vwap']
        for i, band in enumerate(vw['upper_bands']):
            result[f'vwap_upper_{i+1}'] = band
        for i, band in enumerate(vw['lower_bands']):
            result[f'vwap_lower_{i+1}'] = band
    
    # Pivot Points
    pivots = pivot_points(
        result['high'], result['low'], result['close'],
        method=kwargs.get('pivot_method', 'standard')
    )
    for key, value in pivots.items():
        result[f'pivot_{key}'] = value
    
    return result