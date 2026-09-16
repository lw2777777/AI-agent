"""
信号生成模块 - 基于btp_signals.py精简

功能:
1. 交叉检测 (crossover/crossunder)
2. 阈值信号
3. 突破信号
4. 信号转交易
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional


def check_crossover(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """检测向上交叉 (series1 上穿 series2)"""
    prev1 = series1.shift(1)
    prev2 = series2.shift(1)
    return (prev1 <= prev2) & (series1 > series2)


def check_crossunder(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """检测向下交叉 (series1 下穿 series2)"""
    prev1 = series1.shift(1)
    prev2 = series2.shift(1)
    return (prev1 >= prev2) & (series1 < series2)


def check_cross(series1: pd.Series, series2: pd.Series) -> pd.Series:
    """检测交叉 (任意方向)"""
    return check_crossover(series1, series2) | check_crossunder(series1, series2)


def generate_crossover_signals(fast: pd.Series, slow: pd.Series) -> Dict[str, pd.Series]:
    """生成交叉信号"""
    entries = check_crossover(fast, slow)
    exits = check_crossunder(fast, slow)
    return {'entries': entries, 'exits': exits}


def generate_threshold_signals(indicator: pd.Series,
                               lower: float, upper: float) -> Dict[str, pd.Series]:
    """生成阈值信号 (均值回归)"""
    entries = indicator < lower
    exits = indicator > upper
    return {'entries': entries, 'exits': exits}


def generate_breakout_signals(close: pd.Series,
                              upper: pd.Series,
                              lower: pd.Series) -> Dict[str, pd.Series]:
    """生成突破信号"""
    entries = close > upper
    exits = close < lower
    return {'entries': entries, 'exits': exits}


def signals_to_trades(
    signals: Dict[str, pd.Series],
    prices: pd.Series,
    initial_capital: float = 100000.0
) -> List[Dict[str, Any]]:
    """
    将信号转换为交易记录

    Args:
        signals: {'entries': bool_series, 'exits': bool_series}
        prices: 价格序列
        initial_capital: 初始资金

    Returns:
        交易列表
    """
    entries = signals.get('entries', pd.Series(False, index=prices.index))
    exits = signals.get('exits', pd.Series(False, index=prices.index))

    trades = []
    position = 0
    entry_price = 0
    capital = initial_capital

    for idx in prices.index:
        if entries.loc[idx] and position == 0:
            # 开仓
            position = 1
            entry_price = prices.loc[idx]
            trades.append({
                'entry_date': str(idx),
                'entry_price': float(entry_price),
                'action': 'BUY',
                'quantity': int(capital * 0.9 / entry_price)
            })

        elif exits.loc[idx] and position > 0:
            # 平仓
            exit_price = prices.loc[idx]
            trade = trades[-1]
            trade['exit_date'] = str(idx)
            trade['exit_price'] = float(exit_price)
            trade['pnl'] = float((exit_price - entry_price) * trade['quantity'])
            trade['return'] = float((exit_price - entry_price) / entry_price * 100)
            position = 0

    return trades