"""
参数优化模块 - 基于btp_optimize.py精简

支持:
1. 网格搜索优化
2. 走查优化
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Type
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]]
    metric: str


def optimize_strategy(
    strategy_class: Any,
    data: pd.DataFrame,
    param_ranges: Dict[str, Dict[str, Any]],
    metric: str = 'sharpe',
    initial_capital: float = 100000.0,
    max_iterations: int = 500
) -> OptimizationResult:
    """
    网格搜索参数优化

    Args:
        strategy_class: 策略类
        data: OHLCV数据
        param_ranges: 参数范围 {'param_name': {'min': 1, 'max': 20, 'step': 1}}
        metric: 优化目标 ('sharpe', 'return', 'win_rate')
        initial_capital: 初始资金
        max_iterations: 最大迭代次数

    Returns:
        OptimizationResult
    """
    from .engine import BacktestEngine
    from .strategies import get_strategy

    param_names = list(param_ranges.keys())
    results = []
    best_score = -np.inf
    best_params = {}

    # 生成参数组合
    import itertools

    param_values = []
    for name, range_dict in param_ranges.items():
        min_val = range_dict.get('min', 1)
        max_val = range_dict.get('max', 20)
        step = range_dict.get('step', 1)
        values = list(np.arange(min_val, max_val + 0.01, step))
        if isinstance(values[0], float):
            values = [round(v, 2) for v in values]
        else:
            values = [int(v) for v in values]
        param_values.append(values)

    total_combinations = 1
    for v in param_values:
        total_combinations *= len(v)
    total_combinations = min(total_combinations, max_iterations)

    count = 0
    for combo in itertools.product(*param_values):
        if count >= max_iterations:
            break

        params = dict(zip(param_names, combo))

        try:
            engine = BacktestEngine(initial_capital=initial_capital)
            strategy = get_strategy(strategy_class, params)
            result = engine.run_strategy(data, strategy, 'SYMBOL')

            # 计算得分
            score = 0
            if metric == 'sharpe':
                score = result.sharpe_ratio if result.sharpe_ratio is not None else -np.inf
            elif metric == 'return':
                score = result.total_return if result.total_return is not None else -np.inf
            elif metric == 'win_rate':
                score = result.win_rate if result.win_rate is not None else -np.inf
            else:
                # 综合得分
                score = (result.sharpe_ratio or 0) * 0.4 + (result.total_return or 0) * 0.4 + (result.win_rate or 0) * 0.2

            results.append({
                'params': params,
                'score': score,
                'total_return': result.total_return,
                'sharpe': result.sharpe_ratio,
                'win_rate': result.win_rate,
            })

            if score > best_score:
                best_score = score
                best_params = params

        except Exception as e:
            results.append({
                'params': params,
                'score': -np.inf,
                'error': str(e)
            })

        count += 1

    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_results=results,
        metric=metric
    )


def walk_forward_optimize(
    strategy_class: Any,
    data: pd.DataFrame,
    param_ranges: Dict[str, Dict[str, Any]],
    n_splits: int = 5,
    train_ratio: float = 0.7,
    metric: str = 'sharpe',
    initial_capital: float = 100000.0
) -> Dict[str, Any]:
    """
    走查优化

    Args:
        strategy_class: 策略类
        data: OHLCV数据
        param_ranges: 参数范围
        n_splits: 分割数
        train_ratio: 训练比例
        metric: 优化目标
        initial_capital: 初始资金

    Returns:
        包含各折结果和汇总统计
    """
    from .engine import BacktestEngine
    from .strategies import get_strategy

    n = len(data)
    fold_size = n // n_splits
    results = []

    for i in range(n_splits):
        start = i * fold_size
        end = min(start + fold_size, n)

        if end - start < 20:
            continue

        split_point = start + int((end - start) * train_ratio)

        train_data = data.iloc[start:split_point]
        test_data = data.iloc[split_point:end]

        if len(train_data) < 20 or len(test_data) < 5:
            continue

        # 在训练集上优化
        opt_result = optimize_strategy(
            strategy_class=strategy_class,
            data=train_data,
            param_ranges=param_ranges,
            metric=metric,
            initial_capital=initial_capital,
            max_iterations=100
        )

        # 在测试集上测试
        engine = BacktestEngine(initial_capital=initial_capital)
        strategy = get_strategy(strategy_class, opt_result.best_params)
        test_result = engine.run_strategy(test_data, strategy, 'SYMBOL')

        results.append({
            'fold': i + 1,
            'train_start': str(train_data.index[0]),
            'train_end': str(train_data.index[-1]),
            'test_start': str(test_data.index[0]),
            'test_end': str(test_data.index[-1]),
            'best_params': opt_result.best_params,
            'train_score': opt_result.best_score,
            'test_return': test_result.total_return,
            'test_sharpe': test_result.sharpe_ratio,
            'test_win_rate': test_result.win_rate,
        })

    if not results:
        return {'error': 'No valid folds'}

    # 汇总
    test_returns = [r['test_return'] for r in results if r.get('test_return') is not None]
    test_sharpes = [r['test_sharpe'] for r in results if r.get('test_sharpe') is not None]

    return {
        'folds': results,
        'n_splits': len(results),
        'avg_test_return': np.mean(test_returns) if test_returns else 0,
        'avg_test_sharpe': np.mean(test_sharpes) if test_sharpes else 0,
    }