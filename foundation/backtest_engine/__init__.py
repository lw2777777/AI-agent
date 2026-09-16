"""
Backtest Engine - 轻量级回测引擎

核心功能:
1. 数据加载 (Yahoo Finance / 合成数据)
2. 技术指标计算 (30+ 指标)
3. 策略回测 (支持日内)
4. 绩效分析
5. 参数优化
6. Agent桥接接口
"""

from .data import (
    load_data,
    load_data_batch,
    random_ohlc_data,
    generate_intraday_data,
    resample_ohlcv,
    get_data_info,
    quick_data_preview,
    get_daily_data,
    get_intraday_data,
    clear_cache
)

from .indicators import (
    sma, ema, rsi, macd, bollinger_bands, atr,
    stochastic, williams_r, momentum, obv, mfi,
    calculate_indicator, get_indicator_catalog
)

from .strategies import (
    build_sma_crossover,
    build_ema_crossover,
    build_rsi,
    build_macd,
    build_bollinger_bands,
    build_mean_reversion,
    build_momentum,
    build_breakout,
    build_stochastic,
    build_adx_trend,
    build_williams_r,
    build_cci,
    build_obv_trend,
    build_keltner_breakout,
    build_triple_ma,
    get_strategy_class,
    get_strategy_catalog,
    map_optimize_params,
)

from .engine import BacktestEngine, BacktestResult

from .optimize import optimize_strategy, walk_forward_optimize

from .signals import (
    check_crossover, check_crossunder, check_cross,
    generate_crossover_signals, generate_threshold_signals,
    generate_breakout_signals, signals_to_trades
)

from .bridge import (
    BacktestBridge,
    run_backtest_for_agent,
    quick_backtest
)

__all__ = [
    # 数据
    'load_data', 'load_data_batch',
    'random_ohlc_data', 'generate_intraday_data',
    'resample_ohlcv', 'get_data_info',
    'quick_data_preview', 'get_daily_data', 'get_intraday_data',
    'clear_cache',
    
    # 指标
    'sma', 'ema', 'rsi', 'macd', 'bollinger_bands', 'atr',
    'stochastic', 'williams_r', 'momentum', 'obv', 'mfi',
    'calculate_indicator', 'get_indicator_catalog',
    
    # 策略
    'build_sma_crossover', 'build_ema_crossover', 'build_rsi', 'build_macd',
    'build_bollinger_bands', 'build_mean_reversion', 'build_momentum',
    'build_breakout', 'build_stochastic', 'build_adx_trend',
    'build_williams_r', 'build_cci', 'build_obv_trend',
    'build_keltner_breakout', 'build_triple_ma',
    'get_strategy_class', 'get_strategy_catalog', 'map_optimize_params',
    
    # 引擎
    'BacktestEngine', 'BacktestResult',
    
    # 优化
    'optimize_strategy', 'walk_forward_optimize',
    
    # 信号
    'check_crossover', 'check_crossunder', 'check_cross',
    'generate_crossover_signals', 'generate_threshold_signals',
    'generate_breakout_signals', 'signals_to_trades',
    
    # 桥接
    'BacktestBridge', 'run_backtest_for_agent', 'quick_backtest',
]