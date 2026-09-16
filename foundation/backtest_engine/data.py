"""
数据加载模块 - 完整版

支持:
1. Yahoo Finance 数据
2. 合成数据 (无网络时)
3. 数据重采样 (支持日内)
4. 数据缓存
5. 多数据源支持
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Union
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)


# ==================== 配置 ====================

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), '.cache')
os.makedirs(DATA_CACHE_DIR, exist_ok=True)


# ==================== 随机数据生成 ====================

def random_ohlc_data(
    n_bars: int = 500,
    start_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
    start_date: str = '2020-01-01',
    seed: Optional[int] = None,
    freq: str = 'D'  # D=日, 30T=30分钟, 5T=5分钟
) -> pd.DataFrame:
    """
    生成随机OHLCV数据 (支持日内)

    Args:
        n_bars: 数据条数
        start_price: 起始价格
        volatility: 波动率
        trend: 趋势漂移
        start_date: 起始日期
        seed: 随机种子
        freq: 频率 (D=日, 30T=30分钟, 5T=5分钟)

    Returns:
        DataFrame with Open, High, Low, Close, Volume
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=n_bars, freq=freq)

    returns = rng.normal(trend, volatility, n_bars)
    close = start_price * np.exp(np.cumsum(returns))

    # 生成OHLC
    intraday_vol = volatility * 0.5
    open_prices = close * (1 + rng.normal(0, intraday_vol * 0.3, n_bars))
    high_noise = np.abs(rng.normal(0, intraday_vol, n_bars))
    low_noise = np.abs(rng.normal(0, intraday_vol, n_bars))

    high = np.maximum(open_prices, close) * (1 + high_noise)
    low = np.minimum(open_prices, close) * (1 - low_noise)
    volume = rng.integers(500_000, 10_000_000, n_bars)

    data = pd.DataFrame({
        'Open': open_prices,
        'High': high,
        'Low': low,
        'Close': close,
        'Volume': volume,
    }, index=dates)

    # 确保一致性
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)

    return data


def generate_intraday_data(
    symbol: str,
    days: int = 5,
    interval: str = '1m'
) -> pd.DataFrame:
    """
    生成日内合成数据（用于测试日内策略）
    
    Args:
        symbol: 股票代码（用于种子）
        days: 天数
        interval: 间隔 (1m, 5m, 15m, 30m, 1h)
    
    Returns:
        日内OHLCV数据
    """
    # 日内交易分钟数
    minutes_per_day = {
        '1m': 390,   # 6.5小时
        '5m': 78,
        '15m': 26,
        '30m': 13,
        '1h': 6
    }
    
    n_bars = days * minutes_per_day.get(interval, 390)
    
    # 使用symbol作为种子的一部分
    seed = sum(ord(c) for c in symbol) % (2**31)
    
    # 日内波动率更高
    volatility = 0.015 + np.random.default_rng(seed).normal(0, 0.005)
    
    # 日内有特定模式：开盘波动大，盘中平稳，收盘前波动
    data = random_ohlc_data(
        n_bars=n_bars,
        start_price=100.0,
        volatility=volatility,
        trend=0.00005,
        start_date=datetime.now().strftime('%Y-%m-%d'),
        seed=seed,
        freq=interval
    )
    
    return data


# ==================== Yahoo Finance 加载 ====================

def load_yfinance_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
    use_cache: bool = True
) -> Optional[pd.DataFrame]:
    """
    加载Yahoo Finance数据

    Args:
        symbol: 股票代码
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
        interval: 间隔 (1m, 5m, 15m, 30m, 1h, 1d, 1wk, 1mo)
        use_cache: 是否使用缓存

    Returns:
        OHLCV DataFrame
    """
    # 检查缓存
    cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
    cache_file = os.path.join(DATA_CACHE_DIR, f"{cache_key.replace('/', '_')}.parquet")
    
    if use_cache and os.path.exists(cache_file):
        try:
            logger.info(f"Loading cached data: {cache_key}")
            return pd.read_parquet(cache_file)
        except Exception as e:
            logger.warning(f"Cache read failed: {e}")
    
    try:
        import yfinance as yf
        
        # 设置超时
        yf.set_config({'timeout': 15})
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(
            start=start_date or '2020-01-01',
            end=end_date or None,
            interval=interval,
            auto_adjust=True
        )
        
        if data is not None and len(data) > 0:
            # 标准化列名
            data.columns = [c.capitalize() for c in data.columns]
            ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
            available = [c for c in ohlcv if c in data.columns]
            data = data[available]
            
            # 保存缓存
            if use_cache:
                try:
                    data.to_parquet(cache_file)
                except Exception as e:
                    logger.warning(f"Cache save failed: {e}")
            
            logger.info(f"Loaded {len(data)} bars for {symbol}")
            return data
            
    except ImportError:
        logger.warning("yfinance not installed. Run: pip install yfinance")
    except Exception as e:
        logger.warning(f"Failed to load {symbol}: {e}")
    
    return None


def load_yfinance_batch(
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d"
) -> Dict[str, pd.DataFrame]:
    """
    批量加载Yahoo Finance数据
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        interval: 间隔
    
    Returns:
        {symbol: DataFrame}
    """
    try:
        import yfinance as yf
        
        yf.set_config({'timeout': 30})
        
        data = yf.download(
            symbols,
            start=start_date or '2020-01-01',
            end=end_date or None,
            interval=interval,
            auto_adjust=True,
            group_by='ticker',
            progress=False
        )
        
        results = {}
        
        if len(symbols) == 1:
            # 单股票返回不同结构
            symbol = symbols[0]
            if not data.empty:
                results[symbol] = data
        else:
            # 多股票
            for symbol in symbols:
                if symbol in data.columns.levels[0]:
                    results[symbol] = data[symbol]
                else:
                    logger.warning(f"No data for {symbol}")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch load failed: {e}")
        return {}


# ==================== 统一加载接口 ====================

def load_data(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
    use_synthetic: bool = True,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    统一数据加载接口

    Args:
        symbol: 股票代码
        start_date: 开始日期
        end_date: 结束日期
        interval: 间隔
        use_synthetic: 是否使用合成数据作为备选
        use_cache: 是否使用缓存

    Returns:
        OHLCV DataFrame
    """
    # 1. 尝试从Yahoo加载
    data = load_yfinance_data(symbol, start_date, end_date, interval, use_cache)

    if data is not None and len(data) > 0:
        return data

    # 2. 尝试加载内置测试数据
    try:
        from backtesting.test import GOOG
        if symbol.upper() in ('GOOG', 'GOOGL'):
            data = GOOG.copy()
            logger.info(f"Loaded built-in GOOG data: {len(data)} bars")
            return data
    except ImportError:
        pass

    # 3. 使用合成数据
    if use_synthetic:
        logger.info(f"Using synthetic data for {symbol}")
        
        # 检查是否是日内数据
        is_intraday = interval in ['1m', '5m', '15m', '30m', '1h']
        
        seed = sum(ord(c) for c in symbol) % (2**31)
        
        if is_intraday:
            # 日内数据：生成5天
            return generate_intraday_data(symbol, days=5, interval=interval)
        else:
            # 日线数据：生成500天
            return random_ohlc_data(
                n_bars=500,
                start_date=start_date or '2022-01-01',
                seed=seed,
                freq='D'
            )

    raise ValueError(f"No data available for {symbol}")


def load_data_batch(
    symbols: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
    use_synthetic: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    批量加载数据
    
    Args:
        symbols: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        interval: 间隔
        use_synthetic: 是否使用合成数据
    
    Returns:
        {symbol: DataFrame}
    """
    # 尝试从Yahoo批量加载
    results = load_yfinance_batch(symbols, start_date, end_date, interval)
    
    # 补全缺失的符号
    for symbol in symbols:
        if symbol not in results:
            try:
                data = load_data(symbol, start_date, end_date, interval, use_synthetic)
                if not data.empty:
                    results[symbol] = data
            except Exception as e:
                logger.warning(f"Failed to load {symbol}: {e}")
    
    return results


# ==================== 数据预处理 ====================

def resample_ohlcv(data: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    重采样OHLCV数据

    Args:
        data: OHLCV DataFrame
        rule: 重采样规则 ('W', 'M', '4H', '30T', etc.)

    Returns:
        重采样后的DataFrame
    """
    agg = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
    }
    if 'Volume' in data.columns:
        agg['Volume'] = 'sum'

    return data.resample(rule).agg(agg).dropna()


def fill_missing_data(data: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
    """
    填充缺失数据
    
    Args:
        data: OHLCV DataFrame
        method: 填充方法 ('ffill', 'bfill', 'interpolate')
    
    Returns:
        填充后的DataFrame
    """
    df = data.copy()
    
    if method == 'ffill':
        df = df.fillna(method='ffill')
    elif method == 'bfill':
        df = df.fillna(method='bfill')
    elif method == 'interpolate':
        df = df.interpolate()
    
    return df


def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    归一化数据（用于回测）
    
    Args:
        data: OHLCV DataFrame
    
    Returns:
        归一化后的DataFrame
    """
    df = data.copy()
    
    for col in ['Open', 'High', 'Low', 'Close']:
        if col in df.columns:
            df[col] = df[col] / df[col].iloc[0]
    
    return df


def split_data(
    data: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15
) -> Dict[str, pd.DataFrame]:
    """
    分割数据为训练集、验证集、测试集
    
    Args:
        data: OHLCV DataFrame
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    
    Returns:
        {'train': df, 'val': df, 'test': df}
    """
    n = len(data)
    
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    return {
        'train': data.iloc[:train_end],
        'val': data.iloc[train_end:val_end],
        'test': data.iloc[val_end:]
    }


# ==================== 数据信息 ====================

def get_data_info(data: pd.DataFrame) -> Dict[str, Any]:
    """
    获取数据信息
    
    Args:
        data: OHLCV DataFrame
    
    Returns:
        数据信息字典
    """
    return {
        'rows': len(data),
        'columns': data.columns.tolist(),
        'start_date': str(data.index[0]),
        'end_date': str(data.index[-1]),
        'missing_values': data.isnull().sum().to_dict(),
        'date_range_days': (data.index[-1] - data.index[0]).days,
        'is_intraday': (data.index[1] - data.index[0]).seconds < 86400 if len(data) > 1 else False,
        'avg_volume': float(data['Volume'].mean()) if 'Volume' in data.columns else None,
        'price_range': {
            'min': float(data['Low'].min()) if 'Low' in data.columns else None,
            'max': float(data['High'].max()) if 'High' in data.columns else None
        }
    }


def quick_data_preview(symbol: str, days: int = 5) -> pd.DataFrame:
    """
    快速预览数据（最近N天）
    
    Args:
        symbol: 股票代码
        days: 天数
    
    Returns:
        最近N天的数据
    """
    data = load_data(symbol, interval='1d')
    if not data.empty:
        return data.tail(days)
    return pd.DataFrame()


# ==================== 缓存管理 ====================

def clear_cache():
    """清除所有缓存数据"""
    import shutil
    if os.path.exists(DATA_CACHE_DIR):
        shutil.rmtree(DATA_CACHE_DIR)
        os.makedirs(DATA_CACHE_DIR, exist_ok=True)
        logger.info("Cache cleared")


def get_cache_size() -> int:
    """获取缓存大小（字节）"""
    total = 0
    for root, dirs, files in os.walk(DATA_CACHE_DIR):
        for f in files:
            fp = os.path.join(root, f)
            total += os.path.getsize(fp)
    return total


# ==================== 便捷函数 ====================

def get_daily_data(symbol: str, years: int = 2) -> pd.DataFrame:
    """获取日线数据"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    return load_data(
        symbol,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        interval='1d'
    )


def get_intraday_data(symbol: str, days: int = 5, interval: str = '5m') -> pd.DataFrame:
    """获取日内数据"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    return load_data(
        symbol,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        interval=interval
    )