"""
性能评估器
评估交易策略的表现，计算各种量化指标
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """单笔交易结果"""
    symbol: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    quantity: int
    direction: str  # 'long' or 'short'
    profit: float
    profit_pct: float
    holding_period: float  # 持有时间（小时）
    max_favorable: float  # 最大浮盈
    max_adverse: float  # 最大浮亏
    exit_reason: str  # 'take_profit', 'stop_loss', 'time_exit', 'manual'
    commission: float
    slippage: float


@dataclass
class BacktestMetrics:
    """回测指标"""
    # 收益指标
    total_return: float
    annualized_return: float
    cumulative_returns: pd.Series
    
    # 风险指标
    max_drawdown: float
    max_drawdown_pct: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    
    # 交易统计
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy: float
    
    # 其他
    avg_holding_period: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    best_trade: float
    worst_trade: float
    recovery_factor: float
    
    # 月度/年度统计
    monthly_returns: pd.Series
    yearly_returns: pd.Series


class PerformanceEvaluator:
    """性能评估器"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.trade_history = []
        self.performance_cache = {}
        self.name = "PerformanceEvaluator"
        
    def add_trade(self, trade: TradeResult):
        """添加交易记录"""
        self.trade_history.append(trade)
        self.performance_cache = {}  # 清空缓存
        
    def add_trades(self, trades: List[TradeResult]):
        """批量添加交易"""
        self.trade_history.extend(trades)
        self.performance_cache = {}
        
    def evaluate(self, initial_capital: float = 100000) -> BacktestMetrics:
        """
        评估交易表现
        
        Args:
            initial_capital: 初始资金
            
        Returns:
            BacktestMetrics: 回测指标
        """
        if not self.trade_history:
            return self._empty_metrics()
        
        # 计算收益曲线
        equity_curve = self._calculate_equity_curve(initial_capital)
        
        # 计算各项指标
        returns = equity_curve.pct_change().dropna()
        
        # 收益指标
        total_return = (equity_curve.iloc[-1] - initial_capital) / initial_capital
        annualized_return = self._calculate_annualized_return(equity_curve)
        
        # 风险指标
        max_drawdown, max_drawdown_pct = self._calculate_max_drawdown(equity_curve)
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = annualized_return / abs(max_drawdown_pct) if max_drawdown_pct != 0 else 0
        
        # 交易统计
        trades = self.trade_history
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.profit > 0)
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        profits = [t.profit for t in trades]
        winning_profits = [p for p in profits if p > 0]
        losing_profits = [p for p in profits if p <= 0]
        
        avg_win = np.mean(winning_profits) if winning_profits else 0
        avg_loss = np.mean(losing_profits) if losing_profits else 0
        
        profit_factor = abs(sum(winning_profits) / sum(losing_profits)) if sum(losing_profits) != 0 else float('inf')
        expectancy = (win_rate * avg_win - (1 - win_rate) * abs(avg_loss)) if avg_loss != 0 else 0
        
        # 持仓统计
        holding_periods = [t.holding_period for t in trades]
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        # 连续盈亏
        max_consecutive_wins, max_consecutive_losses = self._calculate_consecutive(trades)
        
        # 最佳/最差交易
        best_trade = max(profits) if profits else 0
        worst_trade = min(profits) if profits else 0
        
        # 恢复因子
        recovery_factor = total_return / abs(max_drawdown_pct) if max_drawdown_pct != 0 else 0
        
        # 月度/年度收益
        monthly_returns = self._calculate_periodic_returns(equity_curve, 'M')
        yearly_returns = self._calculate_periodic_returns(equity_curve, 'Y')
        
        return BacktestMetrics(
            total_return=total_return,
            annualized_return=annualized_return,
            cumulative_returns=equity_curve,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            expectancy=expectancy,
            avg_holding_period=avg_holding_period,
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            best_trade=best_trade,
            worst_trade=worst_trade,
            recovery_factor=recovery_factor,
            monthly_returns=monthly_returns,
            yearly_returns=yearly_returns
        )
    
    def _calculate_equity_curve(self, initial_capital: float) -> pd.Series:
        """计算权益曲线"""
        # 按时间排序交易
        sorted_trades = sorted(self.trade_history, key=lambda x: x.exit_time)
        
        equity = initial_capital
        equity_curve = []
        timestamps = []
        
        # 起始点
        start_time = sorted_trades[0].entry_time if sorted_trades else datetime.now()
        equity_curve.append(equity)
        timestamps.append(start_time)
        
        for trade in sorted_trades:
            # 入场时扣除资金（如果有保证金要求，这里简化）
            # 出场时更新权益
            equity += trade.profit
            equity_curve.append(equity)
            timestamps.append(trade.exit_time)
        
        # 创建Series
        index = pd.DatetimeIndex(timestamps)
        return pd.Series(equity_curve, index=index)
    
    def _calculate_annualized_return(self, equity_curve: pd.Series) -> float:
        """计算年化收益率"""
        if len(equity_curve) < 2:
            return 0
        
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0]
        
        # 计算年数
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = max(days / 365.25, 0.001)  # 防止除以0
        
        annualized = (1 + total_return) ** (1 / years) - 1
        return annualized
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> Tuple[float, float]:
        """计算最大回撤"""
        if len(equity_curve) < 2:
            return 0, 0
        
        # 计算累计最高点
        cummax = equity_curve.expanding().max()
        drawdown = cummax - equity_curve
        drawdown_pct = drawdown / cummax
        
        max_drawdown = drawdown.max()
        max_drawdown_pct = drawdown_pct.max()
        
        return max_drawdown, max_drawdown_pct
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """计算夏普比率"""
        if len(returns) < 2 or returns.std() == 0:
            return 0
        
        excess_returns = returns.mean() - self.risk_free_rate / 252
        sharpe = excess_returns / returns.std() * np.sqrt(252)
        return sharpe
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """计算索提诺比率"""
        if len(returns) < 2:
            return 0
        
        # 只计算下行波动率
        downside_returns = returns[returns < 0]
        if len(downside_returns) < 2 or downside_returns.std() == 0:
            return 0
        
        excess_returns = returns.mean() - self.risk_free_rate / 252
        sortino = excess_returns / downside_returns.std() * np.sqrt(252)
        return sortino
    
    def _calculate_consecutive(self, trades: List[TradeResult]) -> Tuple[int, int]:
        """计算连续盈利/亏损次数"""
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            if trade.profit > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        return max_wins, max_losses
    
    def _calculate_periodic_returns(self, equity_curve: pd.Series, freq: str) -> pd.Series:
        """计算周期收益率"""
        if len(equity_curve) < 2:
            return pd.Series()
        
        # 重采样到周期频率
        resampled = equity_curve.resample(freq).last()
        returns = resampled.pct_change().dropna()
        return returns
    
    def _empty_metrics(self) -> BacktestMetrics:
        """返回空指标"""
        empty_series = pd.Series()
        return BacktestMetrics(
            total_return=0,
            annualized_return=0,
            cumulative_returns=empty_series,
            max_drawdown=0,
            max_drawdown_pct=0,
            volatility=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            expectancy=0,
            avg_holding_period=0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            best_trade=0,
            worst_trade=0,
            recovery_factor=0,
            monthly_returns=empty_series,
            yearly_returns=empty_series
        )
    
    def generate_report(self, metrics: BacktestMetrics) -> str:
        """生成可读性报告"""
        lines = []
        lines.append("=" * 60)
        lines.append("📊 交易性能评估报告")
        lines.append("=" * 60)
        
        lines.append("\n📈 收益指标:")
        lines.append(f"  总收益率: {metrics.total_return:.2%}")
        lines.append(f"  年化收益率: {metrics.annualized_return:.2%}")
        lines.append(f"  夏普比率: {metrics.sharpe_ratio:.3f}")
        lines.append(f"  索提诺比率: {metrics.sortino_ratio:.3f}")
        lines.append(f"  卡尔玛比率: {metrics.calmar_ratio:.3f}")
        
        lines.append("\n📉 风险指标:")
        lines.append(f"  最大回撤: ${metrics.max_drawdown:,.2f}")
        lines.append(f"  最大回撤率: {metrics.max_drawdown_pct:.2%}")
        lines.append(f"  年化波动率: {metrics.volatility:.2%}")
        
        lines.append("\n📊 交易统计:")
        lines.append(f"  总交易次数: {metrics.total_trades}")
        lines.append(f"  盈利次数: {metrics.winning_trades}")
        lines.append(f"  亏损次数: {metrics.losing_trades}")
        lines.append(f"  胜率: {metrics.win_rate:.2%}")
        lines.append(f"  平均盈利: ${metrics.avg_win:,.2f}")
        lines.append(f"  平均亏损: ${metrics.avg_loss:,.2f}")
        lines.append(f"  盈亏比: {metrics.profit_factor:.3f}")
        lines.append(f"  期望值: ${metrics.expectancy:,.2f}")
        lines.append(f"  平均持仓时间: {metrics.avg_holding_period:.1f}小时")
        
        lines.append("\n🏆 其他:")
        lines.append(f"  最佳交易: ${metrics.best_trade:,.2f}")
        lines.append(f"  最差交易: ${metrics.worst_trade:,.2f}")
        lines.append(f"  最大连续盈利: {metrics.max_consecutive_wins}")
        lines.append(f"  最大连续亏损: {metrics.max_consecutive_losses}")
        lines.append(f"  恢复因子: {metrics.recovery_factor:.3f}")
        
        lines.append("\n" + "=" * 60)
        
        return "\n".join(lines)