# foundation/backtest_engine/engine.py
"""
回测引擎 - 核心实现
支持: 多策略回测、参数优化、Walk-Forward、绩效分析
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Type, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """单笔交易记录"""
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    fee: float = 0.0
    slippage: float = 0.0
    reason: str = ""
    buy_price: float =0.0

    
    @property
    def value(self) -> float:
        return self.price * self.quantity


@dataclass
class Position:
    """持仓"""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    realized_pnl: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.avg_price


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


class BacktestEngine:
    """
    回测引擎 - 高性能事件驱动回测
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        max_position_pct: float = 0.25
    ):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_position_pct = max_position_pct
        
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self._current_price: Dict[str, float] = {}
        self._current_time: Optional[datetime] = None
        
        self._total_fees = 0.0
        self._total_slippage = 0.0
    
    def reset(self):
        """重置回测状态"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = [self.initial_capital]
        self._current_price = {}
        self._total_fees = 0.0
        self._total_slippage = 0.0
    
    def execute_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = 'market',
        reason: str = ""
    ) -> Optional[Trade]:
        """
        执行订单
        
        Args:
            symbol: 股票代码
            side: 'BUY' 或 'SELL'
            quantity: 数量
            price: 参考价格
            order_type: 'market', 'limit', 'stop'
            reason: 交易原因
        
        Returns:
            Trade对象或None
        """
        if quantity <= 0:
            return None
        
        # 计算执行价格 (含滑点)
        if order_type == 'market':
            if side == 'BUY':
                exec_price = price * (1 + self.slippage)
            else:
                exec_price = price * (1 - self.slippage)
        else:
            exec_price = price
        
        # 计算手续费
        fee = exec_price * quantity * self.commission
        
        if side.upper() == 'BUY':
            cost = exec_price * quantity + fee
            # 检查资金
            if cost > self.capital:
                logger.warning(f"Insufficient capital: need {cost:.2f}, have {self.capital:.2f}")
                return None
            
            # 检查仓位限制
            position_value = cost
            total_value = self.get_portfolio_value()
            if total_value > 0 and position_value / total_value > self.max_position_pct:
                logger.warning(f"Position size {position_value/total_value:.1%} exceeds limit {self.max_position_pct:.1%}")
                return None
            
            self.capital -= (cost)
            self._total_fees += fee
            
            # 更新持仓
            if symbol not in self.positions:
                self.positions[symbol] = Position(symbol=symbol, quantity=0, avg_price=0)
            
            pos = self.positions[symbol]
            total_cost = pos.quantity * pos.avg_price + cost
            pos.quantity += quantity
            pos.avg_price = total_cost / pos.quantity if pos.quantity > 0 else 0

            buy_price_for_trade = exec_price

        else:  # SELL
            if symbol not in self.positions or self.positions[symbol].quantity < quantity:
                logger.warning(f"Insufficient position: need {quantity}, have {self.positions.get(symbol, Position(symbol)).quantity}")
                return None
            
            pos = self.positions[symbol]
            proceeds = exec_price * quantity - fee
            self.capital += proceeds
            
            # 计算已实现盈亏
            realized = (exec_price - pos.avg_price) * quantity-fee
            pos.realized_pnl = getattr(pos, 'realized_pnl', 0.0) + realized

            # SELL 的 buy_price 记录持仓均价（用于事后复盘）
            buy_price_for_trade = pos.avg_price

            pos.quantity -= quantity
            if pos.quantity == 0:
                pos.avg_price = 0
        
        # 记录交易
        trade = Trade(
            timestamp=self._current_time or datetime.now(),
            symbol=symbol,
            side=side.upper(),
            price=exec_price,
            quantity=quantity,
            fee=fee,
            slippage=exec_price * quantity * self.slippage,
            reason=reason,
            buy_price= buy_price_for_trade,              
        )
        self.trades.append(trade)
        self._total_fees += fee
        
        # 更新净值
        self._update_equity()
        
        return trade
    
    def _update_equity(self):
        """更新净值曲线"""
        total_value = self.get_portfolio_value()
        self.equity_curve.append(total_value)
    
    def update_prices(self, prices: Dict[str, float], timestamp: Optional[datetime] = None):
        """更新当前价格"""
        self._current_price.update(prices)
        if timestamp:
            self._current_time = timestamp
        self._update_equity()
    
    def get_portfolio_value(self) -> float:
        """获取当前组合总价值"""
        total = self.capital
        for symbol, pos in self.positions.items():
            if pos.quantity > 0 and symbol in self._current_price:
                total += pos.quantity * self._current_price[symbol]
        return total
    
    def get_position_value(self, symbol: str) -> float:
        """获取某只股票持仓价值"""
        if symbol in self.positions and symbol in self._current_price:
            return self.positions[symbol].quantity * self._current_price[symbol]
        return 0.0
    
    def get_unrealized_pnl(self, symbol: str) -> float:
        """获取未实现盈亏"""
        if symbol in self.positions and symbol in self._current_price:
            pos = self.positions[symbol]
            return (self._current_price[symbol] - pos.avg_price) * pos.quantity
        return 0.0
    
    def get_result(self) -> BacktestResult:
        """获取回测结果"""
        if len(self.equity_curve) < 2:
            return BacktestResult()
        
        equity = pd.Series(self.equity_curve)
        returns = equity.pct_change().dropna()
        
        # 总收益率
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1)
        
        # 年化收益率
        n_days = len(equity)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 0 else 0
        
        # 夏普比率
        ann_factor = np.sqrt(252)
        sharpe = (returns.mean() / returns.std() * ann_factor) if returns.std() > 0 else 0
        
        # 最大回撤
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 胜率
        total_trades = len(self.trades)
        sell_trades   = [t for t in self.trades if t.side == 'SELL']
        winning_trades = 0
        win_rate       = 0
        avg_win        = 0
        avg_loss       = 0
        profit_factor  = 0
    
        if sell_trades:
            # ── 胜率：应该基于真实盈亏，而不是 price > price*0.98 ──
           pnls = [
                (t.price - t.buy_price) * t.quantity - t.fee 
                for t in sell_trades
                if getattr(t, 'buy_price', None) is not None
              ]

           winning_trades = sum(1 for p in pnls if p > 0)
           win_rate = winning_trades / len(sell_trades)
        
           wins  = [p for p in pnls if p > 0]
           loses = [p for p in pnls if p < 0]
           avg_win  = float(np.mean(wins))  if wins  else 0
           avg_loss = float(np.mean(loses)) if loses else 0
           profit_factor = (
                abs(sum(wins) / sum(loses)) if loses and sum(loses) != 0
                else float('inf') if wins else 0
            )
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=total_trades - winning_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            trades=self.trades,
            equity_curve=self.equity_curve,
            metrics={
                'initial_capital': self.initial_capital,
                'final_capital': equity.iloc[-1],
                'total_fees': self._total_fees,
                'total_trades': total_trades,
            }
        )
    
    def run_strategy(
        self,
        data: pd.DataFrame,
        strategy_fn: Callable,
        symbol: str,
        **kwargs
    ) -> BacktestResult:
        """
        运行策略
        
        Args:
            data: OHLCV数据
            strategy_fn: 策略函数，接收 (data, engine, symbol) 并执行交易
            symbol: 交易标的
        
        Returns:
            回测结果
        """
        self.reset()
        
        # 初始化价格
        self._current_price = {symbol: data.iloc[0]['close']}
        self._current_time = data.index[0]
        
        # 逐行执行
        for idx, row in data.iterrows():
            price = row['close']
            self.update_prices({symbol: price}, idx)

            if strategy_fn is not None:
               signal = strategy_fn(data, self, symbol)

               if signal == 'BUY':
                   qty = (self.capital * 0.95) / price
                   self.execute_order(symbol, 'BUY', qty, price, reason='strategy buy')
               elif signal == 'SELL':
                   if symbol in self.positions and self.positions[symbol].quantity > 0:
                       qty = self.positions[symbol].quantity
                       self.execute_order(symbol, 'SELL', qty, price, reason='strategy sell')

        return self.get_result()
            
            # 调用策略 
        return self.get_result()
    
class StrategyRunner:
    """策略运行器 - 支持多策略、多标的"""
    
    def __init__(self, engine: Optional[BacktestEngine] = None):
        self.engine = engine or BacktestEngine()

    def run_backtest(
               self,
               strategy_class: Type,
               data: pd.DataFrame,
               **kwargs
          ) -> BacktestResult:
        """运行策略类"""
        # 策略类需要实现 next() 和 init() 方法
        strategy = strategy_class()
        strategy.data = data
        strategy.engine = self.engine
        
        # 初始化
        if hasattr(strategy, 'init'):
            strategy.init()
        
        # 执行
        for idx, row in data.iterrows():
            strategy.current_bar = row
            strategy.current_index = idx
            if hasattr(strategy, 'next'):
                strategy.next()
        
        return self.engine.get_result()
    
    def run_multiple(
              self,
              strategies: List[Dict[str, Any]],
              data_dict: Dict[str, pd.DataFrame]
          ) -> Dict[str, BacktestResult]:
              """运行多个策略"""
              results = {}
        
              for strategy_config in strategies:
                  name = strategy_config.get('name', 'strategy')
                  symbol = strategy_config.get('symbol', list(data_dict.keys())[0])

                  data = data_dict.get(symbol)
            
                  if data is None:
                       continue
            
                  self.engine.reset()
                  result = self.run_backtest(
                       strategy_config['class'],
                       data,
                       **strategy_config.get('params', {})
                   )
                  results[name] = result
        
              return results


# ============================================================================
# 便捷函数
# ============================================================================

def run_backtest(
    data: pd.DataFrame,
    strategy_code: str,
    initial_capital: float = 100000.0,
    **kwargs
) -> BacktestResult:
    """
    快速运行回测
    
    Args:
        data: OHLCV数据
        strategy_code: 策略代码字符串
        initial_capital: 初始资金
    
    Returns:
        回测结果
    """
    engine = BacktestEngine(initial_capital=initial_capital)
    
    # 构建上下文
    context = {
        'data': data,
        'capital': initial_capital,
        'position': 0,
        'trades': [],
        'equity': [initial_capital],
        'engine': engine
    }
    
    # 执行策略
    try:
        exec(strategy_code, context)
        return engine.get_result()
    except Exception as e:
        logger.error(f"Strategy execution failed: {e}")
        return BacktestResult()

