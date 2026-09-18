# Agent Framework

一个完整的 LLM Agent 工程实践项目:通用 Agent 框架 + 量化交易多 Agent 系统 + 双套评估体系。

---

## 架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          agent_framework                            │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  core/                                                       │   │
│  │    CoreAgent (ReAct 循环)                                     │   │
│  │      ├─ 工具调用                                              │   │
│  │      ├─ 记忆系统 (工作记忆 + 长期记忆)                         │   │
│  │      ├─ 反思机制 (失败驱动自我改进)                            │   │
│  │      ├─ 上下文压缩                                            │   │
│  │      ├─ 熔断 + 重试                                           │   │
│  │      └─ Tracer (OpenTelemetry)                                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  tools/                       eval/                          │   │
│  │    工具注册表                   通用评估框架                    │   │
│  │    calculate                  ├─ scorer 库                    │   │
│  │    get_current_time           ├─ 测试任务集                    │   │
│  │    format_json                ├─ 报告生成                      │   │
│  │    ...                        └─ 版本对比                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                          my_agents (量化系统)                        │
│                                                                     │
│    MarketAgent ──► AlphaAgent ──► RiskAgent ──► ExecutionAgent      │
│    (市场状态)       (信号生成)      (风控闸门)     (订单参数)          │
│         │                │              │              │            │
│         └────────────────┴──────────────┴──────────────┘            │
│                          │                                          │
│                          ▼                                          │
│                 MultiAgentOrchestrator                              │
│                          │                                          │
│                          ▼                                          │
│                    Decision (action / confidence /                   │
│                              SL / TP / position_size)               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     backtest_multi_agent (回测)                      │
│                                                                     │
│   行情数据 ──► 逐 bar 决策 ──► 模拟撮合 ──► BacktestResult           │
│                                              (收益/夏普/回撤/胜率)    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    eval/ (量化回测评估)                               │
│                                                                     │
│   run_batch.py   ──► 多标的 × 多参数 ──► backtest_runs/*.json       │
│   backtest_compare.py  ──► 版本对比 (哪个指标变好/变差)              │
│   backtest_regression.py ──► 净值差分 (哪一段行情变了)               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 项目结构

```
AI agent/
├── README.md
│
├── agent_framework/                    # 通用 Agent 框架
│   ├── core/
│   │   ├── core_agent.py              # ReAct Agent 主逻辑
│   │   ├── context_manager.py         # 上下文压缩
│   │   ├── retry.py                   # 重试策略
│   │   ├── circuit_breaker.py         # 熔断器
│   │   ├── trace.py                   # OpenTelemetry 追踪
│   │   ├── archival_memory.py         # 长期记忆
│   │   ├── reflector.py               # 反思机制
│   │   └── llm_adapter.py             # LLM 适配器
│   ├── tools/
│   │   └── tools_registry.py          # 工具注册表
│   └── eval/                          # ★ 通用 Agent 评估
│       ├── evaluator.py               # 评估器
│       ├── scorers.py                 # 评分函数库
│       ├── suites.py                  # 测试任务集
│       ├── run_eval.py                # 入口
│       ├── compare.py                 # 报告对比
│       └── regression.py              # 指标差分
│
├── my_agents/                          # 量化交易多 Agent
│   ├── multi_agent_orchestrator.py    # 四层编排
│   ├── market_agent.py                # 市场状态
│   ├── alpha_agent.py                 # 信号生成
│   ├── risk_agent.py                  # 风控
│   ├── execution_agent.py             # 订单参数
│   └── reflexion_engine.py            # 反思引擎
│
├── backtest_multi_agent.py            # 量化回测脚本
│
└── eval/                              # ★ 量化回测评估
    ├── run_batch.py                   # 批量回测
    ├── backtest_compare.py            # 版本对比
    └── backtest_regression.py         # 净值差分
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
echo 'export DEEPSEEK_API_KEY="sk-你的key"' >> ~/.zshrc
source ~/.zshrc
```

### 3. 跑通用 Agent 评估

```bash
python3 -m agent_framework.eval.run_eval
```

输出示例:

```
═ Eval Report: all ═
  Total:      7
  Passed:     7 (100.0%)
  Avg Score:  1.000
  Avg Steps:  2.0
  Tokens:     5101
```

### 4. 跑量化回测评估

```bash
# baseline
python3 eval/run_batch.py \
  --tag baseline \
  --symbols 300750,600519,000001 \
  --notes "初始版本"

# 改参数后再跑
python3 eval/run_batch.py \
  --tag v2 \
  --symbols 300750,600519,000001 \
  --notes "alpha min_confidence 0.45→0.55"

# 对比
python3 eval/backtest_compare.py \
  backtest_runs/baseline_*.json \
  backtest_runs/v2_*.json
```

输出示例:

```
════════════════════════════════════════════════════════════
  Compare: baseline → v2
════════════════════════════════════════════════════════════
  baseline:  初始版本
  candidate: alpha min_confidence 0.45→0.55

  avg_return     Δ: +2.30%   ✅
  avg_sharpe     Δ: +0.35    ✅
  avg_win_rate   Δ: +3.20%   ✅
  avg_trades     Δ: -12.5    ✅
  worst_drawdown Δ: +0.80%   ⚠️
  std_return     Δ: -0.50%   ✅
════════════════════════════════════════════════════════════
```

---

## 两套评估的区别

| | 通用 Agent 评估 | 量化回测评估 |
|---|---|---|
| 位置 | `agent_framework/eval/` | `eval/` |
| 测的对象 | `CoreAgent`(LLM 驱动) | `MultiAgentOrchestrator`(算法驱动) |
| 输入 | 7 个任务 (算术/工具/多步) | 股票代码 + agent 参数 |
| 输出 | 通过率、平均分 | 收益、夏普、回撤、胜率 |
| 目的 | 验证 LLM 能否正确调用工具 | 验证量化策略表现 |
| 是否依赖 API | 是 (DeepSeek) | 否 (纯本地计算) |

**为什么分两套:** 两者测试对象、依赖、演化方向完全不同。物理隔离比合并更清晰。

---

## 核心技术点

### CoreAgent

- **ReAct 循环**:Thought → Action → Observation → 循环
- **记忆系统**:工作记忆(session) + 长期记忆(archival,向量检索)
- **反思机制**:失败时自动反思,产出"下一步怎么改"
- **上下文压缩**:超过 token 阈值时自动摘要,防止 context 爆炸
- **熔断器**:连续失败 N 次后快速失败,避免无效重试
- **重试策略**:LLM 调用失败时指数退避重试
- **OpenTelemetry**:完整 trace,可导出到 Jaeger / OTLP

### 量化四层 Agent

- **MarketAgent**:识别 regime(趋势/震荡/波动),提供 support/resistance
- **AlphaAgent**:多信号融合(趋势/MACD/布林带/盘口),维度投票产出方向
- **RiskAgent**:风控闸门(clear/reduce/block/force_close),日内止损,追踪止损
- **ExecutionAgent**:Kelly 公式 + 波动率目标仓位,ATR 动态止损止盈,滑点估算
- **在线学习**:根据实际收益调整 AlphaAgent 权重

### 回测引擎

- 事件驱动,逐 bar 模拟
- 支持做多/做空、止损/止盈、滑点、手续费
- 输出:总收益、年化、夏普、索提诺、卡玛、回撤、胜率、盈亏比

---

## 项目亮点

1. **双套评估体系** — 通用 Agent 评估 + 量化回测评估,职责清晰,互不干扰
2. **完整可观测性** — OpenTelemetry 追踪每个 LLM 调用、工具调用、决策
3. **生产级鲁棒性** — 重试、熔断、上下文压缩、错误恢复
4. **量化方法论** — 防未来函数、多信号融合、风控闸门、在线学习
5. **可回归的评估** — 支持版本对比,改动能量化评估效果

---

## 依赖

```
python >= 3.10
openai            # LLM 调用
httpx             # HTTP 客户端
pandas            # 数据处理
numpy             # 数值计算
opentelemetry-sdk # 追踪
python-dotenv     # 环境变量
akshare           # A 股数据 (量化回测)
baostock          # A 股数据 (备用)
pyarrow           # parquet 缓存
```

