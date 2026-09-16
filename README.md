# AI Agent Framework

一个支持多轮工具调用的轻量 Agent 框架。

## 特性
- ReAct 循环（Thought → Action → Observation）
- 多 LLM Provider 支持（OpenAI / DeepSeek / Anthropic）
- 工具调用（function calling）
- 反思机制（不确定性驱动）
- OpenTelemetry 链路追踪

## 快速开始
```bash
pip install -r requirements.txt
cp .env.example .env   # 填入你的 API key
python3 test_trace.py