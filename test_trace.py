# test_trace.py
from agent_framework.core.trace import Tracer, init_tracer_provider
from agent_framework.core.core_agent import CoreAgent
from agent_framework.core.llm_adapter import LLMAdapter
from agent_framework.tools.tools_registry import get_tools_registry
import inspect
import os
from dotenv import load_dotenv
load_dotenv() #加载.env

from agent_framework.tools.tools_registry import get_tools_registry
# ... 其他 import ...


# 1. 初始化 OTel（用 console exporter，不用起 Jaeger）
init_tracer_provider(service_name="test-agent", exporter="console")


# 2. 建 tracer
tracer = Tracer(service_name="test-agent", exporter="console", verbose=True)

# 3. 建 agent
registry = get_tools_registry()
tools = registry.get_tools(["calculate"], api_keys={})

llm = LLMAdapter(provider="deepseek",
                 api_key=os.environ["DEEPSEEK_API_KEY"],
                 model="deepseek-chat",          # 或 deepseek-reasoner
                 base_url="https://api.deepseek.com")
agent = CoreAgent(name="test", llm=llm, tools=tools, tracer=tracer)

# 4. 跑
result = agent.run("计算 123 * 456")
print("Answer:", result.answer)

# 5. 导出
tracer.export_json("traces/test_run.json")
tracer.shutdown()