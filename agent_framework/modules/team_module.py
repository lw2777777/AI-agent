"""
Team Module - 多Agent团队协作

支持的模式:
1. coordinate: 协调模式 (Leader分配任务)
2. route: 路由模式 (自动路由到最合适的Agent)
3. collaborate: 协作模式 (所有Agent协作)
4. debate: 辩论模式 (多Agent讨论后达成共识)

每个Agent可以有不同的角色:
- researcher: 研究分析
- analyst: 数据分析
- trader: 交易执行
- risk_manager: 风险管理
- reporter: 报告生成
"""

from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class TeamMode(Enum):
    """团队模式"""
    COORDINATE = "coordinate"    # Leader协调
    ROUTE = "route"              # 自动路由
    COLLABORATE = "collaborate"  # 全员协作
    DEBATE = "debate"            # 辩论模式


class TeamRole(Enum):
    """Agent角色"""
    LEADER = "leader"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    TRADER = "trader"
    RISK_MANAGER = "risk_manager"
    REPORTER = "reporter"
    OBSERVER = "observer"


@dataclass
class TeamMember:
    """团队成员"""
    agent: Any
    role: TeamRole
    name: str
    description: str = ""
    response: Optional[str] = None
    confidence: float = 0.0


@dataclass
class TeamResult:
    """团队执行结果"""
    success: bool
    responses: List[Dict[str, Any]]
    aggregated: str
    mode: TeamMode
    duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TeamModule:
    """
    多Agent团队模块
    
    管理多个Agent的协作执行
    
    使用示例:
        team = TeamModule("Research Team", mode="coordinate")
        team.add_member(researcher_agent, TeamRole.RESEARCHER)
        team.add_member(analyst_agent, TeamRole.ANALYST)
        team.add_member(reporter_agent, TeamRole.REPORTER)
        
        result = team.run("分析AAPL股票的投资价值")
    """
    
    def __init__(
        self,
        name: str = "Agent Team",
        mode: Union[str, TeamMode] = TeamMode.COORDINATE,
        description: Optional[str] = None,
        leader: Optional[Any] = None,
        llm: Optional[Any] = None
    ):
        if isinstance(mode, str):
            mode = TeamMode(mode)
        
        self.name = name
        self.mode = mode
        self.description = description
        self.leader = leader
        self.llm = llm
        
        self.members: List[TeamMember] = []
        self.history: List[TeamResult] = []
        self._event_handlers: Dict[str, List[Callable]] = {}
    
    def add_member(
        self,
        agent: Any,
        role: Union[str, TeamRole],
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> 'TeamModule':
        """添加成员"""
        if isinstance(role, str):
            role = TeamRole(role)
        
        member = TeamMember(
            agent=agent,
            role=role,
            name=name or agent.__class__.__name__,
            description=description or ""
        )
        self.members.append(member)
        logger.info(f"Added team member: {member.name} ({role.value})")
        return self
    
    def on(self, event_type: str, handler: Callable) -> 'TeamModule':
        """注册事件处理器"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        return self
    
    def _trigger_event(self, event_type: str, data: Any) -> None:
        """触发事件"""
        for handler in self._event_handlers.get(event_type, []):
            try:
                handler(data)
            except Exception as e:
                logger.warning(f"Event handler error: {e}")
    
    async def _run_member(self, member: TeamMember, query: str) -> Dict[str, Any]:
        """运行单个成员"""
        try:
            start_time = datetime.now()
            
            if hasattr(member.agent, 'run'):
                result = member.agent.run(query)
            elif hasattr(member.agent, '__call__'):
                result = member.agent(query)
            else:
                result = str(member.agent)
            
            duration = (datetime.now() - start_time).total_seconds()
            
            member.response = result.get('answer', str(result)) if isinstance(result, dict) else str(result)
            member.confidence = result.get('confidence', 0.0) if isinstance(result, dict) else 0.0
            
            return {
                'name': member.name,
                'role': member.role.value,
                'response': member.response,
                'confidence': member.confidence,
                'duration': duration,
                'success': True
            }
        except Exception as e:
            logger.error(f"Member {member.name} failed: {e}")
            return {
                'name': member.name,
                'role': member.role.value,
                'response': None,
                'error': str(e),
                'success': False
            }
    
    async def _run_coordinate(self, query: str) -> TeamResult:
        """协调模式: Leader分配任务"""
        start_time = datetime.now()
        responses = []
        
        # 1. Leader分析任务
        if self.leader:
            analysis = self.leader.run(f"分析这个任务并规划子任务: {query}")
            plan = analysis.get('answer', '')
        else:
            plan = query
        
        # 2. 并行执行所有成员 (除了Leader)
        tasks = []
        for member in self.members:
            if member.role != TeamRole.LEADER:
                member_query = f"{plan}\n\n你的角色: {member.role.value}\n任务: {query}"
                tasks.append(self._run_member(member, member_query))
        
        results = await asyncio.gather(*tasks)
        responses.extend(results)
        
        # 3. Leader汇总
        if self.leader:
            summary_input = "\n".join([
                f"{r['name']}: {r.get('response', 'No response')}"
                for r in results if r.get('success')
            ])
            summary = self.leader.run(f"汇总以下分析结果: {summary_input}")
            aggregated = summary.get('answer', '')
        else:
            aggregated = "\n\n".join([
                f"## {r['name']}\n{r.get('response', '')}"
                for r in results if r.get('success')
            ])
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return TeamResult(
            success=True,
            responses=responses,
            aggregated=aggregated,
            mode=TeamMode.COORDINATE,
            duration=duration
        )
    
    async def _run_collaborate(self, query: str) -> TeamResult:
        """协作模式: 所有成员共同讨论"""
        start_time = datetime.now()
        responses = []
        
        # 多轮对话
        context = f"团队任务: {query}\n\n"
        
        for round_num in range(2):  # 2轮讨论
            tasks = []
            for member in self.members:
                member_query = f"{context}\n\n轮次 {round_num + 1}\n你的角色: {member.role.value}\n请提供你的分析和见解。"
                tasks.append(self._run_member(member, member_query))
            
            results = await asyncio.gather(*tasks)
            
            # 更新上下文
            for r in results:
                if r.get('success') and r.get('response'):
                    context += f"{r['name']}: {r['response']}\n\n"
                    responses.append(r)
        
        # 用LLM汇总 (如果有)
        if self.llm:
            aggregated = self.llm.chat(f"汇总以下讨论: {context}")
        else:
            aggregated = context
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return TeamResult(
            success=True,
            responses=responses,
            aggregated=aggregated,
            mode=TeamMode.COLLABORATE,
            duration=duration
        )
    
    async def _run_debate(self, query: str) -> TeamResult:
        """辩论模式: 不同观点辩论"""
        start_time = datetime.now()
        responses = []
        
        # 正反方分组
        pros = [m for m in self.members if m.role in [TeamRole.ANALYST, TeamRole.TRADER]]
        cons = [m for m in self.members if m.role in [TeamRole.RISK_MANAGER, TeamRole.OBSERVER]]
        
        if not pros or not cons:
            pros = self.members[:len(self.members)//2]
            cons = self.members[len(self.members)//2:]
        
        context = f"辩论议题: {query}\n\n"
        
        # 正方陈述
        for member in pros:
            result = await self._run_member(member, f"{context}\n请支持以下观点: {query}\n提供论据和证据。")
            responses.append(result)
            if result.get('response'):
                context += f"{member.name} (支持): {result['response']}\n\n"
        
        # 反方陈述
        for member in cons:
            result = await self._run_member(member, f"{context}\n请反驳以下观点: {query}\n提供反对论据和证据。")
            responses.append(result)
            if result.get('response'):
                context += f"{member.name} (反对): {result['response']}\n\n"
        
        # 主持人总结
        if self.leader:
            summary = self.leader.run(f"总结以下辩论: {context}")
            aggregated = summary.get('answer', '')
        else:
            aggregated = context
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return TeamResult(
            success=True,
            responses=responses,
            aggregated=aggregated,
            mode=TeamMode.DEBATE,
            duration=duration
        )
    
    async def _run_route(self, query: str) -> TeamResult:
        """路由模式: 自动选择最合适的Agent"""
        start_time = datetime.now()
        responses = []
        
        # 1. 分析查询，选择最合适的成员
        if self.llm:
            selection_prompt = f"""
            分析以下查询，选择最合适的Agent角色:
            查询: {query}
            
            可用角色:
            {', '.join([m.role.value for m in self.members])}
            
            只返回角色名称。
            """
            selected_role = self.llm.chat(selection_prompt).strip().lower()
        else:
            # 简单关键词匹配
            keywords = {
                'research': ['research', 'study', 'investigate'],
                'analyst': ['analyze', 'analysis', 'evaluate'],
                'trader': ['trade', 'buy', 'sell', 'price'],
                'risk_manager': ['risk', 'volatility', 'drawdown'],
                'reporter': ['report', 'summary', 'document']
            }
            for role, words in keywords.items():
                if any(w in query.lower() for w in words):
                    selected_role = role
                    break
            else:
                selected_role = 'analyst'
        
        # 2. 找到对应成员
        selected_member = None
        for member in self.members:
            if member.role.value == selected_role:
                selected_member = member
                break
        
        if not selected_member:
            selected_member = self.members[0]
        
        # 3. 执行
        result = await self._run_member(selected_member, query)
        responses.append(result)
        
        duration = (datetime.now() - start_time).total_seconds()
        
        return TeamResult(
            success=result.get('success', False),
            responses=responses,
            aggregated=result.get('response', ''),
            mode=TeamMode.ROUTE,
            duration=duration
        )
    
    def run(self, query: str) -> TeamResult:
        """运行团队"""
        self._trigger_event('team_start', {'query': query, 'mode': self.mode.value})
        
        # 选择执行模式
        mode_handlers = {
            TeamMode.COORDINATE: self._run_coordinate,
            TeamMode.ROUTE: self._run_route,
            TeamMode.COLLABORATE: self._run_collaborate,
            TeamMode.DEBATE: self._run_debate,
        }
        
        handler = mode_handlers.get(self.mode, self._run_coordinate)
        result = asyncio.run(handler(query))
        
        self.history.append(result)
        self._trigger_event('team_complete', {'result': result})
        
        return result
    
    def get_history(self) -> List[TeamResult]:
        """获取执行历史"""
        return self.history
    
    def get_summary(self) -> Dict[str, Any]:
        """获取团队摘要"""
        return {
            'name': self.name,
            'mode': self.mode.value,
            'members': [
                {
                    'name': m.name,
                    'role': m.role.value,
                    'description': m.description
                }
                for m in self.members
            ],
            'total_runs': len(self.history),
            'last_run': self.history[-1].__dict__ if self.history else None
        }
    
    def to_agent_config(self) -> Dict[str, Any]:
        """转换为Agent配置"""
        return {
            'team_name': self.name,
            'team_mode': self.mode.value,
            'team_members': [
                {
                    'name': m.name,
                    'role': m.role.value,
                    'description': m.description
                }
                for m in self.members
            ]
        }
    
    @classmethod
    def from_config(cls, config: Dict[str, Any], agents: List[Any]) -> 'TeamModule':
        """从配置创建"""
        team = cls(
            name=config.get('name', 'Agent Team'),
            mode=config.get('mode', 'coordinate'),
            description=config.get('description')
        )
        
        for i, agent in enumerate(agents):
            roles = config.get('roles', [])
            role = roles[i] if i < len(roles) else 'analyst'
            team.add_member(agent, role)
        
        return team


class TeamBuilder:
    """
    团队构建器 (Fluent API)
    
    使用示例:
        team = (TeamBuilder("Research Team")
            .with_mode("collaborate")
            .with_leader(leader_agent)
            .add_member(researcher_agent, "researcher")
            .add_member(analyst_agent, "analyst")
            .build())
    """
    
    def __init__(self, name: str):
        self._module = TeamModule(name=name)
    
    def with_mode(self, mode: Union[str, TeamMode]) -> 'TeamBuilder':
        self._module.mode = mode if isinstance(mode, TeamMode) else TeamMode(mode)
        return self
    
    def with_leader(self, agent: Any) -> 'TeamBuilder':
        self._module.leader = agent
        return self
    
    def with_llm(self, llm: Any) -> 'TeamBuilder':
        self._module.llm = llm
        return self
    
    def add_member(self, agent: Any, role: Union[str, TeamRole],
                   name: Optional[str] = None,
                   description: Optional[str] = None) -> 'TeamBuilder':
        self._module.add_member(agent, role, name, description)
        return self
    
    def on_event(self, event_type: str, handler: Callable) -> 'TeamBuilder':
        self._module.on(event_type, handler)
        return self
    
    def build(self) -> TeamModule:
        return self._module