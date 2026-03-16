import copy
from typing import Any, AsyncGenerator, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool, tool

from agentchat.api.services.agent_skill import AgentSkillService
from agentchat.api.services.llm import LLMService
from agentchat.api.services.mcp_server import MCPService
from agentchat.api.services.tool import ToolService
from agentchat.core.agents.react_agent import ReactAgent
from agentchat.core.models.manager import ModelManager
from agentchat.services.checkpoint import MySQLCheckpointSaver
from agentchat.services.rag.handler import RagHandler
from agentchat.tools import AgentToolsWithName
from agentchat.tools.openapi_tool.adapter import OpenAPIToolAdapter


class AgentConfig(BaseModel):
    user_id: str
    llm_id: str
    mcp_ids: List[str]
    knowledge_ids: List[str]
    tool_ids: List[str]
    agent_skill_ids: List[str]
    system_prompt: str
    enable_memory: bool = False
    name: str = "agent"


class GeneralAgent:
    def __init__(self, agent_config: AgentConfig):
        self.agent_config = agent_config
        self.conversation_model = None
        self.react_agent: Optional[ReactAgent] = None

        self.tools: List[BaseTool] = []
        self.mcp_agent_as_tools: List[BaseTool] = []
        self.skill_agent_as_tools: List[BaseTool] = []
        self.tool_metadata_map: Dict[str, Dict[str, Any]] = {}
        self.checkpointer = MySQLCheckpointSaver()

        self.stop_streaming = False
        self.last_run_status = "idle"
        self.last_checkpoint_id: Optional[str] = None
        self.last_paused_tools: list[dict[str, Any]] = []
        self.last_response_content = ""
        self.last_context_package = None

    async def init_agent(self):
        self.mcp_agent_as_tools = await self.setup_mcp_agent_as_tools()
        self.tools = await self.setup_tools()
        self.skill_agent_as_tools = await self.setup_agent_skill_as_tools()
        await self.setup_knowledge_tool()
        await self.setup_language_model()
        self.react_agent = ReactAgent(
            model=self.conversation_model,
            system_prompt=self.agent_config.system_prompt,
            tools=self.tools + self.mcp_agent_as_tools + self.skill_agent_as_tools,
            tool_metadata_map=self.tool_metadata_map,
            checkpointer=self.checkpointer,
        )

    async def setup_language_model(self):
        if self.agent_config.llm_id:
            model_config = await LLMService.get_llm_by_id(self.agent_config.llm_id)
            self.conversation_model = ModelManager.get_user_model(**model_config)
        else:
            self.conversation_model = ModelManager.get_conversation_model()

    async def setup_tools(self) -> List[BaseTool]:
        def create_openapi_tool_executor(tool_adapter, tool_name):
            async def _execute_wrapper(**kwargs):
                return await tool_adapter.execute(_tool_name=tool_name, **kwargs)

            return _execute_wrapper

        tools = []
        db_tools = await ToolService.get_tools_from_id(self.agent_config.tool_ids)
        for db_tool in db_tools:
            governance = {
                "name": db_tool.display_name,
                "type": "tool",
                "risk_level": getattr(db_tool, "risk_level", "medium"),
                "approval_policy": getattr(db_tool, "approval_policy", "auto"),
                "idempotent": getattr(db_tool, "idempotent", True),
                "audit_enabled": getattr(db_tool, "audit_enabled", True),
            }
            if db_tool.is_user_defined:
                tool_adapter = OpenAPIToolAdapter(
                    auth_config=db_tool.auth_config,
                    openapi_schema=db_tool.openapi_schema,
                )
                for openapi_tool in tool_adapter.tools:
                    tool_name = openapi_tool["function"].get("name", "")
                    tools.append(
                        StructuredTool(
                            name=tool_name,
                            description=openapi_tool["function"].get("description", ""),
                            coroutine=create_openapi_tool_executor(tool_adapter, tool_name),
                            args_schema=openapi_tool,
                        )
                    )
                    self.tool_metadata_map[tool_name] = governance
            else:
                agent_tool = AgentToolsWithName.get(db_tool.name)
                if agent_tool:
                    tools.append(agent_tool)
                    self.tool_metadata_map[agent_tool.name] = governance
                else:
                    logger.warning(f"Tool `{db_tool.name}` is not registered in AgentToolsWithName")

        return tools

    async def setup_agent_skill_as_tools(self) -> List[BaseTool]:
        agent_skill_as_tools = []
        agent_skills = await AgentSkillService.get_agent_skills_by_ids(self.agent_config.agent_skill_ids)

        for agent_skill in agent_skills:
            self.tool_metadata_map[agent_skill.as_tool_name] = {
                "name": agent_skill.name,
                "type": "skill",
                "risk_level": "medium",
                "approval_policy": "auto",
                "idempotent": True,
                "audit_enabled": True,
            }

            @tool(agent_skill.as_tool_name, description=agent_skill.description)
            async def call_skill_agent(query: str, _skill=agent_skill):
                from agentchat.core.agents.skill_agent import SkillAgent

                skill_agent = SkillAgent(_skill, self.agent_config.user_id)
                await skill_agent.init_skill_agent()
                messages = await skill_agent.ainvoke([HumanMessage(content=query)])
                return "\n".join([message.content for message in messages])

            agent_skill_as_tools.append(call_skill_agent)

        return agent_skill_as_tools

    async def setup_mcp_agent_as_tools(self) -> List[BaseTool]:
        mcp_agent_as_tools = []
        for mcp_id in self.agent_config.mcp_ids:
            mcp_server = await MCPService.get_mcp_server_from_id(mcp_id)
            self.tool_metadata_map[mcp_server.get("mcp_as_tool_name")] = {
                "name": mcp_server.get("server_name"),
                "type": "mcp",
                "risk_level": mcp_server.get("risk_level", "medium"),
                "approval_policy": mcp_server.get("approval_policy", "auto"),
                "idempotent": mcp_server.get("idempotent", True),
                "audit_enabled": mcp_server.get("audit_enabled", True),
            }

            @tool(mcp_server.get("mcp_as_tool_name"), description=mcp_server.get("description"))
            async def call_mcp_agent(query: str, _mcp_server=mcp_server):
                from agentchat.core.agents.mcp_agent import MCPAgent, MCPConfig

                mcp_agent = MCPAgent(MCPConfig(**_mcp_server), self.agent_config.user_id)
                await mcp_agent.init_mcp_agent()
                messages = await mcp_agent.ainvoke([HumanMessage(content=query)])
                return "\n".join([message.content for message in messages])

            mcp_agent_as_tools.append(call_mcp_agent)
        return mcp_agent_as_tools

    async def setup_knowledge_tool(self):
        @tool(parse_docstring=True)
        async def retrival_knowledge(query: str) -> str:
            """
            Retrieve and package relevant knowledge context for the user query.

            Args:
                query: user query

            Returns:
                Structured knowledge context
            """

            context_package = await RagHandler.build_context_package(query, self.agent_config.knowledge_ids)
            return context_package.to_prompt()

        if self.agent_config.knowledge_ids:
            self.tools.append(retrival_knowledge)
            self.tool_metadata_map[retrival_knowledge.name] = {
                "name": "knowledge_context",
                "type": "tool",
                "risk_level": "low",
                "approval_policy": "auto",
                "idempotent": True,
                "audit_enabled": True,
            }

    async def _prepare_messages(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        prepared_messages = copy.deepcopy(messages)
        if not self.agent_config.knowledge_ids:
            return prepared_messages

        latest_human_message = next(
            (message for message in reversed(prepared_messages) if isinstance(message, HumanMessage)),
            None,
        )
        if latest_human_message is None:
            return prepared_messages

        context_package = await RagHandler.build_context_package(
            latest_human_message.content,
            self.agent_config.knowledge_ids,
        )
        self.last_context_package = context_package
        if not context_package.compact_context:
            return prepared_messages

        prepared_messages.insert(
            1 if prepared_messages and isinstance(prepared_messages[0], SystemMessage) else 0,
            SystemMessage(content=context_package.to_prompt()),
        )
        return prepared_messages

    async def astream(
        self,
        messages: Optional[List[BaseMessage]],
        *,
        run_id: str,
        trace_id: Optional[str],
        dialog_id: str,
        resume: bool = False,
        approved_tools: Optional[List[str]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        if self.react_agent is None:
            raise ValueError("GeneralAgent is not initialized")

        prepared_messages = None
        if not resume and messages is not None:
            prepared_messages = await self._prepare_messages(messages)

        async for event in self.react_agent.astream(
            prepared_messages,
            run_id=run_id,
            trace_id=trace_id,
            dialog_id=dialog_id,
            resume=resume,
            approved_tools=approved_tools,
        ):
            if event.get("type") == "response_chunk":
                self.last_response_content = event["data"].get("accumulated", self.last_response_content)
            yield event

        self.last_run_status = self.react_agent.last_run_status
        self.last_checkpoint_id = self.react_agent.last_checkpoint_id
        self.last_paused_tools = self.react_agent.last_paused_tools
        self.last_response_content = self.react_agent.last_response_content

    def stop_streaming_callback(self):
        self.stop_streaming = True
