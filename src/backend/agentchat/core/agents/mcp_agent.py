from typing import List

from pydantic import BaseModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import create_react_agent

from agentchat.api.services.mcp_user_config import MCPUserConfigService
from agentchat.core.models.manager import ModelManager
from agentchat.prompts.completion import CALL_END_PROMPT
from agentchat.services.mcp.manager import MCPManager
from agentchat.utils.convert import convert_mcp_config


class MCPConfig(BaseModel):
    url: str
    type: str = "sse"
    tools: List[str] = []
    server_name: str
    mcp_server_id: str


class MCPAgent:
    def __init__(self, mcp_config: MCPConfig, user_id: str):
        self.mcp_config = mcp_config
        self.mcp_manager = MCPManager([convert_mcp_config(mcp_config.model_dump())])
        self.user_id = user_id
        self.mcp_tools: List[BaseTool] = []
        self.conversation_model = None
        self.react_agent = None

    async def init_mcp_agent(self):
        self.mcp_tools = await self.setup_mcp_tools()
        self.conversation_model = ModelManager.get_conversation_model()
        self.react_agent = create_react_agent(
            model=self.conversation_model,
            tools=self.mcp_tools,
            prompt=CALL_END_PROMPT,
        )

    async def setup_mcp_tools(self):
        raw_tools = await self.mcp_manager.get_mcp_tools()
        wrapped_tools = []

        for raw_tool in raw_tools:
            async def _call_tool(_raw_tool=raw_tool, **kwargs):
                mcp_config = await MCPUserConfigService.get_mcp_user_config(
                    self.user_id,
                    self.mcp_config.mcp_server_id,
                )
                kwargs.update(mcp_config)
                if _raw_tool.coroutine:
                    return await _raw_tool.ainvoke(kwargs)
                return _raw_tool.invoke(kwargs)

            wrapped_tools.append(
                StructuredTool(
                    name=raw_tool.name,
                    description=raw_tool.description,
                    coroutine=_call_tool,
                    args_schema=raw_tool.args_schema,
                )
            )
        return wrapped_tools

    async def ainvoke(self, messages: List[BaseMessage]) -> List[BaseMessage] | str:
        result = await self.react_agent.ainvoke({"messages": messages})
        return [
            message
            for message in result["messages"]
            if not isinstance(message, (HumanMessage, SystemMessage))
        ]
