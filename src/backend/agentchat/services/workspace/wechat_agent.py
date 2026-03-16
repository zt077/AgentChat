import asyncio
import copy
import time
from typing import Any, List

from loguru import logger
from pydantic import BaseModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import create_react_agent

from agentchat.api.services.knowledge import KnowledgeService
from agentchat.api.services.mcp_user_config import MCPUserConfigService
from agentchat.api.services.usage_stats import UsageStatsService
from agentchat.api.services.user import UserService
from agentchat.api.services.workspace_session import WorkSpaceSessionService
from agentchat.core.callbacks import usage_metadata_callback
from agentchat.core.models.manager import ModelManager
from agentchat.database.models.workspace_session import WorkSpaceSessionContext, WorkSpaceSessionCreate
from agentchat.prompts.completion import GenerateTitlePrompt
from agentchat.schema.usage_stats import UsageStatsAgentType
from agentchat.schema.workspace import WorkSpaceAgents
from agentchat.services.mcp.manager import MCPManager
from agentchat.services.rag.handler import RagHandler
from agentchat.tools import WeChatTools
from agentchat.utils.convert import convert_mcp_config


class MCPConfig(BaseModel):
    url: str
    type: str = "sse"
    tools: List[str] = []
    server_name: str
    mcp_server_id: str


class WeChatAgent:
    def __init__(
        self,
        user_id: str,
        session_id: str,
        wechat_account_user: str = None,
        plugins: List[str] = [],
        mcp_configs: List[MCPConfig] = [],
    ):
        self.model = ModelManager.get_conversation_model()
        self.plugin_tools = []
        self.mcp_tools = []
        self.mcp_configs = mcp_configs
        self.tools = []
        self.mcp_manager = MCPManager(convert_mcp_config([mcp_config.model_dump() for mcp_config in mcp_configs]))
        self.plugins = plugins
        self.session_id = session_id
        self.wechat_account_user = wechat_account_user
        self.user_id = user_id
        self.server_dict: dict[str, Any] = {}
        self._initialized = False

    async def init_wechat_agent(self):
        if self._initialized:
            return
        await self.setup_mcp_tools()
        await self.setup_plugin_tools()
        self.tools = self.plugin_tools + self.mcp_tools
        self.react_agent = create_react_agent(model=self.model, tools=self.tools)
        self._initialized = True

    async def setup_mcp_tools(self):
        if not self.mcp_configs:
            self.mcp_tools = []
            return
        raw_tools = await self.mcp_manager.get_mcp_tools()
        mcp_servers_info = await self.mcp_manager.show_mcp_tools()
        self.server_dict = {server_name: [tool["name"] for tool in tools_info] for server_name, tools_info in mcp_servers_info.items()}

        wrapped_tools = []
        for raw_tool in raw_tools:
            async def _call_tool(_raw_tool=raw_tool, **kwargs):
                mcp_config = await MCPUserConfigService.get_mcp_user_config(self.user_id, self.get_mcp_id_by_tool(_raw_tool.name))
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
        self.mcp_tools = wrapped_tools

    async def setup_plugin_tools(self):
        self.plugin_tools = list(WeChatTools.values())

    async def retrival_knowledge_documents(self, query):
        wechat_account_user_id = UserService.get_user_id_by_name(self.wechat_account_user)
        knowledges = await KnowledgeService.select_knowledge(wechat_account_user_id)
        if not knowledges:
            return None
        collection_name = knowledges[0]["id"]
        return await RagHandler.retrieve_ranked_documents(
            top_k=3,
            min_score=0.01,
            query=query,
            index_names=[collection_name],
            collection_names=[collection_name],
            needs_query_rewrite=False,
        )

    async def ainvoke(self, messages: List[BaseMessage]):
        if not self._initialized:
            await self.init_wechat_agent()
        user_messages = copy.deepcopy(messages)

        react_agent_result = None
        if self.tools:
            react_start = time.perf_counter()
            react_agent_task = asyncio.create_task(
                self.react_agent.ainvoke(input={"messages": messages}, config={"callbacks": [usage_metadata_callback]})
            )
            retrival_task = asyncio.create_task(self.retrival_knowledge_documents(query=user_messages[-1].content))
            retrival_result = await retrival_task
            if retrival_result:
                user_messages[0].content = user_messages[0].content + f"\n\n## Additional context\n{retrival_result}"
            try:
                react_agent_result = await asyncio.wait_for(react_agent_task, timeout=1.0)
                logger.info(f"React agent task completed in {time.perf_counter() - react_start:.2f}s")
            except asyncio.TimeoutError:
                react_agent_task.cancel()

        tool_messages = []
        if react_agent_result:
            tool_messages = [
                msg
                for msg in react_agent_result["messages"][:-1]
                if isinstance(msg, ToolMessage) or (isinstance(msg, AIMessage) and msg.tool_calls)
            ]

        response = await self.model.ainvoke(user_messages + tool_messages)
        await self._add_workspace_session(
            title="wechat-chat",
            contexts=WorkSpaceSessionContext(query=user_messages[-1].content, answer=response.content),
        )
        return response

    async def _generate_title(self, query):
        session = await WorkSpaceSessionService.get_workspace_session_from_id(self.session_id, self.wechat_account_user)
        if session:
            return session.get("title")
        response = await self.model.ainvoke(input=GenerateTitlePrompt.format(query=query), config={"callbacks": [usage_metadata_callback]})
        return response.content

    async def _add_workspace_session(self, title, contexts: WorkSpaceSessionContext):
        session = await WorkSpaceSessionService.get_workspace_session_from_id(self.session_id, self.wechat_account_user)
        if session:
            await WorkSpaceSessionService.update_workspace_session_contexts(
                session_id=self.session_id,
                session_context=contexts.model_dump(),
            )
        else:
            await WorkSpaceSessionService.create_workspace_session(
                WorkSpaceSessionCreate(
                    title=title,
                    user_id=self.user_id,
                    session_id=self.session_id,
                    contexts=[contexts.model_dump()],
                    agent=WorkSpaceAgents.WeChatAgent.value,
                )
            )

    async def astream(self, messages: List[BaseMessage]):
        if not self._initialized:
            await self.init_wechat_agent()
        user_messages = copy.deepcopy(messages)

        tool_messages = []
        if self.tools:
            react_result = await self.react_agent.ainvoke(
                input={"messages": messages},
                config={"callbacks": [usage_metadata_callback]},
            )
            tool_messages = [
                msg
                for msg in react_result["messages"][:-1]
                if isinstance(msg, ToolMessage) or (isinstance(msg, AIMessage) and msg.tool_calls)
            ]

        retrival_result = await self.retrival_knowledge_documents(query=user_messages[-1].content)
        if retrival_result:
            user_messages[0].content = user_messages[0].content + f"\n\n## Additional context\n{retrival_result}"

        final_answer = ""
        async for chunk in self.model.astream(user_messages + tool_messages):
            yield {"event": "task_result", "data": {"message": chunk.content}}
            final_answer += chunk.content

        await self._add_workspace_session(
            title="wechat-chat",
            contexts=WorkSpaceSessionContext(query=user_messages[-1].content, answer=final_answer),
        )

    async def _record_agent_token_usage(self, response: AIMessage | AIMessageChunk | BaseMessage, model):
        if response.usage_metadata:
            await UsageStatsService.create_usage_stats(
                model=model,
                user_id=self.user_id,
                agent=UsageStatsAgentType.wechat_agent,
                input_tokens=response.usage_metadata.get("input_tokens"),
                output_tokens=response.usage_metadata.get("output_tokens"),
            )

    def get_mcp_id_by_tool(self, tool_name):
        for server_name, tools in self.server_dict.items():
            if tool_name in tools:
                for config in self.mcp_configs:
                    if server_name == config.server_name:
                        return config.mcp_server_id
        return None
