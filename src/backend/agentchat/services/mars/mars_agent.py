import asyncio
import copy
import time
from typing import List

from loguru import logger
from pydantic import BaseModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.prebuilt import create_react_agent

from agentchat.api.services.usage_stats import UsageStatsService
from agentchat.core.callbacks.usage_metadata import UsageMetadataCallbackHandler
from agentchat.core.models.manager import ModelManager
from agentchat.schema.usage_stats import UsageStatsAgentType
from agentchat.services.mars.mars_tools import MarsTool
from agentchat.services.mars.mars_tools.autobuild import construct_auto_build_prompt


class MarsConfig(BaseModel):
    user_id: str


class MarsEnum:
    AutoBuild_Agent = 1
    Retrieval_Knowledge = 2
    AI_News = 3
    Deep_Search = 4


class MarsAgent:
    def __init__(self, mars_config: MarsConfig):
        self.mars_tools = None
        self.mars_config = mars_config

    async def init_mars_agent(self):
        self.mars_tools = await self.setup_mars_tools()
        self.conversation_model = ModelManager.get_conversation_model()
        self.reasoning_model = ModelManager.get_reasoning_model()
        self.react_agent = create_react_agent(model=self.conversation_model, tools=self.mars_tools)

    async def setup_mars_tools(self) -> List[BaseTool]:
        mars_tools = []
        for name in MarsTool:
            raw_tool = copy.deepcopy(MarsTool[name])
            if name == "auto_build_agent":
                auto_build_prompt = await construct_auto_build_prompt(self.mars_config.user_id)
                raw_tool.description = raw_tool.description.replace("{{{user_configs_placeholder}}}", auto_build_prompt)

            async def _call_tool(_raw_tool=raw_tool, **kwargs):
                kwargs.update({"user_id": self.mars_config.user_id})
                if _raw_tool.coroutine:
                    return await _raw_tool.ainvoke(kwargs)
                return _raw_tool.invoke(kwargs)

            mars_tools.append(
                StructuredTool(
                    name=raw_tool.name,
                    description=raw_tool.description,
                    coroutine=_call_tool,
                    args_schema=raw_tool.args_schema,
                )
            )
        return mars_tools

    async def ainvoke_stream(self, messages: List[BaseMessage]):
        self.reasoning_interrupt = asyncio.Event()
        self.mars_output_queue = asyncio.Queue()
        self.is_call_tool = False
        callback = UsageMetadataCallbackHandler()

        async def run_mars_agent():
            try:
                result = await self.react_agent.ainvoke(
                    input={"messages": messages},
                    config={"callbacks": [callback]},
                )
                for message in result["messages"]:
                    if isinstance(message, ToolMessage):
                        self.is_call_tool = True
                        await self.mars_output_queue.put(
                            {
                                "type": "event",
                                "time": time.time(),
                                "data": {
                                    "title": f"Tool: {message.name}",
                                    "status": "END",
                                    "message": str(message.content),
                                },
                            }
                        )
                    elif isinstance(message, AIMessage) and message.content:
                        await self.mars_output_queue.put(
                            {
                                "type": "response_chunk",
                                "time": time.time(),
                                "data": message.content,
                            }
                        )
            finally:
                await self.mars_output_queue.put(None)

        async def run_reasoning_model():
            try:
                response = await self.reasoning_model.astream(messages)
                async for chunk in response:
                    if self.reasoning_interrupt.is_set():
                        break
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                        yield {
                            "type": "reasoning_chunk",
                            "time": time.time(),
                            "data": delta.reasoning_content,
                        }
                    if hasattr(delta, "content") and delta.content:
                        if self.is_call_tool:
                            break
                        yield {
                            "type": "response_chunk",
                            "time": time.time(),
                            "data": delta.content,
                        }
            except Exception as err:
                logger.error(f"Reasoning stream error: {err}")

        yield {
            "type": "response_chunk",
            "time": time.time(),
            "data": "#### Starting task execution\n",
        }

        mars_task = asyncio.create_task(run_mars_agent())

        async for reasoning_chunk in run_reasoning_model():
            yield reasoning_chunk

        while True:
            mars_chunk = await self.mars_output_queue.get()
            if mars_chunk is None:
                break
            yield mars_chunk

        await mars_task

    async def _record_agent_token_usage(self, response: AIMessage | AIMessageChunk | BaseMessage, model):
        if response.usage_metadata:
            await UsageStatsService.create_usage_stats(
                model=model,
                user_id=self.mars_config.user_id,
                agent=UsageStatsAgentType.mars_agent,
                input_tokens=response.usage_metadata.get("input_tokens"),
                output_tokens=response.usage_metadata.get("output_tokens"),
            )
