import time
from dataclasses import dataclass
from time import perf_counter
from typing import Any, AsyncGenerator, Dict, List, NotRequired, Optional, TypedDict, Union

from loguru import logger
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.config import get_stream_writer
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.types import StateSnapshot

from agentchat.api.services.observability import ObservabilityService
from agentchat.core.callbacks import usage_metadata_callback
from agentchat.prompts.completion import DEFAULT_CALL_PROMPT
from agentchat.services.checkpoint import MySQLCheckpointSaver


class StreamEventData(TypedDict, total=False):
    title: str
    status: str
    message: str
    run_id: str
    checkpoint_id: str
    tool_name: str


class StreamOutput(TypedDict):
    type: str
    timestamp: float
    data: Union[StreamEventData, Dict[str, Any]]


class ReactAgentState(MessagesState):
    tool_call_count: NotRequired[int]
    model_call_count: NotRequired[int]


@dataclass
class ToolGovernance:
    name: str
    display_name: str
    tool_type: str = "tool"
    risk_level: str = "medium"
    approval_policy: str = "auto"
    idempotent: bool = True
    audit_enabled: bool = True


class ReactAgent:
    def __init__(
        self,
        model: BaseChatModel,
        system_prompt: Optional[str] = None,
        tools: Optional[List[BaseTool]] = None,
        tool_metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
        checkpointer: Optional[MySQLCheckpointSaver] = None,
    ):
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.tool_metadata_map = tool_metadata_map or {}
        self.graph = None
        self.checkpointer = checkpointer or MySQLCheckpointSaver()

        self.current_run_id: Optional[str] = None
        self.current_trace_id: Optional[str] = None
        self.approved_tools: set[str] = set()

        self.last_response_content = ""
        self.last_run_status = "idle"
        self.last_checkpoint_id: Optional[str] = None
        self.last_paused_tools: list[dict[str, Any]] = []

    def _wrap_stream_output(self, output_type: str, data: Dict[str, Any]) -> StreamOutput:
        return {
            "type": output_type,
            "timestamp": time.time(),
            "data": data,
        }

    async def _init_agent(self):
        if self.graph is None:
            self.graph = await self._setup_react_graph()

    def get_tool_by_name(self, tool_name: str) -> Optional[BaseTool]:
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        return None

    def get_tool_governance(self, tool_name: str) -> ToolGovernance:
        metadata = self.tool_metadata_map.get(tool_name, {})
        return ToolGovernance(
            name=tool_name,
            display_name=metadata.get("name", tool_name),
            tool_type=metadata.get("type", "tool"),
            risk_level=metadata.get("risk_level", "medium"),
            approval_policy=metadata.get("approval_policy", "auto"),
            idempotent=metadata.get("idempotent", True),
            audit_enabled=metadata.get("audit_enabled", True),
        )

    def _tool_requires_approval(self, governance: ToolGovernance) -> bool:
        if governance.approval_policy == "always":
            return True
        if governance.approval_policy == "on_high_risk":
            return governance.risk_level in {"high", "critical"}
        return False

    def _tool_is_approved(self, governance: ToolGovernance) -> bool:
        if not self._tool_requires_approval(governance):
            return True
        return governance.name in self.approved_tools

    def _extract_pending_tools(self, snapshot: StateSnapshot) -> list[dict[str, Any]]:
        values = snapshot.values or {}
        messages = values.get("messages", [])
        if not messages:
            return []
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return []

        pending_tools = []
        for tool_call in last_message.tool_calls or []:
            governance = self.get_tool_governance(tool_call["name"])
            pending_tools.append(
                {
                    "tool_name": tool_call["name"],
                    "display_name": governance.display_name,
                    "tool_type": governance.tool_type,
                    "risk_level": governance.risk_level,
                    "approval_policy": governance.approval_policy,
                    "idempotent": governance.idempotent,
                    "requires_approval": self._tool_requires_approval(governance),
                    "approved": self._tool_is_approved(governance),
                    "args": tool_call.get("args", {}),
                }
            )
        return pending_tools

    async def _setup_react_graph(self):
        workflow = StateGraph(ReactAgentState)
        workflow.add_node("call_tool_node", self._call_tool_node)
        workflow.add_node("execute_tool_node", self._execute_tool_node)
        workflow.add_edge(START, "call_tool_node")
        workflow.add_conditional_edges("call_tool_node", self._should_continue)
        workflow.add_edge("execute_tool_node", "call_tool_node")
        interrupt_before = ["execute_tool_node"] if self.tools else None
        return workflow.compile(checkpointer=self.checkpointer, interrupt_before=interrupt_before)

    async def _should_continue(self, state: ReactAgentState) -> Union[str, Any]:
        last_message = state["messages"][-1]
        if last_message.tool_calls:
            return "execute_tool_node"
        return END

    async def _call_tool_node(self, state: ReactAgentState, config: RunnableConfig) -> Dict[str, List[BaseMessage]]:
        stream_writer = get_stream_writer()
        is_first_call = state.get("tool_call_count", 0) == 0
        select_tool_message = (
            "Select tools for current request"
            if is_first_call
            else f"Continue tool selection after {state.get('tool_call_count', 0)} tool rounds"
        )

        stream_writer(
            {
                "title": select_tool_message,
                "status": "START",
                "message": "Inspecting whether a tool call is needed.",
                "run_id": self.current_run_id,
            }
        )

        started_at = perf_counter()
        status = "ok"
        tool_call_count = 0
        tool_invocation_model = self.model.bind_tools(self.tools) if self.tools else self.model
        try:
            response: AIMessage = await tool_invocation_model.ainvoke(
                state["messages"],
                config={"callbacks": [usage_metadata_callback]},
            )
            tool_call_names = sorted({tool_call["name"] for tool_call in response.tool_calls or []})
            tool_call_count = len(tool_call_names)
            stream_writer(
                {
                    "title": select_tool_message,
                    "status": "END",
                    "message": (
                        f"Selected tools: {', '.join(tool_call_names)}"
                        if response.tool_calls
                        else "Model answered directly without tools."
                    ),
                    "run_id": self.current_run_id,
                }
            )
            state["messages"].append(response)
            return {"messages": state["messages"]}
        except Exception as err:
            status = "error"
            stream_writer(
                {
                    "title": select_tool_message,
                    "status": "ERROR",
                    "message": str(err),
                    "run_id": self.current_run_id,
                }
            )
            raise
        finally:
            if self.current_run_id:
                await ObservabilityService.record_span(
                    run_id=self.current_run_id,
                    trace_id=self.current_trace_id,
                    span_type="model",
                    name="main_agent_tool_selection",
                    started_at=started_at,
                    status=status,
                    input_payload={"message_count": len(state["messages"])},
                    output_payload={"tool_call_count": tool_call_count},
                )

    async def _execute_tool_node(self, state: ReactAgentState) -> Dict[str, Any]:
        stream_writer = get_stream_writer()
        last_message = state["messages"][-1]
        tool_calls = last_message.tool_calls
        tool_messages: List[BaseMessage] = []

        if not tool_calls:
            logger.warning("Execute tool node reached without tool calls.")
            return {"messages": state["messages"], "tool_call_count": state.get("tool_call_count", 0)}

        for tool_call in tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            tool_call_id = tool_call["id"]
            governance = self.get_tool_governance(tool_name)
            tool_title = f"Execute {governance.tool_type}: {governance.display_name}"

            if not self._tool_is_approved(governance):
                block_message = f"Tool {governance.display_name} requires approval before execution."
                stream_writer(
                    {
                        "status": "PAUSED",
                        "title": tool_title,
                        "message": block_message,
                        "run_id": self.current_run_id,
                    }
                )
                tool_messages.append(ToolMessage(content=block_message, name=tool_name, tool_call_id=tool_call_id))
                if self.current_run_id and governance.audit_enabled:
                    await ObservabilityService.record_tool_audit(
                        run_id=self.current_run_id,
                        trace_id=self.current_trace_id,
                        tool_name=tool_name,
                        tool_type=governance.tool_type,
                        risk_level=governance.risk_level,
                        approval_policy=governance.approval_policy,
                        approved=False,
                        blocked=True,
                        idempotent=governance.idempotent,
                        args_payload=tool_args,
                        error_message=block_message,
                    )
                continue

            started_at = perf_counter()
            status = "ok"
            try:
                stream_writer(
                    {
                        "status": "START",
                        "title": tool_title,
                        "message": f"args={tool_args}",
                        "run_id": self.current_run_id,
                        "tool_name": tool_name,
                    }
                )
                current_tool = self.get_tool_by_name(tool_name)
                if current_tool is None:
                    raise ValueError(f"Tool '{tool_name}' not found.")

                if current_tool.coroutine:
                    tool_result = await current_tool.ainvoke(tool_args)
                else:
                    tool_result = current_tool.invoke(tool_args)

                tool_result_str = str(tool_result)
                stream_writer(
                    {
                        "status": "END",
                        "title": tool_title,
                        "message": f"result={tool_result_str}",
                        "run_id": self.current_run_id,
                        "tool_name": tool_name,
                    }
                )
                tool_messages.append(ToolMessage(content=tool_result_str, name=tool_name, tool_call_id=tool_call_id))

                if self.current_run_id and governance.audit_enabled:
                    await ObservabilityService.record_tool_audit(
                        run_id=self.current_run_id,
                        trace_id=self.current_trace_id,
                        tool_name=tool_name,
                        tool_type=governance.tool_type,
                        risk_level=governance.risk_level,
                        approval_policy=governance.approval_policy,
                        approved=True,
                        blocked=False,
                        idempotent=governance.idempotent,
                        args_payload=tool_args,
                        result_excerpt=tool_result_str,
                    )
            except Exception as err:
                status = "error"
                error_message = f"Tool {tool_name} failed: {err}"
                stream_writer(
                    {
                        "status": "ERROR",
                        "title": tool_title,
                        "message": error_message,
                        "run_id": self.current_run_id,
                        "tool_name": tool_name,
                    }
                )
                logger.error(error_message)
                tool_messages.append(ToolMessage(content=error_message, name=tool_name, tool_call_id=tool_call_id))
                if self.current_run_id and governance.audit_enabled:
                    await ObservabilityService.record_tool_audit(
                        run_id=self.current_run_id,
                        trace_id=self.current_trace_id,
                        tool_name=tool_name,
                        tool_type=governance.tool_type,
                        risk_level=governance.risk_level,
                        approval_policy=governance.approval_policy,
                        approved=True,
                        blocked=False,
                        idempotent=governance.idempotent,
                        args_payload=tool_args,
                        error_message=error_message,
                    )
            finally:
                if self.current_run_id:
                    await ObservabilityService.record_span(
                        run_id=self.current_run_id,
                        trace_id=self.current_trace_id,
                        span_type="tool",
                        name=tool_name,
                        started_at=started_at,
                        status=status,
                        input_payload={"args": tool_args},
                        output_payload={"tool_type": governance.tool_type, "risk_level": governance.risk_level},
                    )

        state["messages"].extend(tool_messages)
        return {"messages": state["messages"], "tool_call_count": state.get("tool_call_count", 0) + 1}

    async def astream(
        self,
        messages: Optional[List[BaseMessage]],
        *,
        run_id: str,
        trace_id: Optional[str] = None,
        resume: bool = False,
        approved_tools: Optional[List[str]] = None,
        dialog_id: Optional[str] = None,
    ) -> AsyncGenerator[StreamOutput, None]:
        if not resume:
            if not messages or not isinstance(messages[-1], (HumanMessage, AIMessage, ToolMessage)):
                logger.warning("Input messages list is empty or last message type is unexpected.")
                return
            if self.system_prompt and not any(isinstance(m, SystemMessage) for m in messages):
                messages = [SystemMessage(content=self.system_prompt or DEFAULT_CALL_PROMPT), *messages]

        await self._init_agent()

        self.current_run_id = run_id
        self.current_trace_id = trace_id
        self.approved_tools = set(approved_tools or [])
        self.last_response_content = ""
        self.last_run_status = "running"
        self.last_paused_tools = []

        config: RunnableConfig = {
            "configurable": {
                "thread_id": run_id,
                "checkpoint_ns": "main-agent",
            },
            "metadata": {
                "run_id": run_id,
                "trace_id": trace_id,
                "dialog_id": dialog_id,
            },
            "callbacks": [usage_metadata_callback],
        }

        current_input: Optional[Dict[str, Any]] = None
        if not resume:
            current_input = {
                "messages": messages,
                "tool_call_count": 0,
                "model_call_count": 0,
            }

        try:
            while True:
                async for stream_type, token in self.graph.astream(
                    input=current_input,
                    config=config,
                    stream_mode=["messages", "custom"],
                ):
                    if stream_type == "custom":
                        yield self._wrap_stream_output("event", token)
                    elif stream_type == "messages" and isinstance(token[0], AIMessageChunk):
                        if token[0].content:
                            self.last_response_content += token[0].content
                            yield self._wrap_stream_output(
                                "response_chunk",
                                {
                                    "chunk": token[0].content,
                                    "accumulated": self.last_response_content,
                                    "run_id": run_id,
                                },
                            )

                snapshot = await self.graph.aget_state(config)
                self.last_checkpoint_id = snapshot.config["configurable"].get("checkpoint_id")

                if self.current_run_id:
                    await ObservabilityService.update_run(
                        self.current_run_id,
                        latest_checkpoint_id=self.last_checkpoint_id,
                    )

                pending_tools = self._extract_pending_tools(snapshot)
                if "execute_tool_node" in snapshot.next:
                    if all(item["approved"] for item in pending_tools):
                        if pending_tools:
                            yield self._wrap_stream_output(
                                "event",
                                {
                                    "title": "Tool governance",
                                    "status": "AUTO_APPROVED",
                                    "message": "Pending tools passed governance checks. Resuming execution.",
                                    "run_id": run_id,
                                    "checkpoint_id": self.last_checkpoint_id,
                                },
                            )
                        current_input = None
                        continue

                    self.last_paused_tools = pending_tools
                    self.last_run_status = "paused"
                    if self.current_run_id:
                        await ObservabilityService.update_run(
                            self.current_run_id,
                            status="paused",
                            latest_checkpoint_id=self.last_checkpoint_id,
                            paused_tools=pending_tools,
                        )
                        for item in pending_tools:
                            if item["requires_approval"] and not item["approved"]:
                                await ObservabilityService.record_tool_audit(
                                    run_id=self.current_run_id,
                                    trace_id=self.current_trace_id,
                                    tool_name=item["tool_name"],
                                    tool_type=item["tool_type"],
                                    risk_level=item["risk_level"],
                                    approval_policy=item["approval_policy"],
                                    approved=False,
                                    blocked=True,
                                    idempotent=item["idempotent"],
                                    args_payload=item["args"],
                                    error_message="Execution paused pending approval.",
                                )
                    yield self._wrap_stream_output(
                        "approval_required",
                        {
                            "title": "Approval required",
                            "status": "PAUSED",
                            "message": "Execution paused because one or more tools require approval.",
                            "run_id": run_id,
                            "checkpoint_id": self.last_checkpoint_id,
                            "tools": pending_tools,
                        },
                    )
                    break

                self.last_run_status = "completed"
                if self.current_run_id:
                    await ObservabilityService.update_run(
                        self.current_run_id,
                        status="completed",
                        latest_checkpoint_id=self.last_checkpoint_id,
                    )
                break

        except Exception as err:
            self.last_run_status = "failed"
            logger.error(f"Agent execution error: {err}")
            if self.current_run_id:
                await ObservabilityService.update_run(
                    self.current_run_id,
                    status="failed",
                    latest_checkpoint_id=self.last_checkpoint_id,
                    error_message=str(err),
                    finish=True,
                )
            if not self.last_response_content:
                error_chunk = "Execution failed. Please retry the request."
                yield self._wrap_stream_output(
                    "response_chunk",
                    {
                        "chunk": error_chunk,
                        "accumulated": error_chunk,
                        "run_id": run_id,
                    },
                )
