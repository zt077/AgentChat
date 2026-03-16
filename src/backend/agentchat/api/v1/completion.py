import json
from typing import Callable, List, Optional
from uuid import uuid4

import loguru
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from starlette.types import Receive

from agentchat.api.services.dialog import DialogService
from agentchat.api.services.history import HistoryService
from agentchat.api.services.observability import ObservabilityService
from agentchat.api.services.user import UserPayload, get_login_user
from agentchat.core.agents.general_agent import AgentConfig, GeneralAgent
from agentchat.prompts.completion import SYSTEM_PROMPT
from agentchat.schema.completion import CompletionReq
from agentchat.services.memory.client import memory_client
from agentchat.utils.contexts import (
    get_trace_id_context,
    set_agent_name_context,
    set_user_id_context,
)
from agentchat.utils.helpers import (
    build_completion_history_messages,
    build_completion_system_prompt,
    build_completion_user_input,
)

router = APIRouter(tags=["Completion"])


class WatchedStreamingResponse(StreamingResponse):
    def __init__(
        self,
        content,
        callback: Callable = None,
        status_code: int = 200,
        headers=None,
        media_type: str | None = None,
        background=None,
    ):
        super().__init__(content, status_code, headers, media_type, background)
        self.callback = callback

    async def listen_for_disconnect(self, receive: Receive) -> None:
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                loguru.logger.info("http.disconnect. stop task and streaming")
                if self.callback:
                    self.callback()
                break


@router.post("/completion", description="Main chat completion endpoint")
async def completion(
    *,
    req: CompletionReq,
    login_user: UserPayload = Depends(get_login_user),
):
    db_config = await DialogService.get_agent_by_dialog_id(dialog_id=req.dialog_id)
    agent_config = AgentConfig(**db_config)

    set_user_id_context(login_user.user_id)
    set_agent_name_context(agent_config.name)
    agent_config.user_id = login_user.user_id

    trace_id: Optional[str]
    try:
        trace_id = get_trace_id_context()
    except Exception:
        trace_id = None

    run_id = req.run_id or uuid4().hex
    chat_agent = GeneralAgent(agent_config)
    await chat_agent.init_agent()

    original_user_input = req.user_input or ""
    events: list[dict] = []

    existing_run = None
    if req.resume:
        existing_run = await ObservabilityService.get_run(run_id)
        if existing_run is None:
            raise HTTPException(status_code=404, detail="run_id not found")
        if existing_run.user_id != login_user.user_id:
            raise HTTPException(status_code=403, detail="No permission to resume this run")
        if existing_run.dialog_id != req.dialog_id:
            raise HTTPException(status_code=400, detail="dialog_id does not match the run")
        await ObservabilityService.update_run(run_id, status="running", paused_tools=[])
        messages: Optional[List[BaseMessage]] = None
        original_user_input = (existing_run.request_payload or {}).get("user_input", "")
    else:
        await ObservabilityService.create_run(
            run_id=run_id,
            dialog_id=req.dialog_id,
            user_id=login_user.user_id,
            agent_name=agent_config.name,
            trace_id=trace_id,
            checkpoint_thread_id=run_id,
            request_payload={
                "user_input": original_user_input,
                "file_url": req.file_url,
                "dialog_id": req.dialog_id,
            },
        )

        req.user_input = build_completion_user_input(file_url=req.file_url, user_input=original_user_input)

        system_prompt = agent_config.system_prompt if agent_config.system_prompt.strip() else SYSTEM_PROMPT
        if agent_config.enable_memory:
            history = await memory_client.search(query=original_user_input, run_id=req.dialog_id)
            history_text = "\n".join(msg.get("memory", "") for msg in history.get("results", []))
        else:
            history_records = await HistoryService.select_history(dialog_id=req.dialog_id)
            history_text = build_completion_history_messages(history_records)

        system_prompt = build_completion_system_prompt(system_prompt, history_text)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=req.user_input)]

        await HistoryService.save_chat_history(
            role="user",
            content=original_user_input,
            events=[],
            dialog_id=req.dialog_id,
            memory_enable=agent_config.enable_memory,
        )

    async def general_generate():
        response_content = ""
        run_info_event = {
            "type": "run_info",
            "timestamp": 0,
            "data": {
                "run_id": run_id,
                "dialog_id": req.dialog_id,
                "resume": req.resume,
                "approved_tools": req.approved_tools,
            },
        }
        yield f"data: {json.dumps(run_info_event, ensure_ascii=False)}\n\n"

        try:
            async for event in chat_agent.astream(
                messages,
                run_id=run_id,
                trace_id=trace_id,
                dialog_id=req.dialog_id,
                resume=req.resume,
                approved_tools=req.approved_tools,
            ):
                if event.get("type") == "response_chunk":
                    response_content = event["data"].get("accumulated", response_content)
                else:
                    events.append(event)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        finally:
            if chat_agent.last_run_status == "completed":
                if agent_config.enable_memory and original_user_input:
                    await memory_client.add(
                        messages=[
                            {"role": "user", "content": original_user_input},
                            {"role": "assistant", "content": response_content},
                        ],
                        run_id=req.dialog_id,
                    )

                if response_content:
                    await HistoryService.save_chat_history(
                        role="assistant",
                        content=response_content,
                        events=events,
                        dialog_id=req.dialog_id,
                        memory_enable=agent_config.enable_memory,
                    )

                await ObservabilityService.update_run(
                    run_id,
                    status="completed",
                    latest_checkpoint_id=chat_agent.last_checkpoint_id,
                    final_response=response_content,
                    finish=True,
                )
                run_detail = await ObservabilityService.get_run_detail(run_id=run_id, user_id=login_user.user_id)
                await ObservabilityService.create_eval_record(
                    run_id=run_id,
                    dialog_id=req.dialog_id,
                    user_id=login_user.user_id,
                    trace_id=trace_id,
                    query=original_user_input or "",
                    response=response_content,
                    tool_trace=(run_detail or {}).get("tool_audits"),
                    source_context=chat_agent.last_context_package.model_dump() if chat_agent.last_context_package else None,
                )
            elif chat_agent.last_run_status == "paused":
                await ObservabilityService.update_run(
                    run_id,
                    status="paused",
                    latest_checkpoint_id=chat_agent.last_checkpoint_id,
                    paused_tools=chat_agent.last_paused_tools,
                )
            elif chat_agent.last_run_status == "failed":
                await ObservabilityService.update_run(
                    run_id,
                    status="failed",
                    latest_checkpoint_id=chat_agent.last_checkpoint_id,
                    finish=True,
                )

    return WatchedStreamingResponse(
        content=general_generate(),
        callback=chat_agent.stop_streaming_callback,
        media_type="text/event-stream",
    )
