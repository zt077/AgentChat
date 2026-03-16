from __future__ import annotations

from datetime import datetime
from time import perf_counter
from typing import Any, Optional

from sqlmodel import select

from agentchat.database.models.agent_observability import (
    AgentEvalRecordTable,
    AgentRunTable,
    AgentSpanTable,
    ToolExecutionAuditTable,
)
from agentchat.database.session import async_session_getter


def _safe_excerpt(value: Any, max_length: int = 2000) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text if len(text) <= max_length else f"{text[: max_length - 3]}..."


class ObservabilityService:
    @classmethod
    async def create_run(
        cls,
        *,
        run_id: str,
        dialog_id: str,
        user_id: str,
        agent_name: str,
        trace_id: Optional[str],
        checkpoint_thread_id: str,
        request_payload: Optional[dict[str, Any]] = None,
    ) -> AgentRunTable:
        async with async_session_getter() as session:
            run = AgentRunTable(
                run_id=run_id,
                dialog_id=dialog_id,
                user_id=user_id,
                agent_name=agent_name,
                trace_id=trace_id,
                checkpoint_thread_id=checkpoint_thread_id,
                request_payload=request_payload,
            )
            session.add(run)
            await session.commit()
            await session.refresh(run)
            return run

    @classmethod
    async def get_run(cls, run_id: str) -> Optional[AgentRunTable]:
        async with async_session_getter() as session:
            result = await session.exec(select(AgentRunTable).where(AgentRunTable.run_id == run_id))
            return result.first()

    @classmethod
    async def update_run(
        cls,
        run_id: str,
        *,
        status: Optional[str] = None,
        latest_checkpoint_id: Optional[str] = None,
        final_response: Optional[str] = None,
        error_message: Optional[str] = None,
        paused_tools: Optional[list[dict[str, Any]]] = None,
        finish: bool = False,
    ) -> None:
        async with async_session_getter() as session:
            result = await session.exec(select(AgentRunTable).where(AgentRunTable.run_id == run_id))
            run = result.first()
            if run is None:
                return
            if status is not None:
                run.status = status
            if latest_checkpoint_id is not None:
                run.latest_checkpoint_id = latest_checkpoint_id
            if final_response is not None:
                run.final_response = final_response
            if error_message is not None:
                run.error_message = _safe_excerpt(error_message, max_length=4000)
            if paused_tools is not None:
                run.paused_tools = paused_tools
            if finish:
                run.finish_time = datetime.now()
            session.add(run)
            await session.commit()

    @classmethod
    async def record_span(
        cls,
        *,
        run_id: str,
        trace_id: Optional[str],
        span_type: str,
        name: str,
        started_at: float,
        status: str = "ok",
        input_payload: Optional[dict[str, Any]] = None,
        output_payload: Optional[dict[str, Any]] = None,
        tags: Optional[dict[str, Any]] = None,
        parent_span_id: Optional[str] = None,
    ) -> None:
        finished_at = perf_counter()
        async with async_session_getter() as session:
            span = AgentSpanTable(
                run_id=run_id,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                span_type=span_type,
                name=name,
                status=status,
                duration_ms=(finished_at - started_at) * 1000,
                input_payload=input_payload,
                output_payload=output_payload,
                tags=tags,
                finish_time=datetime.now(),
            )
            session.add(span)
            await session.commit()

    @classmethod
    async def record_tool_audit(
        cls,
        *,
        run_id: str,
        trace_id: Optional[str],
        tool_name: str,
        tool_type: str,
        risk_level: str,
        approval_policy: str,
        approved: bool,
        blocked: bool,
        idempotent: bool,
        args_payload: Optional[dict[str, Any]] = None,
        result_excerpt: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        async with async_session_getter() as session:
            audit = ToolExecutionAuditTable(
                run_id=run_id,
                trace_id=trace_id,
                tool_name=tool_name,
                tool_type=tool_type,
                risk_level=risk_level,
                approval_policy=approval_policy,
                approved=approved,
                blocked=blocked,
                idempotent=idempotent,
                args_payload=args_payload,
                result_excerpt=_safe_excerpt(result_excerpt),
                error_message=_safe_excerpt(error_message, max_length=4000),
            )
            session.add(audit)
            await session.commit()

    @classmethod
    async def create_eval_record(
        cls,
        *,
        run_id: str,
        dialog_id: str,
        user_id: str,
        trace_id: Optional[str],
        query: str,
        response: str,
        tool_trace: Optional[list[dict[str, Any]]] = None,
        source_context: Optional[dict[str, Any]] = None,
        labels: Optional[dict[str, Any]] = None,
    ) -> None:
        async with async_session_getter() as session:
            eval_record = AgentEvalRecordTable(
                run_id=run_id,
                dialog_id=dialog_id,
                user_id=user_id,
                trace_id=trace_id,
                query=query,
                response=response,
                tool_trace=tool_trace,
                source_context=source_context,
                labels=labels,
            )
            session.add(eval_record)
            await session.commit()

    @classmethod
    async def list_runs(
        cls,
        *,
        user_id: str,
        dialog_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        async with async_session_getter() as session:
            statement = select(AgentRunTable).where(AgentRunTable.user_id == user_id)
            if dialog_id:
                statement = statement.where(AgentRunTable.dialog_id == dialog_id)
            if status:
                statement = statement.where(AgentRunTable.status == status)
            statement = statement.order_by(AgentRunTable.create_time.desc()).limit(limit)
            results = await session.exec(statement)
            return [row.to_dict() for row in results.all()]

    @classmethod
    async def get_run_detail(cls, *, run_id: str, user_id: str) -> Optional[dict[str, Any]]:
        async with async_session_getter() as session:
            run_result = await session.exec(
                select(AgentRunTable).where(
                    AgentRunTable.run_id == run_id,
                    AgentRunTable.user_id == user_id,
                )
            )
            run = run_result.first()
            if run is None:
                return None

            spans = await session.exec(select(AgentSpanTable).where(AgentSpanTable.run_id == run_id))
            audits = await session.exec(select(ToolExecutionAuditTable).where(ToolExecutionAuditTable.run_id == run_id))
            eval_records = await session.exec(select(AgentEvalRecordTable).where(AgentEvalRecordTable.run_id == run_id))

            return {
                "run": run.to_dict(),
                "spans": [span.to_dict() for span in spans.all()],
                "tool_audits": [audit.to_dict() for audit in audits.all()],
                "eval_records": [record.to_dict() for record in eval_records.all()],
            }
