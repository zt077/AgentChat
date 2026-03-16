from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from sqlalchemy import JSON, Column, DateTime, Float, LargeBinary, String, Text, text
from sqlmodel import Field

from agentchat.database.models.base import SQLModelSerializable


class AgentRunTable(SQLModelSerializable, table=True):
    __tablename__ = "agent_run"

    run_id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True)
    dialog_id: str = Field(index=True, description="Dialog id")
    user_id: str = Field(index=True, description="User id")
    agent_name: str = Field(default="agent", description="Agent name")
    trace_id: Optional[str] = Field(default=None, index=True, description="Trace id")
    status: str = Field(default="running", description="Run status")
    checkpoint_thread_id: str = Field(description="Checkpoint thread id")
    latest_checkpoint_id: Optional[str] = Field(default=None, description="Latest checkpoint id")
    request_payload: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    final_response: Optional[str] = Field(default=None, sa_column=Column(Text))
    error_message: Optional[str] = Field(default=None, sa_column=Column(Text))
    paused_tools: Optional[list[dict[str, Any]]] = Field(default=None, sa_column=Column(JSON))
    create_time: Optional[datetime] = Field(
        sa_column=Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    )
    update_time: Optional[datetime] = Field(
        sa_column=Column(
            DateTime,
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
            onupdate=text("CURRENT_TIMESTAMP"),
        )
    )
    finish_time: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))


class AgentSpanTable(SQLModelSerializable, table=True):
    __tablename__ = "agent_span"

    span_id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True)
    run_id: str = Field(index=True, description="Parent run id")
    trace_id: Optional[str] = Field(default=None, index=True, description="Trace id")
    parent_span_id: Optional[str] = Field(default=None, index=True, description="Parent span id")
    span_type: str = Field(description="Span type")
    name: str = Field(description="Span name")
    status: str = Field(default="ok", description="Span status")
    duration_ms: Optional[float] = Field(default=None, sa_column=Column(Float))
    input_payload: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    output_payload: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    tags: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    create_time: Optional[datetime] = Field(
        sa_column=Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    )
    finish_time: Optional[datetime] = Field(default=None, sa_column=Column(DateTime))


class ToolExecutionAuditTable(SQLModelSerializable, table=True):
    __tablename__ = "tool_execution_audit"

    audit_id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True)
    run_id: str = Field(index=True, description="Parent run id")
    trace_id: Optional[str] = Field(default=None, index=True, description="Trace id")
    tool_name: str = Field(description="Tool name")
    tool_type: str = Field(default="tool", description="Tool type")
    risk_level: str = Field(default="medium", description="Risk level")
    approval_policy: str = Field(default="auto", description="Approval policy")
    approved: bool = Field(default=True, description="Whether execution was approved")
    blocked: bool = Field(default=False, description="Whether execution was blocked")
    idempotent: bool = Field(default=True, description="Whether the tool is idempotent")
    args_payload: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    result_excerpt: Optional[str] = Field(default=None, sa_column=Column(Text))
    error_message: Optional[str] = Field(default=None, sa_column=Column(Text))
    create_time: Optional[datetime] = Field(
        sa_column=Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    )


class AgentEvalRecordTable(SQLModelSerializable, table=True):
    __tablename__ = "agent_eval_record"

    eval_id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True)
    run_id: str = Field(index=True, description="Parent run id")
    dialog_id: str = Field(index=True, description="Dialog id")
    user_id: str = Field(index=True, description="User id")
    trace_id: Optional[str] = Field(default=None, index=True, description="Trace id")
    query: str = Field(sa_column=Column(Text), description="User query")
    response: str = Field(sa_column=Column(Text), description="Assistant response")
    status: str = Field(default="pending", description="Eval status")
    labels: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    tool_trace: Optional[list[dict[str, Any]]] = Field(default=None, sa_column=Column(JSON))
    source_context: Optional[dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    create_time: Optional[datetime] = Field(
        sa_column=Column(DateTime, nullable=False, server_default=text("CURRENT_TIMESTAMP"))
    )
