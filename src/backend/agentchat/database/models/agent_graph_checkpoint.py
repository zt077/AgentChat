from datetime import datetime
from typing import Optional

from sqlalchemy import Column, DateTime, LargeBinary, String, text
from sqlmodel import Field

from agentchat.database.models.base import SQLModelSerializable


class AgentGraphCheckpointTable(SQLModelSerializable, table=True):
    __tablename__ = "agent_graph_checkpoint"

    thread_id: str = Field(primary_key=True, description="LangGraph thread id")
    checkpoint_ns: str = Field(default="", primary_key=True, description="LangGraph checkpoint namespace")
    checkpoint_id: str = Field(primary_key=True, description="LangGraph checkpoint id")
    checkpoint_type: str = Field(sa_column=Column(String(64), nullable=False))
    checkpoint_payload: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    metadata_type: str = Field(sa_column=Column(String(64), nullable=False))
    metadata_payload: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    parent_checkpoint_id: Optional[str] = Field(default=None, description="Parent checkpoint id")
    create_time: Optional[datetime] = Field(
        sa_column=Column(
            DateTime,
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
        ),
        description="Create time",
    )


class AgentGraphWriteTable(SQLModelSerializable, table=True):
    __tablename__ = "agent_graph_write"

    thread_id: str = Field(primary_key=True, description="LangGraph thread id")
    checkpoint_ns: str = Field(default="", primary_key=True, description="LangGraph checkpoint namespace")
    checkpoint_id: str = Field(primary_key=True, description="LangGraph checkpoint id")
    task_id: str = Field(primary_key=True, description="Task id")
    write_idx: int = Field(primary_key=True, description="Write order index")
    channel: str = Field(sa_column=Column(String(128), nullable=False))
    payload_type: str = Field(sa_column=Column(String(64), nullable=False))
    payload: bytes = Field(sa_column=Column(LargeBinary, nullable=False))
    task_path: str = Field(default="", description="Task path")
    create_time: Optional[datetime] = Field(
        sa_column=Column(
            DateTime,
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
        ),
        description="Create time",
    )
