from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from sqlalchemy import JSON, Column, DateTime, Text, text
from sqlmodel import Field

from agentchat.database.models.base import SQLModelSerializable


class ToolAuthType(str, Enum):
    bearer: str = "Bearer"
    basic: str = "Basic"


class ToolTable(SQLModelSerializable, table=True):
    __tablename__ = "tool"

    tool_id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True)
    name: Optional[str] = Field(description="Tool name")
    display_name: str = Field(description="Tool display name")
    user_id: str = Field(description="Owner user id")
    logo_url: str = Field(description="Tool logo url")
    description: str = Field(sa_column=Column(Text), description="Tool description")
    openapi_schema: Optional[dict] = Field(default=None, sa_column=Column(JSON), description="OpenAPI schema")
    is_user_defined: bool = Field(default=False, description="Whether this is a user-defined tool")
    auth_config: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON), description="Auth config")
    risk_level: str = Field(default="medium", description="Tool risk level")
    approval_policy: str = Field(default="auto", description="Tool approval policy")
    idempotent: bool = Field(default=True, description="Whether repeated calls are safe")
    audit_enabled: bool = Field(default=True, description="Whether tool execution is audited")
    update_time: Optional[datetime] = Field(
        sa_column=Column(
            DateTime,
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
            onupdate=text("CURRENT_TIMESTAMP"),
        ),
        description="Update time",
    )
    create_time: Optional[datetime] = Field(
        sa_column=Column(
            DateTime,
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
        ),
        description="Create time",
    )
