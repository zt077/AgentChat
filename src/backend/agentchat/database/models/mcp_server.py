from datetime import datetime
from typing import List, Optional
from uuid import uuid4

from sqlalchemy import JSON, VARCHAR, Column, DateTime, text
from sqlmodel import Field

from agentchat.database.models.base import SQLModelSerializable


class MCPServerTable(SQLModelSerializable, table=True):
    __tablename__ = "mcp_server"

    mcp_server_id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True)
    server_name: str = Field(default="MCP Server", description="MCP server name")
    user_id: str = Field(description="Owner user id")
    user_name: str = Field(description="Owner user name")
    description: str = Field(description="Server description for sub-agent usage")
    mcp_as_tool_name: str = Field(description="Synthetic tool name used by the main agent")
    url: str = Field(description="MCP server endpoint")
    type: str = Field(sa_column=Column(VARCHAR(255), nullable=False), description="Transport type")
    logo_url: str = Field(description="MCP server logo url")
    config: List[dict] = Field(default_factory=list, sa_column=Column(JSON), description="Config schema")
    tools: List[str] = Field(default_factory=list, sa_column=Column(JSON), description="MCP tool names")
    params: List[dict] = Field(default_factory=list, sa_column=Column(JSON), description="Tool parameter schemas")
    imported_config: Optional[dict] = Field(default=None, sa_column=Column(JSON), description="Imported config")
    config_enabled: bool = Field(default=False, description="Whether per-user config is required")
    risk_level: str = Field(default="medium", description="MCP risk level")
    approval_policy: str = Field(default="auto", description="MCP approval policy")
    idempotent: bool = Field(default=True, description="Whether repeated MCP calls are safe")
    audit_enabled: bool = Field(default=True, description="Whether MCP execution is audited")
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


class MCPServerStdioTable(SQLModelSerializable, table=True):
    __tablename__ = "mcp_stdio_server"

    mcp_server_id: str = Field(default_factory=lambda: uuid4().hex, primary_key=True)
    mcp_server_path: str = Field(description="Local stdio server path")
    mcp_server_command: str = Field(description="Execution command")
    mcp_server_env: str = Field(description="Execution env")
    user_id: str = Field(description="Owner user id")
    name: str = Field(default="MCP Server", description="Display name")
    create_time: Optional[datetime] = Field(
        sa_column=Column(
            DateTime,
            nullable=False,
            server_default=text("CURRENT_TIMESTAMP"),
        ),
        description="Create time",
    )
