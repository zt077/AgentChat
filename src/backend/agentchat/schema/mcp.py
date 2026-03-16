from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from agentchat.settings import app_settings


class MCPBaseConfig(BaseModel):
    server_name: str
    transport: str
    personal_config: Optional[Dict[str, Any]] = None


class MCPSSEConfig(MCPBaseConfig):
    transport: Literal["sse"] = "sse"
    url: str
    headers: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = None
    sse_read_timeout: Optional[float] = None
    session_kwargs: Optional[Dict[str, Any]] = None


class MCPStdioConfig(MCPBaseConfig):
    transport: Literal["stdio"] = "stdio"
    command: str
    args: list[str]
    env: Optional[Dict[str, str]] = None
    cwd: Optional[Path] = None
    encoding: str = "utf-8"
    encoding_error_handler: Optional[str] = "ignore"
    session_kwargs: Optional[Dict[str, Any]] = None


class MCPStreamableHttpConfig(MCPBaseConfig):
    transport: Literal["streamable_http"] = "streamable_http"
    url: str
    headers: Optional[Dict[str, Any]] = None
    timeout: Optional[float] = None
    sse_read_timeout: Optional[float] = None
    terminate_on_close: Optional[bool] = None
    session_kwargs: Optional[Dict[str, Any]] = None


class MCPWebsocketConfig(MCPBaseConfig):
    transport: Literal["websocket"] = "websocket"
    url: str
    session_kwargs: Optional[Dict[str, Any]] = None


class MCPServerImportedReq(BaseModel):
    server_name: str
    imported_config: dict
    logo_url: str = app_settings.default_config.get("mcp_logo_url", "")
    risk_level: str = "medium"
    approval_policy: str = "auto"
    idempotent: bool = True
    audit_enabled: bool = True

    @model_validator(mode="after")
    def set_default_logo_url(self):
        if not self.logo_url:
            self.logo_url = app_settings.default_config.get("mcp_logo_url", "")
        return self


class MCPServerUpdateReq(BaseModel):
    server_id: str
    name: str = None
    logo_url: str = None
    imported_config: dict = None
    risk_level: str = None
    approval_policy: str = None
    idempotent: bool = None
    audit_enabled: bool = None


class MCPResponseFormat(BaseModel):
    mcp_as_tool_name: str = Field(..., description="Generated synthetic tool name for the MCP sub-agent")
    description: str = Field(..., description="Generated sub-agent description")
