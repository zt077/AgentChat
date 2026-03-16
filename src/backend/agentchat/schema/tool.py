from pydantic import BaseModel


class ToolCreateReq(BaseModel):
    display_name: str
    description: str
    logo_url: str
    auth_config: dict = None
    openapi_schema: dict = None
    risk_level: str = "medium"
    approval_policy: str = "auto"
    idempotent: bool = True
    audit_enabled: bool = True


class ToolUpdateReq(BaseModel):
    tool_id: str
    description: str = None
    logo_url: str = None
    auth_config: dict = None
    display_name: str = None
    openapi_schema: dict = None
    risk_level: str = None
    approval_policy: str = None
    idempotent: bool = None
    audit_enabled: bool = None


class ToolDeleteReq(BaseModel):
    tool_id: str
