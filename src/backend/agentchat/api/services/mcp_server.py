from datetime import datetime, timedelta
from typing import Any, Dict

import pytz

from agentchat.api.services.mcp_user_config import MCPUserConfigService
from agentchat.database.dao.mcp_server import MCPServerDao
from agentchat.database.models.user import AdminUser, SystemUser


class MCPService:
    @classmethod
    async def create_mcp_server(
        cls,
        *,
        url: str,
        type: str,
        tools: list,
        params: dict,
        server_name: str,
        user_id: str,
        user_name: str,
        logo_url: str,
        mcp_as_tool_name: str,
        description: str,
        config: dict = None,
        imported_config: dict = None,
        config_enabled: bool = False,
        risk_level: str = "medium",
        approval_policy: str = "auto",
        idempotent: bool = True,
        audit_enabled: bool = True,
    ):
        return await MCPServerDao.create_mcp_server(
            url=url,
            type=type,
            config=config,
            tools=tools,
            params=params,
            server_name=server_name,
            user_id=user_id,
            user_name=user_name,
            mcp_as_tool_name=mcp_as_tool_name,
            description=description,
            config_enabled=config_enabled,
            logo_url=logo_url,
            imported_config=imported_config,
            risk_level=risk_level,
            approval_policy=approval_policy,
            idempotent=idempotent,
            audit_enabled=audit_enabled,
        )

    @classmethod
    async def get_mcp_server_from_id(cls, mcp_server_id):
        result = await MCPServerDao.get_mcp_server_from_id(mcp_server_id)
        return result.to_dict()

    @classmethod
    async def update_mcp_server(cls, server_id: str, update_data: dict):
        if update_data:
            return await MCPServerDao.update_mcp_server(mcp_server_id=server_id, update_data=update_data)

    @classmethod
    async def get_server_from_tool_name(cls, tool_name):
        results = await MCPServerDao.get_server_from_tool_name(tool_name)
        return results.to_dict()

    @classmethod
    async def delete_server_from_id(cls, mcp_server_id):
        return await MCPServerDao.delete_mcp_server(mcp_server_id)

    @classmethod
    async def verify_user_permission(cls, server_id, user_id, action: str = "update"):
        mcp_server = await MCPServerDao.get_mcp_server_from_id(server_id)
        if mcp_server:
            if user_id not in (mcp_server.user_id, AdminUser):
                raise ValueError("No permission to access this MCP server")
        else:
            raise ValueError("Server not found")

    @classmethod
    async def get_all_servers(cls, user_id):
        if user_id in (AdminUser, SystemUser):
            all_servers = await MCPServerDao.get_all_mcp_servers()
        else:
            personal_servers = await MCPServerDao.get_mcp_servers_from_user(user_id)
            admin_servers = await MCPServerDao.get_mcp_servers_from_user(SystemUser)
            all_servers = personal_servers + admin_servers
        all_servers = [server.to_dict() for server in all_servers]
        for server in all_servers:
            user_config = await MCPUserConfigService.show_mcp_user_config(user_id, server["mcp_server_id"])
            if user_config.get("config"):
                server["config"] = user_config.get("config")
        return all_servers

    @classmethod
    async def mcp_server_need_update(cls):
        server = await MCPServerDao.get_first_mcp_server()
        current_time = datetime.now(pytz.timezone("Asia/Shanghai"))
        time_difference = current_time - server.update_time.replace(tzinfo=pytz.timezone("Asia/Shanghai"))
        return time_difference > timedelta(days=7)

    @classmethod
    async def get_mcp_tools_info(cls, server_id):
        server = await MCPServerDao.get_mcp_server_from_id(server_id)
        server = server.to_dict()
        tools_info = []
        for param in server["params"]:
            tool_schema = []
            properties = param["input_schema"]["properties"]
            required = param["input_schema"].get("required", [])
            for param_key, param_value in properties.items():
                tool_schema.append(
                    {
                        "name": param_key,
                        "description": param_value.get("description", ""),
                        "type": param_value.get("type"),
                        "required": param_key in required,
                    }
                )

            tools_info.append(
                {
                    "tool_name": param["name"],
                    "tool_description": param.get("description", ""),
                    "tool_schema": tool_schema,
                }
            )
        return tools_info

    @classmethod
    async def get_mcp_server_ids_from_name(cls, mcp_servers_name, user_id):
        mcp_servers = await MCPServerDao.get_mcp_server_ids_from_name(mcp_servers_name, user_id)
        mcp_servers.extend(await MCPServerDao.get_mcp_server_ids_from_name(mcp_servers_name, SystemUser))
        return [mcp_server.mcp_server_id for mcp_server in mcp_servers]

    @classmethod
    def validate_imported_config(cls, payload: Dict[str, Any]):
        if "mcpServers" not in payload:
            raise ValueError("Missing field: mcpServers")

        mcp_servers = payload["mcpServers"]
        if not isinstance(mcp_servers, dict):
            raise ValueError("mcpServers must be a dict")
        if not mcp_servers:
            raise ValueError("mcpServers cannot be empty")

        for server_name, server_conf in mcp_servers.items():
            if not isinstance(server_name, str) or not server_name.strip():
                raise ValueError(f"Invalid mcpServer name: {server_name}")
            if not isinstance(server_conf, dict):
                raise ValueError(f"mcpServer `{server_name}` config must be an object")

            for required_field in ("type", "url"):
                if required_field not in server_conf:
                    raise ValueError(f"mcpServer `{server_name}` missing required field: {required_field}")
                if not server_conf[required_field]:
                    raise ValueError(f"mcpServer `{server_name}` field `{required_field}` cannot be empty")

            if "headers" in server_conf and not isinstance(server_conf["headers"], dict):
                raise ValueError(f"mcpServer `{server_name}` headers must be a dict")
