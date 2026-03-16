from sqlmodel import and_, delete, func, select, update

from agentchat.database.models.mcp_server import MCPServerTable
from agentchat.database.session import session_getter


class MCPServerDao:
    @classmethod
    async def create_mcp_server(
        cls,
        *,
        url: str,
        type: str,
        config: dict,
        tools: list,
        params: dict,
        config_enabled: bool,
        logo_url: str,
        server_name: str,
        user_id: str,
        user_name: str,
        mcp_as_tool_name: str,
        description: str,
        imported_config: dict = None,
        risk_level: str = "medium",
        approval_policy: str = "auto",
        idempotent: bool = True,
        audit_enabled: bool = True,
    ):
        with session_getter() as session:
            mcp_server = MCPServerTable(
                url=url,
                type=type,
                config=config,
                tools=tools,
                params=params,
                server_name=server_name,
                user_id=user_id,
                logo_url=logo_url,
                user_name=user_name,
                imported_config=imported_config,
                config_enabled=config_enabled,
                mcp_as_tool_name=mcp_as_tool_name,
                description=description,
                risk_level=risk_level,
                approval_policy=approval_policy,
                idempotent=idempotent,
                audit_enabled=audit_enabled,
            )
            session.add(mcp_server)
            session.commit()

    @classmethod
    async def get_mcp_server_from_id(cls, mcp_server_id):
        with session_getter() as session:
            sql = select(MCPServerTable).where(MCPServerTable.mcp_server_id == mcp_server_id)
            return session.exec(sql).first()

    @classmethod
    async def delete_mcp_server(cls, mcp_server_id):
        with session_getter() as session:
            sql = delete(MCPServerTable).where(MCPServerTable.mcp_server_id == mcp_server_id)
            session.exec(sql)
            session.commit()

    @classmethod
    async def update_mcp_server(cls, mcp_server_id: str, update_data: dict):
        with session_getter() as session:
            sql = update(MCPServerTable).where(MCPServerTable.mcp_server_id == mcp_server_id).values(**update_data)
            session.exec(sql)
            session.commit()

    @classmethod
    async def get_first_mcp_server(cls):
        with session_getter() as session:
            statement = select(MCPServerTable)
            return session.exec(statement).first()

    @classmethod
    async def get_server_from_tool_name(cls, tool_name):
        with session_getter() as session:
            sql = select(MCPServerTable).where(func.json_contains(MCPServerTable.tools, func.json_array(tool_name)))
            return session.exec(sql).first()

    @classmethod
    async def get_mcp_servers_from_user(cls, user_id):
        with session_getter() as session:
            sql = select(MCPServerTable).where(MCPServerTable.user_id == user_id)
            return session.exec(sql).all()

    @classmethod
    async def get_all_mcp_servers(cls):
        with session_getter() as session:
            sql = select(MCPServerTable)
            return session.exec(sql).all()

    @classmethod
    async def get_mcp_server_ids_from_name(cls, mcp_servers_name, user_id):
        with session_getter() as session:
            sql = select(MCPServerTable).where(
                and_(
                    MCPServerTable.server_name.in_(mcp_servers_name),
                    MCPServerTable.user_id == user_id,
                )
            )
            return session.exec(sql).all()
