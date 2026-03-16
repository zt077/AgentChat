from typing import Dict, List, Optional, Union

from loguru import logger
from langchain.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from agentchat.core.models.manager import ModelManager
from agentchat.database import AgentSkill
from agentchat.schema.agent_skill import AgentSkillFile, AgentSkillFolder


class SkillAgent:
    def __init__(self, skill: AgentSkill, user_id: str):
        self.skill = skill
        self.user_id = user_id
        self.skill_folder: Optional[AgentSkillFolder] = None
        self.file_cache: Dict[str, AgentSkillFile] = {}

        self.conversation_model = None
        self.tools = None
        self.react_agent = None
        self._initialized = False

    async def init_skill_agent(self):
        try:
            self.load_skill_folder(self.skill.folder)
            self.setup_language_model()
            self.tools = self.setup_skill_agent_tools()
            self.react_agent = self.setup_react_agent()
            self._initialized = True
        except Exception as err:
            logger.error(f"SkillAgent init failed: {err}")
            raise

    def setup_react_agent(self):
        skill_md = self.get_skill_md() or f"name: {self.skill.name}\ndescription: {self.skill.description or ''}"
        return create_react_agent(
            model=self.conversation_model,
            tools=self.tools,
            prompt=self._build_system_prompt(skill_md),
        )

    def setup_language_model(self):
        self.conversation_model = ModelManager.get_conversation_model()

    def _build_system_prompt(self, skill_md: str) -> str:
        return (
            f"You are the dedicated skill agent `{self.skill.name}`.\n\n"
            f"## Skill document\n{skill_md}\n\n"
            "You may use tools to inspect skill files when needed."
        )

    def load_skill_folder(self, json_data: dict) -> AgentSkillFolder:
        def parse_item(item_data: dict) -> Union[AgentSkillFile, AgentSkillFolder]:
            item_type = item_data.get("type", "file")
            if item_type == "file":
                file_obj = AgentSkillFile(
                    name=item_data["name"],
                    path=item_data["path"],
                    type=item_data["type"],
                    content=item_data.get("content", ""),
                )
                self.file_cache[file_obj.path] = file_obj
                return file_obj

            folder_items = [parse_item(sub_item) for sub_item in item_data.get("folder", [])]
            return AgentSkillFolder(
                name=item_data["name"],
                path=item_data["path"],
                type=item_data["type"],
                folder=folder_items,
            )

        self.skill_folder = parse_item(json_data)
        return self.skill_folder

    def get_file_content(self, path: str) -> Optional[str]:
        file_obj = self.file_cache.get(path)
        return file_obj.content if file_obj else None

    def list_files(self, pattern: str = None) -> List[str]:
        if pattern:
            return [path for path in self.file_cache.keys() if pattern in path]
        return list(self.file_cache.keys())

    def get_skill_md(self) -> Optional[str]:
        if not self.skill_folder:
            return None
        for item in self.skill_folder.folder:
            if isinstance(item, AgentSkillFile) and item.name == "SKILL.md":
                return item.content
        return None

    def setup_skill_agent_tools(self):
        @tool(parse_docstring=True)
        def get_file_content(file_path: str) -> str:
            """
            Read a file inside the skill folder.

            Args:
                file_path: target file path

            Returns:
                File content or an error message
            """

            content = self.get_file_content(file_path)
            if content is None:
                return "File not found.\nAvailable files:\n" + "\n".join(self.list_files())
            return content

        @tool(parse_docstring=True)
        def list_skill_files(pattern: str = None) -> str:
            """
            List files inside the skill folder.

            Args:
                pattern: optional substring filter

            Returns:
                Matched file paths
            """

            files = self.list_files(pattern)
            return "\n".join(files) if files else "No files found"

        return [get_file_content, list_skill_files]

    async def ainvoke(self, messages: List[BaseMessage]) -> List[BaseMessage] | str:
        if not self._initialized:
            await self.init_skill_agent()

        result = await self.react_agent.ainvoke({"messages": messages})
        return [
            message
            for message in result["messages"]
            if not isinstance(message, (HumanMessage, SystemMessage))
        ]
