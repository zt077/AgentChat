from typing import Any, Sequence

from langchain_core.messages import BaseMessage, HumanMessage

from agentchat.core.callbacks import usage_metadata_callback
from agentchat.core.models.manager import ModelManager


class StructuredResponseAgent:
    def __init__(self, response_format):
        self.response_format = response_format
        self.structured_model = ModelManager.get_conversation_model().with_structured_output(
            response_format,
            method="function_calling",
        )

    def get_structured_response(self, messages: Sequence[BaseMessage] | str) -> Any:
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]
        return self.structured_model.invoke(
            messages,
            config={"callbacks": [usage_metadata_callback]},
        )
