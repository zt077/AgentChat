from pydantic import BaseModel, Field

from agentchat.core.models.manager import ModelManager
from agentchat.prompts.rewrite import system_query_rewrite, user_query_write


class QueryRewriteResult(BaseModel):
    variations: list[str] = Field(default_factory=list)


class QueryRewrite:
    def __init__(self):
        self.client = ModelManager.get_conversation_model().with_structured_output(
            QueryRewriteResult,
            method="function_calling",
        )

    async def rewrite(self, user_input):
        rewrite_prompt = user_query_write.format(user_input=user_input)
        response = await self.client.ainvoke(
            [
                ("system", system_query_rewrite),
                ("user", rewrite_prompt),
            ]
        )
        variations = [item.strip() for item in response.variations if item and item.strip()]
        return variations or [user_input]


query_rewriter = QueryRewrite()
