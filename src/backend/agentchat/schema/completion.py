from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class CompletionReq(BaseModel):
    user_input: Optional[str] = Field(default=None, description="User input")
    dialog_id: str = Field(description="Dialog id")
    file_url: Optional[str] = Field(default=None, description="Uploaded file url")
    run_id: Optional[str] = Field(default=None, description="Durable execution run id")
    resume: bool = Field(default=False, description="Whether to resume an interrupted run")
    approved_tools: List[str] = Field(default_factory=list, description="Tools approved for this run")

    @model_validator(mode="after")
    def validate_request(self):
        if self.resume:
            if not self.run_id:
                raise ValueError("run_id is required when resume=true")
        elif not self.user_input:
            raise ValueError("user_input is required for a new completion run")
        return self


class ToolCall(BaseModel):
    tool_name: str = Field(..., description="Tool name")
    tool_args: Any = Field(..., description="Tool args")
    message: str = Field(..., description="Reason for this tool call")


StepTools = List[ToolCall]


class PlanToolFlow(BaseModel):
    root: Dict[str, StepTools] = Field(
        ...,
        description="Planned tool call flow keyed by step name",
    )

    def dict(self, **kwargs) -> Dict[str, Any]:
        return super().dict(**kwargs)

    def model_dump(self, **kwargs):
        return super().model_dump(**kwargs)
