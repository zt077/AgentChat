from fastapi import APIRouter, Depends, HTTPException

from agentchat.api.services.observability import ObservabilityService
from agentchat.api.services.user import UserPayload, get_login_user
from agentchat.schema.schemas import resp_200

router = APIRouter(prefix="/observability", tags=["Observability"])


@router.get("/runs")
async def list_runs(
    dialog_id: str | None = None,
    status: str | None = None,
    limit: int = 20,
    login_user: UserPayload = Depends(get_login_user),
):
    results = await ObservabilityService.list_runs(
        user_id=login_user.user_id,
        dialog_id=dialog_id,
        status=status,
        limit=limit,
    )
    return resp_200(data=results)


@router.get("/runs/{run_id}")
async def get_run_detail(run_id: str, login_user: UserPayload = Depends(get_login_user)):
    result = await ObservabilityService.get_run_detail(run_id=run_id, user_id=login_user.user_id)
    if result is None:
        raise HTTPException(status_code=404, detail="run not found")
    return resp_200(data=result)
