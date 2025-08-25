from typing import Annotated, Optional

from fastapi import Depends, Header
from pydantic import BaseModel

from ..fastapi.token_getter import get_bearer_token
from .types.openai import OpenAIChatCompletionRequest


class _CreateAgentDto(BaseModel):
    request: OpenAIChatCompletionRequest
    api_key: Optional[str] = None


async def get_dto(
    chat_completion_request: OpenAIChatCompletionRequest,
    authorization: str = Header(None),
):
    api_key = get_bearer_token(authorization)
    return CreateAgentDto(request=chat_completion_request, api_key=api_key)


CreateAgentDto = Annotated[_CreateAgentDto, Depends(get_dto)]
