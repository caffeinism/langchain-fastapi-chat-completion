from typing import Annotated, Mapping, Optional

from fastapi import Body, Depends, Header, Query, Request
from pydantic import BaseModel
from starlette.datastructures import Headers, QueryParams

from ..fastapi.token_getter import get_bearer_token
from .types.openai import OpenAIChatCompletionRequest


class _CreateAgentDto(BaseModel):
    request: OpenAIChatCompletionRequest
    api_key: Optional[str] = None
    extra_body: dict
    extra_headers: Mapping[str, str]
    extra_query: Mapping[str, str]


def clean_header(headers: Mapping[str, str]):
    del headers["connection"]
    del headers["keep-alive"]
    del headers["proxy-authenticate"]
    del headers["proxy-authorization"]
    del headers["te"]
    del headers["trailer"]
    del headers["transfer-encoding"]
    del headers["upgrade"]
    del headers["proxy-connection"]
    del headers["host"]
    del headers["accept"]
    del headers["accept-encoding"]
    del headers["content-length"]
    del headers["content-type"]
    del headers["content-encoding"]
    del headers["user-agent"]
    del headers["transfer-encoding"]
    del headers["expect"]
    del headers["forwarded"]
    del headers["x-forwarded-for"]
    del headers["x-forwarded-proto"]
    del headers["x-forwarded-host"]
    del headers["via"]
    del headers["authorization"]
    del headers["cookie"]
    del headers["access-control-request-method"]
    del headers["access-control-request-headers"]
    del headers["origin"]
    return headers


def clean_body(body: dict, chat_completion_request: OpenAIChatCompletionRequest):
    field_names = set(chat_completion_request.model_fields.keys())
    return {k: v for k, v in body.items() if k not in field_names}


async def get_dto(
    request: Request,
    chat_completion_request: OpenAIChatCompletionRequest,
    authorization: str = Header(None),
):
    raw_body = clean_body(await request.json(), chat_completion_request)
    raw_headers = clean_header(request.headers.mutablecopy())
    raw_query = request.query_params

    api_key = get_bearer_token(authorization)
    return CreateAgentDto(
        request=chat_completion_request,
        api_key=api_key,
        extra_body=raw_body,
        extra_headers=raw_headers,
        extra_query=raw_query,
    )


CreateAgentDto = Annotated[_CreateAgentDto, Depends(get_dto)]
