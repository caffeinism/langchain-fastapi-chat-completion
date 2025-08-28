import inspect
from functools import wraps
from typing import Callable

from fastapi import APIRouter
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response

from langchain_fastapi_chat_completion.chat_completion.chat_completion_compatible_api import (
    ChatCompletionCompatibleAPI,
)
from langchain_fastapi_chat_completion.chat_completion.http_stream_response_adapter import (
    HttpStreamResponseAdapter,
)

from ..core.base_agent_factory import BaseAgentFactory
from ..core.create_agent_dto import CreateAgentDto
from ..core.utils.tiny_di_container import TinyDIContainer


def create_chat_completion_router(
    path: str,
    tiny_di_container: TinyDIContainer,
    ainvoke_adapter: Callable = lambda x: x,
    astream_events_adapter: Callable = lambda x: x,
):
    chat_completion_router = APIRouter()
    agent_factory = tiny_di_container.resolve(BaseAgentFactory)

    for key, value in inspect.signature(agent_factory.create_agent).parameters.items():
        if value.annotation is CreateAgentDto:
            break
    else:
        raise ValueError("You must accept CreateAgentDto as an argument.")

    @wraps(agent_factory.create_agent)
    async def _completions(**kwargs) -> JSONResponse:
        dto: CreateAgentDto = kwargs[key]
        dto.request = jsonable_encoder(dto.request)
        agent = agent_factory.create_agent_with_async_context(**kwargs)

        adapter = ChatCompletionCompatibleAPI.from_agent(
            agent,
            dto.request.get("model"),
            ainvoke_adapter=ainvoke_adapter,
            astream_events_adapter=astream_events_adapter,
        )

        response_factory = HttpStreamResponseAdapter()
        if dto.request.get("stream") is True:
            stream = adapter.astream(dto.request.get("messages"))
            return response_factory.to_streaming_response(stream)
        else:
            return JSONResponse(
                content=await adapter.ainvoke(dto.request.get("messages"))
            )

    anns = dict(getattr(_completions, "__annotations__", {}))
    anns["return"] = Response
    _completions.__annotations__ = anns
    sig = inspect.signature(_completions)
    _completions.__signature__ = sig.replace(return_annotation=Response)

    chat_completion_router.post(path)(_completions)

    return chat_completion_router


def create_openai_chat_completion_router(
    tiny_di_container: TinyDIContainer,
    path: str = "",
    ainvoke_adapter: Callable = lambda x: x,
    astream_events_adapter: Callable = lambda x: x,
):
    router = create_chat_completion_router(
        path=path,
        tiny_di_container=tiny_di_container,
        ainvoke_adapter=ainvoke_adapter,
        astream_events_adapter=astream_events_adapter,
    )
    open_ai_router = APIRouter()
    open_ai_router.include_router(router)

    return open_ai_router
