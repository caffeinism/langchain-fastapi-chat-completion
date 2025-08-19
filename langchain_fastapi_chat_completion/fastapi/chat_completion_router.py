import inspect
from functools import wraps

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from langchain_fastapi_chat_completion.chat_completion.chat_completion_compatible_api import (
    ChatCompletionCompatibleAPI,
)
from langchain_fastapi_chat_completion.chat_completion.http_stream_response_adapter import (
    HttpStreamResponseAdapter,
)
from langchain_fastapi_chat_completion.core.base_agent_factory import BaseAgentFactory
from langchain_fastapi_chat_completion.core.create_agent_dto import CreateAgentDto
from langchain_fastapi_chat_completion.core.utils.tiny_di_container import (
    TinyDIContainer,
)


def create_chat_completion_router(
    path: str,
    tiny_di_container: TinyDIContainer,
    event_adapter: callable = lambda event: None,
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

        agent = agent_factory.create_agent_with_async_context(**kwargs)

        adapter = ChatCompletionCompatibleAPI.from_agent(
            agent, dto.request.model, event_adapter=event_adapter
        )

        response_factory = HttpStreamResponseAdapter()
        if dto.request.stream is True:
            stream = adapter.astream(dto.request.messages)
            return response_factory.to_streaming_response(stream)
        else:
            return JSONResponse(content=await adapter.ainvoke(dto.request.messages))

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
    event_adapter: callable = lambda event: None,
):
    router = create_chat_completion_router(
        path=path, tiny_di_container=tiny_di_container, event_adapter=event_adapter
    )
    open_ai_router = APIRouter()
    open_ai_router.include_router(router)

    return open_ai_router
