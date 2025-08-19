from typing import Callable, Optional, Union

from fastapi import FastAPI
from langchain_core.runnables import Runnable

from langchain_fastapi_chat_completion.core.base_agent_factory import BaseAgentFactory
from langchain_fastapi_chat_completion.core.create_agent_dto import CreateAgentDto
from langchain_fastapi_chat_completion.core.langchain_openai_api_bridge import (
    LangchainOpenaiApiBridge,
)
from langchain_fastapi_chat_completion.fastapi.chat_completion_router import (
    create_openai_chat_completion_router,
)


class LangchainOpenaiApiBridgeFastAPI(LangchainOpenaiApiBridge):
    def __init__(
        self,
        app: FastAPI,
        agent_factory_provider: Union[
            Callable[[], BaseAgentFactory],
            Callable[[CreateAgentDto], Runnable],
            BaseAgentFactory,
        ],
    ) -> None:
        super().__init__(agent_factory_provider=agent_factory_provider)
        self.app = app

    def bind_openai_chat_completion(
        self,
        path: str,
        ainvoke_adapter: Callable = lambda x: x,
        astream_events_adapter: Callable = lambda x: x,
    ) -> None:
        chat_completion_router = create_openai_chat_completion_router(
            path=path,
            tiny_di_container=self.tiny_di_container,
            ainvoke_adapter=ainvoke_adapter,
            astream_events_adapter=astream_events_adapter,
        )

        self.app.include_router(chat_completion_router)
