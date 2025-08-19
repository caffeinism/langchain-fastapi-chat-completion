import inspect
from typing import Callable, Optional, Union

from langchain_core.runnables import Runnable

from langchain_fastapi_chat_completion.core.base_agent_factory import BaseAgentFactory
from langchain_fastapi_chat_completion.core.create_agent_dto import CreateAgentDto
from langchain_fastapi_chat_completion.core.function_agent_factory import (
    FunctionAgentFactory,
)
from langchain_fastapi_chat_completion.core.utils import TinyDIContainer


class LangchainOpenaiApiBridge:
    def __init__(
        self,
        agent_factory_provider: Union[
            Callable[[], BaseAgentFactory],
            Callable[[CreateAgentDto], Runnable],
            BaseAgentFactory,
        ],
        tiny_di_container: Optional[TinyDIContainer] = None,
    ) -> None:
        self.tiny_di_container = tiny_di_container or TinyDIContainer()

        if self.__is_callable_runnable_provider(agent_factory_provider):
            self.tiny_di_container.register(
                BaseAgentFactory, to=FunctionAgentFactory(fn=agent_factory_provider)
            )
        else:
            self.tiny_di_container.register(BaseAgentFactory, to=agent_factory_provider)

    @staticmethod
    def __is_callable_runnable_provider(
        agent_factory_provider: Union[
            Callable[[], BaseAgentFactory],
            Callable[[CreateAgentDto], Runnable],
            BaseAgentFactory,
        ],
    ):
        if not callable(agent_factory_provider):
            return False

        try:
            signature = inspect.signature(agent_factory_provider)
            params = list(signature.parameters.values())
            if len(params) > 0:
                return True
        except Exception:
            return False

        return False
