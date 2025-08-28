from injector import Binder, Module, singleton

from langchain_fastapi_chat_completion.core import BaseAgentFactory
from tests.test_functional.injector.with_injector_my_agent_factory import (
    WithInjectorMyAgentFactory,
)


class MyAppModule(Module):
    def configure(self, binder: Binder):
        binder.bind(BaseAgentFactory, to=WithInjectorMyAgentFactory)
