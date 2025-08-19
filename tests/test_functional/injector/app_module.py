from injector import Binder, Module, singleton

from langchain_fastapi_chat_completion.assistant import (
    InMemoryMessageRepository,
    InMemoryRunRepository,
    InMemoryThreadRepository,
    MessageRepository,
    RunRepository,
    ThreadRepository,
)
from langchain_fastapi_chat_completion.core import BaseAgentFactory
from tests.test_functional.injector.with_injector_my_agent_factory import (
    WithInjectorMyAgentFactory,
)


class MyAppModule(Module):
    def configure(self, binder: Binder):
        binder.bind(ThreadRepository, to=InMemoryThreadRepository, scope=singleton)
        binder.bind(MessageRepository, to=InMemoryMessageRepository, scope=singleton)
        binder.bind(RunRepository, to=InMemoryRunRepository, scope=singleton)
        binder.bind(BaseAgentFactory, to=WithInjectorMyAgentFactory)
