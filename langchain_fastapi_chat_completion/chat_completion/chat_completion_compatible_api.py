from typing import AsyncContextManager, AsyncIterator, Callable, List, Optional

from langchain_core.runnables import Runnable
from langgraph.graph.state import CompiledStateGraph
from openai.types.chat import ChatCompletionMessage

from langchain_fastapi_chat_completion.chat_completion.langchain_invoke_adapter import (
    LangchainInvokeAdapter,
)
from langchain_fastapi_chat_completion.chat_completion.langchain_stream_adapter import (
    LangchainStreamAdapter,
)
from langchain_fastapi_chat_completion.core.utils.pydantic_async_iterator import (
    ato_dict,
)


class ChatCompletionCompatibleAPI:

    @staticmethod
    def from_agent(
        agent: AsyncContextManager[Runnable],
        llm_model: str,
        system_fingerprint: Optional[str] = "",
        ainvoke_adapter: Callable = lambda x: x,
        astream_events_adapter: Callable = lambda x: x,
    ):
        return ChatCompletionCompatibleAPI(
            LangchainStreamAdapter(llm_model, system_fingerprint),
            LangchainInvokeAdapter(llm_model, system_fingerprint),
            agent,
            ainvoke_adapter,
            astream_events_adapter,
        )

    def __init__(
        self,
        stream_adapter: LangchainStreamAdapter,
        invoke_adapter: LangchainInvokeAdapter,
        agent: AsyncContextManager[Runnable],
        ainvoke_adapter: Callable = lambda x: x,
        astream_events_adapter: Callable = lambda x: x,
    ) -> None:
        self.stream_adapter = stream_adapter
        self.invoke_adapter = invoke_adapter
        self.agent = agent
        self.ainvoke_adapter = ainvoke_adapter
        self.astream_events_adapter = astream_events_adapter

    async def astream(
        self, messages: List[ChatCompletionMessage]
    ) -> AsyncIterator[dict]:
        async with self.agent as runnable:
            input = self.__to_input(runnable, messages)
            astream_event = self.astream_events_adapter(runnable.astream_events)(
                input=input,
                version="v2",
            )
            async for it in ato_dict(
                self.stream_adapter.ato_chat_completion_chunk_stream(astream_event)
            ):
                yield it

    async def ainvoke(self, messages: List[ChatCompletionMessage]) -> dict:
        async with self.agent as runnable:
            input = self.__to_input(runnable, messages)
            result = await self.ainvoke_adapter(runnable.ainvoke)(
                input=input,
            )

        return self.invoke_adapter.to_chat_completion_object(result).model_dump(
            mode="json"
        )

    def __to_input(self, runnable: Runnable, messages: List[ChatCompletionMessage]):
        for message in messages:
            if "content" not in message:
                message["content"] = ""

        if isinstance(runnable, CompiledStateGraph):
            return self.__to_react_agent_input(messages)
        else:
            return self.__to_chat_model_input(messages)

    def __to_react_agent_input(self, messages: List[ChatCompletionMessage]):
        return {
            "messages": [message for message in messages],
        }

    def __to_chat_model_input(self, messages: List[ChatCompletionMessage]):
        return [message for message in messages]
