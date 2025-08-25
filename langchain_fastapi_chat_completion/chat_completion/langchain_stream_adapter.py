import uuid
from typing import AsyncIterator

from langchain_core.messages import AIMessageChunk, ToolMessage
from langchain_core.runnables.schema import StreamEvent
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from langchain_fastapi_chat_completion.chat_completion.chat_completion_chunk_choice_adapter import (
    to_openai_chat_completion_chunk_object,
)
from langchain_fastapi_chat_completion.chat_completion.chat_completion_chunk_object_factory import (
    create_final_chat_completion_chunk_object,
)


class NotGiven:
    def __bool__(self):
        return False

    def __repr__(self) -> str:
        return "NOT_GIVEN"


NOT_GIVEN = NotGiven()


class LangchainStreamAdapter:
    def __init__(self, llm_model: str, system_fingerprint: str = ""):
        self.llm_model = llm_model
        self.system_fingerprint = system_fingerprint

    async def ato_chat_completion_chunk_stream(
        self,
        astream_event: AsyncIterator[StreamEvent],
        id: str = "",
    ) -> AsyncIterator[ChatCompletionChunk]:
        if id == "":
            id = str(uuid.uuid4())

        is_tool_call = False
        role = NOT_GIVEN
        async for event in astream_event:
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk: AIMessageChunk = event["data"]["chunk"]
                if role is NOT_GIVEN:
                    role = "assistant"

                chat_completion_chunk = to_openai_chat_completion_chunk_object(
                    chunk=chunk,
                    id=id,
                    model=self.llm_model,
                    system_fingerprint=self.system_fingerprint,
                    role=role,
                    finish_reason=None,
                )
                role = None
                yield chat_completion_chunk
                is_tool_call = is_tool_call or any(
                    choice.delta.tool_calls for choice in chat_completion_chunk.choices
                )
            elif kind == "on_chat_model_end":
                role = NOT_GIVEN
                yield create_final_chat_completion_chunk_object(
                    id=id,
                    model=self.llm_model,
                    finish_reason="tool_calls" if is_tool_call else "stop",
                )
                is_tool_call = False
            elif kind == "on_chain_end":
                if event["name"] == "tools":
                    it: ToolMessage
                    for it in event["data"]["output"]["messages"]:
                        chat_completion_chunk = to_openai_chat_completion_chunk_object(
                            chunk=it,
                            id=id,
                            model=self.llm_model,
                            system_fingerprint=self.system_fingerprint,
                            role="tool",
                        )
                        yield chat_completion_chunk
                    yield create_final_chat_completion_chunk_object(
                        id=id,
                        model=self.llm_model,
                        finish_reason="tool_calls",
                    )
