from typing import Dict
from unittest.mock import patch

import pytest
from langchain_core.messages import ToolMessage

from langchain_fastapi_chat_completion.chat_completion.langchain_stream_adapter import (
    LangchainStreamAdapter,
)
from tests.stream_utils import assemble_stream, generate_stream
from tests.test_unit.core.agent_stream_utils import (
    create_on_chain_end_event,
    create_on_chat_model_end_event,
    create_on_chat_model_stream_event,
)


class ChatCompletionChunkStub:
    def __init__(self, value: Dict):
        self.dict = lambda: value
        self.choices = []


class TestToChatCompletionChunkStream:
    instance = LangchainStreamAdapter(llm_model="some")

    @pytest.mark.asyncio
    @patch(
        "langchain_fastapi_chat_completion.chat_completion.langchain_stream_adapter.to_openai_chat_completion_chunk_object",
        side_effect=lambda chunk, id, model, system_fingerprint, role, finish_reason: (
            ChatCompletionChunkStub({"key": chunk.content})
        ),
    )
    async def test_stream_contains_every_on_chat_model_stream(
        self, to_openai_chat_completion_chunk_object
    ):
        on_chat_model_stream_event1 = create_on_chat_model_stream_event(content="hello")
        on_chat_model_stream_event2 = create_on_chat_model_stream_event(content="moto")
        on_chat_model_stream_event3 = create_on_chat_model_end_event()
        input_stream = generate_stream(
            [
                on_chat_model_stream_event1,
                on_chat_model_stream_event2,
                on_chat_model_stream_event3,
            ]
        )

        response_stream = self.instance.ato_chat_completion_chunk_stream(input_stream)

        items = await assemble_stream(response_stream)
        assert items[0].dict() == ChatCompletionChunkStub({"key": "hello"}).dict()
        assert items[1].dict() == ChatCompletionChunkStub({"key": "moto"}).dict()

    @pytest.mark.asyncio
    @patch(
        "langchain_fastapi_chat_completion.chat_completion.langchain_stream_adapter.to_openai_chat_completion_chunk_object",
        side_effect=lambda chunk, id, model, system_fingerprint, role, finish_reason: (
            ChatCompletionChunkStub({"key": chunk.content, "role": role})
        ),
    )
    async def test_stream_first_chunk_role(
        self, to_openai_chat_completion_chunk_object
    ):
        on_chat_model_stream_event1 = create_on_chat_model_stream_event(
            content="first chunk"
        )
        on_chat_model_stream_event2 = create_on_chat_model_stream_event(
            content="remain"
        )
        on_chat_model_stream_event3 = create_on_chat_model_end_event()
        input_stream = generate_stream(
            [
                on_chat_model_stream_event1,
                on_chat_model_stream_event2,
                on_chat_model_stream_event3,
            ]
        )

        response_stream = self.instance.ato_chat_completion_chunk_stream(input_stream)

        items = await assemble_stream(response_stream)
        assert items[0].dict()["role"] == "assistant"
        assert items[1].dict()["role"] is None

    @pytest.mark.asyncio
    @patch(
        "langchain_fastapi_chat_completion.chat_completion.langchain_stream_adapter.to_openai_chat_completion_chunk_object",
        side_effect=lambda chunk, id, model, system_fingerprint, role, finish_reason=None: (
            ChatCompletionChunkStub(
                {"key": chunk.content, "role": role, "finish_reason": finish_reason}
            )
        ),
    )
    @patch(
        "langchain_fastapi_chat_completion.chat_completion.langchain_stream_adapter.create_final_chat_completion_chunk_object",
        side_effect=lambda id, model, system_fingerprint=None, finish_reason=None: (
            ChatCompletionChunkStub({"finish_reason": finish_reason})
        ),
    )
    async def test_stream_tool_chunk(
        self,
        to_openai_chat_completion_chunk_object,
        create_final_chat_completion_chunk_object,
    ):
        input_stream = generate_stream(
            [
                create_on_chain_end_event(
                    name="tools",
                    messages=[
                        ToolMessage(tool_call_id="some id", content="some content")
                    ],
                ),
            ]
        )

        response_stream = self.instance.ato_chat_completion_chunk_stream(input_stream)

        items = await assemble_stream(response_stream)

        assert items[0].dict()["role"] == "tool"
        assert items[1].dict()["finish_reason"] == "tool_calls"
