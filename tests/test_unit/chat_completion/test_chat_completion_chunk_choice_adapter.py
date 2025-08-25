from unittest.mock import patch

from langchain_core.messages import AIMessageChunk, ToolMessage

from langchain_fastapi_chat_completion.chat_completion.chat_completion_chunk_choice_adapter import (
    to_openai_chat_completion_chunk_choice,
    to_openai_chat_completion_chunk_object,
    to_openai_chat_message,
)


class FixtureEventChunk:
    def __init__(self, content: str, tool_call_chunks: list = []):
        self.content = content
        self.tool_call_chunks = tool_call_chunks


class TestToChatMessage:
    def test_message_have_content(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_message(chunk)

        assert result.content == "some content"

    def test_message_have_args_role(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_message(chunk, role="assistant")

        assert result.role == "assistant"

    def test_message_have_no_role_by_default(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_message(chunk)

        assert result.role is None

    def test_message_have_tool_role(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_message(chunk, role="tool")

        assert result.role == "tool"

    def test_tool_message_have_tool_calls(self):
        chunk = ToolMessage(content="some content", tool_call_id="some id")

        result = to_openai_chat_message(chunk, role="tool")

        assert result.role == "tool"
        assert result.content == "some content"
        assert result.tool_calls[0].id == "some id"

        assert result.role == "tool"


class TestToCompletionChunkChoice:
    def test_choice_finish_reason_is_none_by_default(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_completion_chunk_choice(chunk)

        assert result.finish_reason is None

    def test_choice_finish_reason_is_same_as_provided(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_completion_chunk_choice(chunk, finish_reason="stop")

        assert result.finish_reason == "stop"

    def test_choice_index_0_by_default(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_completion_chunk_choice(chunk)

        assert result.index == 0

    def test_choice_index_is_same_as_provided(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_completion_chunk_choice(chunk, index=69)

        assert result.index == 69

    def test_delta_message_have_content(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_completion_chunk_choice(chunk)

        assert result.delta.content == "some content"

    def test_delta_message_have_args_role(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_completion_chunk_choice(chunk, role="assistant")

        assert result.delta.role == "assistant"

    def test_delta_message_have_none_role_by_default(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_completion_chunk_choice(chunk)

        assert result.delta.role is None


class TestToCompletionChunkObject:
    def test_id_is_same_as_provider(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_completion_chunk_object(chunk, id="a")

        assert result.id == "a"

    def test_id_is_empty_string_when_not_defined_as_argument(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_completion_chunk_object(chunk)

        assert result.id == ""

    def test_id_empty_string_when_not_defined(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_completion_chunk_object(chunk)

        assert result.id == ""

    def test_object_is_always_chat_completion_chunk(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_completion_chunk_object(chunk)

        assert result.object == "chat.completion.chunk"

    @patch("time.time", return_value=1638316800)
    def test_created_is_current_time(self, mock_time):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_completion_chunk_object(chunk)

        assert result.created == 1638316800

    def test_choice_finish_reason_is_none_by_default(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_completion_chunk_object(chunk)

        assert result.choices[0].finish_reason is None

    def test_choice_finish_reason_is_same_as_provided(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_completion_chunk_object(chunk, finish_reason="stop")

        assert result.choices[0].finish_reason == "stop"

    def test_choice_index_0_by_default(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_completion_chunk_object(chunk)

        assert result.choices[0].index == 0

    def test_delta_message_have_content(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_completion_chunk_object(chunk)

        assert result.choices[0].delta.content == "some content"

    def test_delta_message_have_args_role(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_completion_chunk_object(chunk, role="assistant")

        assert result.choices[0].delta.role == "assistant"

    def test_delta_message_have_none_role_by_default(self):
        chunk = AIMessageChunk(content="some content")

        result = to_openai_chat_completion_chunk_object(chunk)

        assert result.choices[0].delta.role is None

    def test_delta_message_have_tool_calls(self):
        chunk = AIMessageChunk(
            content="some content",
            tool_call_chunks=[
                {
                    "name": "my_func",
                    "args": '{"x": 1}',
                    "id": "some id",
                    "index": 0,
                    "type": "tool_call_chunk",
                }
            ],
        )

        result = to_openai_chat_completion_chunk_object(chunk)

        assert result.choices[0].delta.tool_calls is not None
        assert result.choices[0].delta.tool_calls[0].function.name == "my_func"
        assert result.choices[0].delta.tool_calls[0].function.arguments == '{"x": 1}'
        assert result.choices[0].delta.tool_calls[0].id == "some id"
