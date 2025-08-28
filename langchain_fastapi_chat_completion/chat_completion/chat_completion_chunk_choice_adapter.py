from typing import Optional
from uuid import uuid4

from langchain_core.messages import BaseMessageChunk
from langchain_core.runnables.schema import StreamEvent
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    ChoiceDeltaToolCall,
    ChoiceDeltaToolCallFunction,
)

from langchain_fastapi_chat_completion.chat_completion.chat_completion_chunk_object_factory import (
    create_chat_completion_chunk_object,
)


def to_openai_chat_message(
    chunk: BaseMessageChunk,
    role: str | None = None,
) -> ChoiceDelta:
    if getattr(chunk, "tool_call_chunks", None):
        tool_calls = [
            ChoiceDeltaToolCall(
                id=tool_call_chunk["id"] or str(uuid4()),
                index=tool_call_chunk["index"] or 0,
                function=ChoiceDeltaToolCallFunction(
                    arguments=tool_call_chunk["args"],
                    name=tool_call_chunk["name"],
                ),
                type="function",
            )
            for tool_call_chunk in chunk.tool_call_chunks
        ]
    elif getattr(chunk, "tool_call_id", None):
        tool_calls = [ChoiceDeltaToolCall(id=chunk.tool_call_id, index=0)]
    else:
        tool_calls = None

    return ChoiceDelta(
        content=chunk.content,
        role=role,
        tool_calls=tool_calls,
    )


def to_openai_chat_completion_chunk_choice(
    chunk: BaseMessageChunk,
    index: int = 0,
    role: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> Choice:
    message = to_openai_chat_message(chunk, role)

    return Choice(
        index=index,
        delta=message,
        finish_reason=finish_reason,
    )


def to_openai_chat_completion_chunk_object(
    chunk: BaseMessageChunk,
    id: str = "",
    model: str = "",
    system_fingerprint: Optional[str] = None,
    role: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> ChatCompletionChunk:

    choice1 = to_openai_chat_completion_chunk_choice(
        chunk, index=0, role=role, finish_reason=finish_reason
    )

    return create_chat_completion_chunk_object(
        id=id,
        model=model,
        system_fingerprint=system_fingerprint,
        choices=[choice1],
    )
