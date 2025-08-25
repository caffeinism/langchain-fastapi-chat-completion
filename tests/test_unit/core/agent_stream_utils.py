from langchain_core.messages import BaseMessage
from langchain_core.runnables.schema import EventData, StandardStreamEvent


class ChunkStub:
    def __init__(self, content: str, tool_call_chunks: list = []):
        self.content = content
        self.tool_call_chunks = tool_call_chunks


def create_stream_output_event(
    content: str = "", run_id: str = "", name: str = "", event: str = ""
):
    event_data = EventData(output=ChunkStub(content=content))
    return StandardStreamEvent(run_id=run_id, event=event, name=name, data=event_data)


def create_on_chat_model_stream_event(
    content: str = "", name: str = "", run_id: str = ""
):
    return StandardStreamEvent(
        run_id=run_id,
        event="on_chat_model_stream",
        name=name,
        data=EventData(chunk=ChunkStub(content=content)),
    )


def create_on_chat_model_end_event(run_id: str = ""):
    return StandardStreamEvent(
        run_id=run_id,
        event="on_chat_model_end",
        name="",
        data=EventData(output=ChunkStub(content="")),
    )


def create_on_chain_end_event(name: str, messages: list[BaseMessage], run_id: str = ""):
    return StandardStreamEvent(
        run_id=run_id,
        event="on_chain_end",
        name=name,
        data=EventData(output={"messages": messages}),
    )
