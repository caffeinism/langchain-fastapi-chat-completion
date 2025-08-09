from typing import Optional

from openai.types.chat import (
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.shared.reasoning_effort import ReasoningEffort
from pydantic import BaseModel


class CreateAgentDto(BaseModel):
    api_key: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    assistant_id: Optional[str] = ""
    thread_id: Optional[str] = ""
    tools: list[ChatCompletionToolParam] = []
    tool_choice: ChatCompletionToolChoiceOptionParam = "none"
    reasoning_effort: ReasoningEffort = None
