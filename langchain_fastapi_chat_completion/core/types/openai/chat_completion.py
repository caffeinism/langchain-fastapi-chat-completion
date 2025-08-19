from typing import List, Optional

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.shared.reasoning_effort import ReasoningEffort
from pydantic import BaseModel


class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionMessageParam]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 1
    stream: Optional[bool] = False
    tools: list[ChatCompletionToolParam] = []
    tool_choice: ChatCompletionToolChoiceOptionParam = "none"
    reasoning_effort: ReasoningEffort | None = None
