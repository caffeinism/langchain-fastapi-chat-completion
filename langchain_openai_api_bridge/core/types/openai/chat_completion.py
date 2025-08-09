from typing import List, Optional
from pydantic import BaseModel
from openai.types.chat import (
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    ChatCompletionMessageParam,
)
from openai.types.shared.reasoning_effort import ReasoningEffort

class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatCompletionMessageParam]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 1
    stream: Optional[bool] = False
    tools: list[ChatCompletionToolParam] = []
    tool_choice: ChatCompletionToolChoiceOptionParam = "none"
    reasoning_effort: ReasoningEffort | None = None
