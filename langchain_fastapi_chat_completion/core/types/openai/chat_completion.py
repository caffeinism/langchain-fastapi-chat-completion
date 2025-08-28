from typing import Dict, Iterable, List, Literal, Optional, Union

from openai.types.chat import (
    ChatCompletionAudioParam,
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
    completion_create_params,
)
from openai.types.chat.chat_completion_audio_param import ChatCompletionAudioParam
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_prediction_content_param import (
    ChatCompletionPredictionContentParam,
)
from openai.types.chat.chat_completion_stream_options_param import (
    ChatCompletionStreamOptionsParam,
)
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.shared.chat_model import ChatModel
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.metadata import Metadata
from pydantic import BaseModel


class OpenAIChatCompletionRequest(BaseModel):
    messages: list[ChatCompletionMessageParam]
    model: Union[str, ChatModel]
    audio: ChatCompletionAudioParam | None = None
    frequency_penalty: float | None = None
    function_call: completion_create_params.FunctionCall | None = None
    functions: list[completion_create_params.Function] | None = None
    logit_bias: Dict[str, int] | None = None
    logprobs: bool | None = None
    max_completion_tokens: int | None = None
    max_tokens: int | None = None
    metadata: Metadata | None = None
    modalities: List[Literal["text", "audio"]] | None = None
    n: int | None = None
    parallel_tool_calls: bool | None = None
    prediction: ChatCompletionPredictionContentParam | None = None
    presence_penalty: float | None = None
    prompt_cache_key: str | None = None
    reasoning_effort: ReasoningEffort | None = None
    response_format: completion_create_params.ResponseFormat | None = None
    safety_identifier: str | None = None
    seed: int | None = None
    service_tier: Literal["auto", "default", "flex", "scale", "priority"] | None = None
    stop: Union[str] | List[str] | None = None
    store: bool | None = None
    stream: bool | None = None
    stream_options: ChatCompletionStreamOptionsParam | None = None
    temperature: float | None = None
    tool_choice: ChatCompletionToolChoiceOptionParam | None = None
    tools: list[ChatCompletionToolParam] | None = None
    top_logprobs: int | None = None
    top_p: float | None = None
    user: str | None = None
    verbosity: Literal["low", "medium", "high"] | None = None
    web_search_options: completion_create_params.WebSearchOptions | None = None
