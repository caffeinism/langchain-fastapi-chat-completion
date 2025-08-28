import uvicorn
from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import ChatOpenAI

from langchain_fastapi_chat_completion.core.create_agent_dto import CreateAgentDto
from langchain_fastapi_chat_completion.fastapi.langchain_openai_api_bridge_fastapi import (
    LangchainOpenaiApiBridgeFastAPI,
)

_ = load_dotenv(find_dotenv())


app = FastAPI(
    title="Langchain Agent OpenAI API Bridge",
    version="1.0",
    description="OpenAI API exposing langchain agent",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


def create_agent(dto: CreateAgentDto):
    return ChatOpenAI(
        temperature=dto.request.get("temperature") or 0.7,
        model=dto.request.get("model"),
        max_tokens=dto.request.get("max_tokens"),
        api_key=dto.api_key,
    )


bridge = LangchainOpenaiApiBridgeFastAPI(app=app, agent_factory_provider=create_agent)


def astream_events_adapter(astream_events):
    async def _astream_events(
        input,
        config=None,
        **kwargs,
    ):
        async for event in astream_events(input, config, **kwargs):
            yield event

    return _astream_events


bridge.bind_openai_chat_completion(
    path="/my-custom-events-path/chat/completions",
    astream_events_adapter=astream_events_adapter,
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost")
