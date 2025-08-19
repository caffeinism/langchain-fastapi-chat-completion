from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import Runnable
from langgraph.prebuilt import create_react_agent

from langchain_fastapi_chat_completion.core.base_agent_factory import BaseAgentFactory
from langchain_fastapi_chat_completion.core.create_agent_dto import CreateAgentDto


class MyAnthropicAgentFactory(BaseAgentFactory):

    def create_agent(self, dto: CreateAgentDto) -> Runnable:
        llm = ChatAnthropic(
            model=dto.request.model,
            streaming=True,
        )

        return create_react_agent(
            llm,
            [],
            prompt="""You are a helpful assistant.""",
        )
