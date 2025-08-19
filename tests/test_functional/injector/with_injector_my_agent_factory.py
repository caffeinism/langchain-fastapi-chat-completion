from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from langchain_fastapi_chat_completion.core.base_agent_factory import BaseAgentFactory
from langchain_fastapi_chat_completion.core.create_agent_dto import CreateAgentDto


@tool
def magic_number_tool(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


class WithInjectorMyAgentFactory(BaseAgentFactory):

    def create_agent(self, dto: CreateAgentDto) -> Runnable:
        llm = ChatOpenAI(
            model=dto.request.model,
            api_key=dto.api_key,
            streaming=True,
            temperature=dto.request.temperature,
        )

        return create_react_agent(
            llm,
            [magic_number_tool],
            prompt="""You are a helpful assistant.""",
        )
