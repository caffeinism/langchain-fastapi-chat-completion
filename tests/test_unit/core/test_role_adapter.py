from langchain_fastapi_chat_completion.core.role_adapter import to_openai_role


class TestOpenAiRole:
    def test_ai_role_is_assistant(self):
        result = to_openai_role("ai")

        assert result == "assistant"

    def test_user_is_user(self):
        result = to_openai_role("user")

        assert result == "user"

    def test_assistant_is_assistant(self):
        result = to_openai_role("assistant")

        assert result == "assistant"
