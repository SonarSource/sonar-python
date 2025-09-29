from SonarPythonAnalyzerFakeStub import CustomStubBase
from typing import Any, Literal, Callable

class UserProxyAgent(CustomStubBase):
    def __init__(
            self,
            name: str,
            is_termination_msg: Callable[[dict[str, Any]], bool] | None = None,
            max_consecutive_auto_reply: int | None = None,
            human_input_mode: Literal["ALWAYS", "TERMINATE", "NEVER"] = "ALWAYS",
            function_map: dict[str, Callable[..., Any]] | None = None,
            code_execution_config: dict[str, Any] | Literal[False] = {},
            default_auto_reply: str | dict[str, Any] | None = "",
            llm_config: LLMConfig | dict[str, Any] | Literal[False] | None = False,
            system_message: str | list[str] | None = "",
            description: str | None = None,
            **kwargs: Any,
    ) -> UserProxyAgent: ...
