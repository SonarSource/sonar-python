from SonarPythonAnalyzerFakeStub import CustomStubBase
from typing import Any, Literal

class CodeAgent(CustomStubBase):
    def __init__(
            self,
            tools: list[Tool],
            model: Model,
            prompt_templates: PromptTemplates | None = None,
            additional_authorized_imports: list[str] | None = None,
            planning_interval: int | None = None,
            executor_type: Literal["local", "e2b", "modal", "docker", "wasm"] = "local",
            executor_kwargs: dict[str, Any] | None = None,
            max_print_outputs_length: int | None = None,
            stream_outputs: bool = False,
            use_structured_outputs_internally: bool = False,
            code_block_tags: str | tuple[str, str] | None = None,
            **kwargs,
    ) -> CodeAgent: ...
