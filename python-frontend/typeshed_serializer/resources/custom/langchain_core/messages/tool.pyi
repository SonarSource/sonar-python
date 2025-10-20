from typing import Any

from SonarPythonAnalyzerFakeStub import CustomStubBase


class ToolCall(CustomStubBase):
    name: str
    args: dict[str, Any]
