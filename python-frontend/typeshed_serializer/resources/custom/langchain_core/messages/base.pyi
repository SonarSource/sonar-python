from typing import Union

from langchain_core.messages.tool import ToolCall

from SonarPythonAnalyzerFakeStub import CustomStubBase


class BaseMessage(CustomStubBase):
    content: Union[str, list[Union[str, dict]]]
    tool_calls: list[ToolCall] = []

    def text(self)-> str: ...
