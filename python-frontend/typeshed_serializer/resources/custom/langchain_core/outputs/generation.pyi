from langchain_core.messages.base import BaseMessage

from SonarPythonAnalyzerFakeStub import CustomStubBase


class Generation(CustomStubBase):
    text: str
    message: BaseMessage
