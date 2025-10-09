from SonarPythonAnalyzerFakeStub import CustomStubBase

from anthropic.resources.beta.messages.messages import Messages, AsyncMessages

class Beta(CustomStubBase):
    messages: Messages


class AsyncBeta(CustomStubBase):
    messages: AsyncMessages

