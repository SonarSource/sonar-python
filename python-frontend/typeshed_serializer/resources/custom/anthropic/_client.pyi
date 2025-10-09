from SonarPythonAnalyzerFakeStub import CustomStubBase

from anthropic.resources.messages.messages import Messages, AsyncMessages
from anthropic.resources.beta.beta import Beta, AsyncBeta

class Anthropic(CustomStubBase):
    messages: Messages
    beta: Beta


class AsyncAnthropic(CustomStubBase):
    messages: AsyncMessages
    beta: AsyncBeta


class Client(CustomStubBase):
    messages: Messages
    beta: Beta


class AsyncClient(CustomStubBase):
    messages: AsyncMessages
    beta: AsyncBeta
