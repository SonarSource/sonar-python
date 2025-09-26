from SonarPythonAnalyzerFakeStub import CustomStubBase
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from typing import List

class Choice(CustomStubBase):
    message: ChatCompletionMessage

class ChatCompletion(CustomStubBase):
    choices: List[Choice]
