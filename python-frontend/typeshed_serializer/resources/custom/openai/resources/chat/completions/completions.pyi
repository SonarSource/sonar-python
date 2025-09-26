from SonarPythonAnalyzerFakeStub import CustomStubBase
from openai.types.chat.chat_completion import ChatCompletion

class Completions(CustomStubBase):
    def create(self, *args, **kwargs) -> ChatCompletion: ...
