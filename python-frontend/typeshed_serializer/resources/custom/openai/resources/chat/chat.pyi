from SonarPythonAnalyzerFakeStub import CustomStubBase
from openai.resources.chat.completions.completions import Completions

class Chat(CustomStubBase):
    completions: Completions
