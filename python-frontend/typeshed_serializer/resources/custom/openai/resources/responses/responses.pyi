from SonarPythonAnalyzerFakeStub import CustomStubBase
from openai.types.responses.response import Response

class Responses(CustomStubBase):
    def create(self, *args, **kwargs) -> Response: ...
