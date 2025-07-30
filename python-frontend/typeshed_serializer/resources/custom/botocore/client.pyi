from SonarPythonAnalyzerFakeStub import CustomStubBase
from typing import Any

class BaseClient(CustomStubBase):
    def invoke(self, FunctionName: str, InvocationType: str, Payload: str) -> Any: ...
