from SonarPythonAnalyzerFakeStub import CustomStubBase
from aiobotocore.client import AioBaseClient

class AioSession(CustomStubBase):
    def create_client(self, *args, **kwargs) -> AioBaseClient: ...

def get_session(*args, **kwargs) -> AioSession: ...

