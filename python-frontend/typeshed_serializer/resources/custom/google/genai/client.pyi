from SonarPythonAnalyzerFakeStub import CustomStubBase
from google.genai.models import AsyncModels, Models


class AsyncClient(CustomStubBase):
    models: AsyncModels


class Client(CustomStubBase):
    models: Models
    aio: AsyncClient
