from SonarPythonAnalyzerFakeStub import CustomStubBase

from .query import Query

class Session(CustomStubBase):
    def query(self, *args, **kwargs) -> Query: ...
