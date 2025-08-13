from SonarPythonAnalyzerFakeStub import CustomStubBase
from .query import Query
from typing import Type


class Session(CustomStubBase):
    def query(self, *args, **kwargs) -> Query: ...

# sessionmaker is a class with a __call__ method that returns a Session class 
def sessionmaker(*args, **kwargs) -> Type[Session]: ...
