from SonarPythonAnalyzerFakeStub import CustomStubBase
from typing import Optional

class Resource(CustomStubBase):
    parent_resource: Resource
    api: RestApi

    def add_method(self, *args, **kwargs) -> None: ...
    def get_resource(self, *args, **kwargs) -> Resource: ...
    def add_resource(self, *args, **kwargs) -> Resource: ...
    def add_resource2(self, *args, **kwargs) -> Resource: ...


class RestApi(CustomStubBase):
    root: Resource
