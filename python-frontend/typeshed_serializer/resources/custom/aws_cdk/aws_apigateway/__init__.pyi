from SonarPythonAnalyzerFakeStub import CustomStubBase

class Resource(CustomStubBase):
    def add_method(self, *args, **kwargs) -> None: ...

class IResource(CustomStubBase):
    def add_resource(self, *args, **kwargs) -> Resource: ...

class RestApi(CustomStubBase):
    root: IResource
