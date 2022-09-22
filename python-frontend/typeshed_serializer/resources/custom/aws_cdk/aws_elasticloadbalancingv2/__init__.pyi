from SonarPythonAnalyzerFakeStub import CustomStubBase


class ApplicationLoadBalancer(CustomStubBase):

    def add_listener(self, *args, **kwargs) -> None: ...


class NetworkLoadBalancer(CustomStubBase):

    def add_listener(self, *args, **kwargs) -> None: ...
