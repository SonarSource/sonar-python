from SonarPythonAnalyzerFakeStub import CustomStubBase


class ApplicationListener(CustomStubBase): ...


class NetworkListener(CustomStubBase): ...


class ApplicationLoadBalancer(CustomStubBase):

    def add_listener(self, *args, **kwargs) -> ApplicationListener: ...


class NetworkLoadBalancer(CustomStubBase):

    def add_listener(self, *args, **kwargs) -> NetworkListener: ...
