from SonarPythonAnalyzerFakeStub import CustomStubBase

class Port(CustomStubBase):
    def all_tcp(self) -> Port: ...

class Connections(CustomStubBase):
    def allow_from(self, other, port_range:Port, description) -> None: ...

## All constructs with connections attributes
class BastionHostLinux(CustomStubBase):
    connections: Connections

class ClientVpnEndpoint(CustomStubBase):
    connections: Connections

class Instance(CustomStubBase):
    connections: Connections

class LaunchTemplate(CustomStubBase):
    connections: Connections

class SecurityGroup(CustomStubBase):
    connections: Connections
