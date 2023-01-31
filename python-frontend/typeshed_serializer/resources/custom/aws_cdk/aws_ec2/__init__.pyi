from SonarPythonAnalyzerFakeStub import CustomStubBase

class Port(CustomStubBase):
    def all_tcp(self, *args, **kwargs) -> Port: ...

class Connections(CustomStubBase):
    def allow_from(self, other, port_range:Port, description, *args, **kwargs) -> None: ...
    def allow_from_any_ipv4(self, port_range:Port, description, *args, **kwargs) -> None: ...
    def allow_default_port_from(self, other, description, *args, **kwargs) -> None: ...
    def allow_default_port_from_any_ipv4(self, description, *args, **kwargs) -> None: ...
    def add_resource2(self, *args, **kwargs) -> None: ...

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
    def add_ingress_rule(self, peer, connection:Port, description, remote_rule, *args, **kwargs) -> None: ...

class Vpc(CustomStubBase):
    def select_subnets(self, availability_zones, one_per_az, subnet_filters, subnet_group_name, subnet_name, subnets, subnet_type, *args, **kwargs) -> SelectedSubnets: ...

class SelectedSubnets(CustomStubBase): ...
