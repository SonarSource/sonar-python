from SonarPythonAnalyzerFakeStub import CustomStubBase
from aws_cdk.aws_ec2 import Connections

## constructs with connections attributes
class Function(CustomStubBase):
    connections: Connections

class DockerImageFunction(CustomStubBase):
    connections: Connections

class SingletonFunction(CustomStubBase):
    connections: Connections

class Alias(CustomStubBase):
    connections: Connections

class Version(CustomStubBase):
    connections: Connections
