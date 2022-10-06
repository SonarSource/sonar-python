from SonarPythonAnalyzerFakeStub import CustomStubBase
from aws_cdk.aws_ec2 import Connections

## constructs with connections attributes
class ExternalService(CustomStubBase):
    connections: Connections

class FargateService(CustomStubBase):
    connections: Connections

class Cluster(CustomStubBase):
    connections: Connections

class Ec2Service(CustomStubBase):
    connections: Connections
