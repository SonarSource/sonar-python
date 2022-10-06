from SonarPythonAnalyzerFakeStub import CustomStubBase
from aws_cdk.aws_ec2 import Connections

## constructs with connections attributes
class DeliveryStream(CustomStubBase):
    connections: Connections
