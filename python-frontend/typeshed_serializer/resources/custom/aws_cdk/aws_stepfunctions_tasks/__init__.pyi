from SonarPythonAnalyzerFakeStub import CustomStubBase
from aws_cdk.aws_ec2 import Connections

## constructs with connections attributes
class SageMakerCreateTrainingJob(CustomStubBase):
    connections: Connections

class SageMakerCreateModel(CustomStubBase):
    connections: Connections

class EcsRunTask(CustomStubBase):
    connections: Connections
