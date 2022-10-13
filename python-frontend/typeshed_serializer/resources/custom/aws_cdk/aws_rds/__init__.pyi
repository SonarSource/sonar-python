from SonarPythonAnalyzerFakeStub import CustomStubBase
from aws_cdk.aws_ec2 import Connections

## constructs with connections attributes
class DatabaseInstance(CustomStubBase):
    connections: Connections

class DatabaseInstanceReadReplica(CustomStubBase):
    connections: Connections

class DatabaseCluster(CustomStubBase):
    connections: Connections

class ServerlessClusterFromSnapshot(CustomStubBase):
    connections: Connections

class DatabaseProxy(CustomStubBase):
    connections: Connections

class DatabaseInstanceFromSnapshot(CustomStubBase):
    connections: Connections

class ServerlessCluster(CustomStubBase):
    connections: Connections

class DatabaseClusterFromSnapshot(CustomStubBase):
    connections: Connections
