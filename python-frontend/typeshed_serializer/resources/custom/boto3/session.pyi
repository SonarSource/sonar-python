from SonarPythonAnalyzerFakeStub import CustomStubBase

from botocore.client import BaseClient
from boto3.resources.base import ServiceResource

class Session(CustomStubBase):
  def client(self, *args, **kwargs) -> BaseClient: ...
  def resource(self, *args, **kwargs) -> ServiceResource: ...
