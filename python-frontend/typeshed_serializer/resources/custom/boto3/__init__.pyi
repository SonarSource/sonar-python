from botocore.client import BaseClient
from boto3.resources.base import ServiceResource
from boto3.session import Session

def client(*args, **kwargs) -> BaseClient: ...

def resource(*args, **kwargs) -> ServiceResource: ...
