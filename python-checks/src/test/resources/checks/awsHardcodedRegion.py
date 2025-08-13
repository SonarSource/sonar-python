import boto3
from boto3.session import Session
import os


def test_boto3_client():
    client = boto3.client('lambda', region_name='us-west-2')  # Noncompliant {{AWS region should not be set with a hardcoded String}}
    #                               ^^^^^^^^^^^^^^^^^^^^^^^

    client2 = boto3.client('lambda', 'us-west-2')  # Noncompliant

    my_region = 'us-west-2'
    client2 = boto3.client('lambda', region_name=my_region)  # Noncompliant

    my_region_from_os = os.environ["AWS_REGION"]
    client3 = boto3.client('lambda', region_name=my_region_from_os)  # Compliant

    client_without_region = boto3.client('lambda')  # Compliant
    client_without_region2 = boto3.client('lambda', other_argument='us-west-2')  # Compliant

    safe_method = boto3.safe(region_name='us-west-2')

def test_boto3_resource():
    s3_from_session = boto3.resource('s3', None)
    s3_from_session = boto3.resource('s3', 'us-west-2')  # Noncompliant
    s3_from_session = boto3.resource('s3', region_name='us-west-2')  # Noncompliant


def test_boto3_session():
    session = Session(None, None, None, None)
    session = Session(None, None, 'us-west-2', None)
    session = Session(None, None, None, 'us-west-2')  # Noncompliant
    session = Session(None, None, None, region_name='us-west-2')  # Noncompliant

    client_from_session = session.client(None, None)
    client_from_session = session.client(None, 'us-west-2')  # Noncompliant
    client_from_session = session.client(None, region_name='us-west-2')  # Noncompliant

    resource_from_session = session.resource(None, None)
    resource_from_session = session.resource(None, 'us-west-2')  # Noncompliant
    resource_from_session = session.resource(None, region_name='us-west-2')  # Noncompliant
