import requests
import boto3
from botocore.config import Config

client = boto3.client('s3')  # Noncompliant

def from_lambda_function():
    requests.post('https://api.example.com/data')  # Noncompliant 
    client

def lambda_handler(event, context):
    requests.get('https://api.example.com/data')  # Noncompliant {{Set an explicit timeout for this network call to prevent hanging executions in Lambda functions.}}
#   ^^^^^^^^^^^^
    requests.post('https://api.example.com/data')  # Noncompliant 
    requests.put('https://api.example.com/data')  # Noncompliant 
    requests.patch('https://api.example.com/data')  # Noncompliant 
    requests.request("POST", 'https://api.example.com/data')  # Noncompliant 

    session = requests.Session()
    session.get('https://api.example.com/data')  # Noncompliant 
#   ^^^^^^^^^^^
    session.post('https://api.example.com/data')  # Noncompliant 
    session.put('https://api.example.com/data')  # Noncompliant 
    session.patch('https://api.example.com/data')  # Noncompliant 
    session.request("POST", 'https://api.example.com/data')  # Noncompliant 


    requests.get('https://api.example.com/data', timeout=None)  # Noncompliant
    some_timeout = None
    requests.get('https://api.example.com/data', timeout=some_timeout)  # Noncompliant

    requests.get('https://api.example.com/data', timeout=5)  # Compliant
    requests.get('https://api.example.com/data', timeout=(5, 10))  # Compliant

    from_lambda_function()
    return {}

def regular_function():
    response = requests.get('https://api.example.com/data')  # Noncompliant 
    return response.json()

def boto3_lambda_handler(event, context):
    boto3.client('s3')  # Noncompliant
#   ^^^^^^^^^^^^
    boto3.resource('s3')  # Noncompliant
#   ^^^^^^^^^^^^^^

    config = Config(read_timeout=5, connect_timeout=5)
    only_read_timeout_config = Config(read_timeout=5)
    only_connect_timeout_config = Config(connect_timeout=5)
    empty_config = Config()

    boto3.client( # Noncompliant
        's3',
        None, # region_name
        None, # api_version
        None, # use_ssl
        None, # verify
        None, # endpoint_url
        None, # aws_access_key_id
        None, # aws_secret_access_key
        None, # aws_session_token
        empty_config # config
    )

    # Compliant
    boto3.client( 
        's3',
        None, # region_name
        None, # api_version
        None, # use_ssl
        None, # verify
        None, # endpoint_url
        None, # aws_access_key_id
        None, # aws_secret_access_key
        None, # aws_session_token
        config # config
    )

    boto3.resource( # Noncompliant
        's3',
        None, # region_name
        None, # api_version
        None, # use_ssl
        None, # verify
        None, # endpoint_url
        None, # aws_access_key_id
        None, # aws_secret_access_key
        None, # aws_session_token
        empty_config # config
    )

    # Compliant
    boto3.resource( 
        's3',
        None, # region_name
        None, # api_version
        None, # use_ssl
        None, # verify
        None, # endpoint_url
        None, # aws_access_key_id
        None, # aws_secret_access_key
        None, # aws_session_token
        config # config
    )

    boto3.client('s3', config=config)  # Compliant
    boto3.client('s3', config=only_read_timeout_config)  # Compliant
    boto3.client('s3', config=only_connect_timeout_config)  # Compliant

    boto3.resource('s3', config=config)  # Compliant
    boto3.resource('s3', config=only_read_timeout_config)  # Compliant
    boto3.resource('s3', config=only_connect_timeout_config)  # Compliant

    session = boto3.Session()
    session.client('s3', config=config)  # Compliant
    session.client('s3', config=only_read_timeout_config)  # Compliant
    session.client('s3', config=only_connect_timeout_config)  # Compliant

    session.resource('s3', config=config)  # Compliant
    session.resource('s3', config=only_read_timeout_config)  # Compliant
    session.resource('s3', config=only_connect_timeout_config)  # Compliant

    boto3.client('s3', config=Config(read_timeout=5, connect_timeout=5))  # Compliant
    boto3.client('s3', config=Config())  # Noncompliant

# COVERAGE
def coverage_lambda_handler(event, context):
    boto3.client('s3', config="") 
    boto3.client('s3', config=SomeOtherClass()) 
