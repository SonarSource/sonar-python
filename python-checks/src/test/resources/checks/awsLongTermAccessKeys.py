import boto3
import os

def get_temporary_credentials() -> str:
    """Used for mocking the IAM role assuming part"""

    return "AKIAIOSFODNN7EXAMPLE"

def noncompliant_hardcoded_credentials():
    s3_client = boto3.client(
        's3',
        aws_access_key_id='AKIAIOSFODNN7EXAMPLE',  # Noncompliant
        aws_secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE'
    )

    s3_resource = boto3.resource(
        's3',
        aws_access_key_id='AKIAIOSFODNN7EXAMPLE',  # Noncompliant
        aws_secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE'
    )

    s3_client = boto3.client(
        's3',
        None, # region_name
        None, # api_version
        None, # use_ssl
        None, # verify
        None, # endpoint_url
        'AKIAIOSFODNN7EXAMPLE',  # Noncompliant
        'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE'
    )

    s3_resource = boto3.resource(
        's3',
        None, # region_name
        None, # api_version
        None, # use_ssl
        None, # verify
        None, # endpoint_url
        'AKIAIOSFODNN7EXAMPLE',  # Noncompliant
        'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE'
    )


def noncompliant_session_with_credentials():
    session = boto3.Session()
    ec2_from_session = session.client('ec2', 
        aws_access_key_id='AKIAIOSFODNN7EXAMPLE',  # Noncompliant
        aws_secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE'
    )
    s3_from_session = session.resource('s3', 
        aws_access_key_id='AKIAIOSFODNN7EXAMPLE',  # Noncompliant
        aws_secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE'
    )

    session = boto3.session.Session()
    ec2_from_session = session.client('ec2', 
        aws_access_key_id='AKIAIOSFODNN7EXAMPLE',  # Noncompliant
        aws_secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE'
    )

    session = boto3.Session(
        aws_access_key_id='AKIAIOSFODNN7EXAMPLE',  # Noncompliant
        aws_secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE'
    )

    session = boto3.Session(
        'AKIAIOSFODNN7EXAMPLE',  # Noncompliant
        'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE'
    )
    ec2_from_session = session.client('ec2', 
        None, # region_name
        None, # api_version
        None, # use_ssl
        None, # verify
        None, # endpoint_url
        'AKIAIOSFODNN7EXAMPLE',  # Noncompliant
        'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE'
    )
    s3_from_session = session.resource('s3', 
        None, # region_name
        None, # api_version
        None, # use_ssl
        None, # verify
        None, # endpoint_url
        'AKIAIOSFODNN7EXAMPLE',  # Noncompliant
        'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE'
    )

def noncompliant_variable_assignment():
    access_key = 'AKIAIOSFODNN7EXAMPLE'
    secret_key = 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE'
    lambda_client = boto3.client('lambda', 
        aws_access_key_id=access_key,  # Noncompliant
        aws_secret_access_key=secret_key
    )

def noncompliant_only_one_parameter():
    s3_client = boto3.client(
        's3',
        aws_access_key_id='AKIAIOSFODNN7EXAMPLE',  # Noncompliant {{Make sure using long-term access keys is safe here.}}
    )
    s3_client = boto3.client(
        's3',
        aws_secret_access_key='wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLE' # Noncompliant {{Make sure using long-term secret keys is safe here.}}
    )

  
def compliant_environment_credentials():
    s3_resource = boto3.resource(
        's3',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),  
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )

def compliant_iam_roles():
    s3_client_compliant = boto3.client('s3')
    s3_resource_compliant = boto3.resource('s3')
    ec2_client_compliant = boto3.client('ec2', region_name='us-east-1')

    session = boto3.Session()
    ec2_compliant = session.client('ec2')
    s3_compliant = session.resource('s3')

def compliant_temporary_credentials():
    sts_client = boto3.client('sts')
    assumed_role_object = sts_client.assume_role(
        RoleArn='arn:aws:iam::account-of-role-to-assume:role/name-of-role',
        RoleSessionName="AssumeRoleSession1"
    )
    credentials = assumed_role_object['Credentials']
    
    s3_client_temp = boto3.client(
        's3',
        aws_access_key_id=credentials['AccessKeyId'], 
        aws_secret_access_key=credentials['SecretAccessKey'],
        aws_session_token=credentials['SessionToken']
    )

def compliant_with_non_string_literals():
    key = get_temporary_credentials()
    s3_client = boto3.client(
        's3',
        aws_access_key_id=key, 
        aws_secret_access_key=key
    )

def compliant_unknown_variable(access_key, secret_key):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key, 
        aws_secret_access_key=secret_key
    )

