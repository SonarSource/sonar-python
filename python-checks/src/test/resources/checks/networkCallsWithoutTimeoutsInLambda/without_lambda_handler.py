import requests
import boto3
from botocore.config import Config

def a_function(event, context):
    requests.get('https://api.example.com/data') # Compliant
    boto3.client('s3')  # Compliant

