import boto3

s3 = boto3.client('s3')
dynamodb = boto3.client('dynamodb')

def foo():
    response = s3.list_objects_v2(Bucket="my-bucket")  # Noncompliant
    response = dynamodb.scan(TableName="table")  # Noncompliant
